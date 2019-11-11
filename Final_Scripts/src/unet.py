import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor as FT
#import cv2
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


def predict(learner, data, validation=True, thresh=0.5):
    model = learner.model.eval()
    preds = []
    images = []
    masks = []
    dl = None
    if validation:
        dl = data.valid_dl
    else:
        dl = data.test_dl
    for image, mask in dl:
        out = model(image.half().cuda())
        out = torch.sigmoid(out).cpu().data.numpy()
        out = out.astype(float)
        out = (out > thresh)*1
        image = image.cpu().data.numpy()
        mask = mask.cpu().data.numpy()
        for i in range(out.shape[0]):
            masks.append(mask[i])
            preds.append(out[i][0])
            images.append(image[i][0])
    return images, preds, masks

def plot_predictions(true, pred, mask):
    """draw image, pred, mask side by side"""
    fig, ax = plt.subplots(1,3, figsize=(20,10))
    axes = ax.flatten()
    for ax, im, t in zip(axes, [true, pred, mask], ["image", "pred", "mask"]) :
        ax.imshow(im, cmap="gray")
        ax.set_title(t, fontdict={"fontsize":20})

def dice_score(logits, targets, thresh=0.5):
    hard_preds = (torch.sigmoid(logits) > thresh).float()
    m1 = hard_preds.view(-1)  # Flatten
    m2 = targets.view(-1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection) / (m1.sum() + m2.sum() + 1e-6)

def eval_preds(preds, target, thresh=0.5):
    iflat = np.array(preds).reshape(-1)
    tflat = np.array(target).reshape(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection) / (iflat.sum() + tflat.sum() + 1e-6))

def dice_loss(logits, target):
    logits = torch.sigmoid(logits)
    smooth = 1.0
    iflat = logits.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def conv_bn_relu(in_channel, out_channel):
    block = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channel),
        nn.ReLU()
    )
    return block


def maxpool3D():
    block = nn.MaxPool3d(2, stride=2)
    return block


def one_by_one_conv(in_channel, out_channel):
    block = nn.Conv3d(in_channel, out_channel, 1)
    return block


class VolumetricUnet(nn.Module):
    def __init__(self, in_channel, out_channel, num_classes):
        super(VolumetricUnet, self).__init__()

        # inits
        self.conv1 = conv_bn_relu(in_channel, out_channel)
        self.conv2 = conv_bn_relu(out_channel, out_channel * 2)
        self.pool = maxpool3D()  # D/2, H/2, W/2

        self.conv3 = conv_bn_relu(out_channel * 2, out_channel * 2)
        self.conv4 = conv_bn_relu(out_channel * 2, out_channel * 4)

        self.conv5 = conv_bn_relu(out_channel * 4, out_channel * 4)
        self.conv6 = conv_bn_relu(out_channel * 4, out_channel * 8)

        self.conv7 = conv_bn_relu(out_channel * 8, out_channel * 8)
        self.conv8 = conv_bn_relu(out_channel * 8, out_channel * 16)

        self.conv9 = conv_bn_relu(out_channel * 8 + out_channel * 16, out_channel * 8)
        self.conv10 = conv_bn_relu(out_channel * 8, out_channel * 8)

        self.conv11 = conv_bn_relu(out_channel * 4 + out_channel * 8, out_channel * 4)
        self.conv12 = conv_bn_relu(out_channel * 4, out_channel * 4)

        self.conv13 = conv_bn_relu(out_channel * 2 + out_channel * 4, out_channel * 2)
        self.conv14 = conv_bn_relu(out_channel * 2, out_channel * 2)

        self.one_by_one = one_by_one_conv(out_channel * 2, num_classes)

    def forward(self, x):
        out1 = self.conv2(self.conv1(x))
        down1 = self.pool(out1)

        out2 = self.conv4(self.conv3(down1))
        down2 = self.pool(out2)

        out3 = self.conv6(self.conv5(down2))
        down3 = self.pool(out3)

        middle = self.conv8(self.conv7(down3))


        up1 = F.interpolate(middle, scale_factor=2, mode='trilinear', align_corners=True)
        cat1 = torch.cat([out3, up1], dim=1)

        out4 = self.conv10(self.conv9(cat1))
        up2 = F.interpolate(out4, scale_factor=2, mode='trilinear', align_corners=True)
        cat2 = torch.cat([out2, up2], dim=1)

        out5 = self.conv12(self.conv11(cat2))
        up3 = F.interpolate(out5, scale_factor=2, mode='trilinear', align_corners=True)
        cat3 = torch.cat([out1, up3], dim=1)

        out6 = self.conv14(self.conv13(cat3))
        final_out = self.one_by_one(out6)

        return final_out
    
    

#transforms here
def flip(x, p=0.5):
    """flips each slice  along horizontal axis"""
    if np.random.random() < p:
        x = np.flip(x, axis=0)
    return x

def crop_zoom(image, mask, p=0.3, zoom=(1, 1.3, 1.3)):
    if np.random.random() < p:
        image = ndimage.zoom(image, zoom)
        mask = ndimage.zoom(mask, zoom)
        _, x, y = image.shape
        cropx, cropy = (x-256)//2, (y-256)//2
        image = image[:,cropx+1:x-cropx, cropy+1:y-cropy].copy()
        mask = (mask[:,cropx+1:x-cropx, cropy+1:y-cropy].copy() > 0.5).astype(float)
    return image, mask

def random_rotate_3D_transform2(image_voxel, mask_voxel, angle=30, p=0.3):
    """rotate by +-angle"""
    H, W = mask_voxel.shape[-1], mask_voxel.shape[-2]
    if np.random.random() < p:
        angle = np.random.randint(-angle, angle, size=1)
        M = cv2.getRotationMatrix2D((H / 2, W / 2), angle, 1)
        image_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in image_voxel])
        mask_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in mask_voxel])
    return image_voxel, mask_voxel

def rotate_and_zoom(image_voxel, mask_voxel, angle, p):
    image_voxel, mask_voxel = random_rotate_3D_transform2(image_voxel, mask_voxel, angle=angle, p=p)
    mask_voxel = (mask_voxel > 0)*1
    image_voxel, mask_voxel = crop_zoom(image_voxel, mask_voxel, p=p)
    return image_voxel, mask_voxel

class Transformer:
    def __init__(self, angle, p):
        self.p = p 
        self.angle = angle
    
    def transform(self, image_voxel, mask_voxel):
        return rotate_and_zoom(image_voxel, mask_voxel, angle=self.angle, p=self.p)