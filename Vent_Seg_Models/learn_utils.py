import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor as FT
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
from tqdm import tqdm_notebook
from fastai.callbacks import *
from ipyexperiments import IPyExperimentsPytorch

__all__ = ['get_img_pred_masks', 'plot_predictions', 'dice_score', 'dice_loss', 'MixedLoss',
           'CatchNanGrad', 'CatchNanActs', 'CatchNanParameters', 'lsuv_init', 'cond_init']

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
gnorm_types = (nn.GroupNorm,)
insnorm_types = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
norm_types = bn_types + gnorm_types + insnorm_types

def cond_init(m:nn.Module, init_func:LayerFunc):
    "Initialize the non-batchnorm layers of `m` with `init_func`."
    if (not isinstance(m, norm_types)) and (not isinstance(m, nn.PReLU)) and requires_grad(m): 
        init_default(m, init_func)
        
class BackwardHookCallback(LearnerCallback):
    "Callback that can be used to register hooks on `modules`. Implement the corresponding function in `self.hook`."
    def __init__(self, learn:Learner, modules:Sequence[nn.Module]=None, do_remove:bool=True):
        super().__init__(learn)
        self.modules,self.do_remove = modules,do_remove

    def on_train_begin(self, **kwargs):
        "Register the `Hooks` on `self.modules`."
        if not self.modules:
            self.modules = [m for m in flatten_model(self.learn.model)
                            if hasattr(m, 'weight')]
        # needs to be is_forward=False, detach=False
        self.hooks = Hooks(self.modules, self.hook, is_forward=False, detach=False)

    def on_train_end(self, **kwargs):
        "Remove the `Hooks`."
        if self.do_remove: self.remove()

    def remove(self): 
        if getattr(self, 'hooks', None): self.hooks.remove()
    
    def __del__(self): self.remove()
        
        
class CatchNanGrad(BackwardHookCallback):
    "Catch NaN when first appears in grad"

    def on_train_begin(self, **kwargs):
        super().on_train_begin(**kwargs)
        self.stop = False

    def hook(self, m:nn.Module, i:Tensors, o:Tensors)->Tuple[Rank0Tensor,Rank0Tensor]:
        "Take the mean and std of `o`."
        if (not self.stop) and (torch.any(torch.isnan(o[0]))): 
            self.stop = True
            print(m,o)
            return (m, o)
    
    def on_backward_end(self, train, epoch, num_batch, **kwargs):
        "Called after backprop but before optimizer step."
        if train and self.stop: 
            print (f'Epoch/Batch ({epoch}/{num_batch}): Invalid Grad, terminating training.')
            return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}
        
class CatchNanActs(HookCallback):
    "Catch NaN when first appears in acts"

    def on_train_begin(self, **kwargs):
        super().on_train_begin(**kwargs)
        self.stop = False

    def hook(self, m:nn.Module, i:Tensors, o:Tensors)->Tuple[Rank0Tensor,Rank0Tensor]:
        "Take the mean and std of `o`."
        if (not self.stop) and (torch.any(torch.isnan(o[0]))): 
            self.stop = True
            print(m,o)
            return (m, o)
            
    def on_loss_begin(self, train, epoch, num_batch, **kwargs):
        "Called after forward pass but before loss has been computed."
        if train and self.stop: 
            print (f'Epoch/Batch ({epoch}/{num_batch}): Invalid Activation, terminating training.')
            return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}
        
class CatchNanParameters(LearnerCallback):
    "Catch NaN when first appears in parameter weight"
    
    def on_train_begin(self, **kwargs):
        super().on_train_begin(**kwargs)
        self.modules = [m for m in flatten_model(self.learn.model) if hasattr(m, 'weight')]
        
    def on_backward_end(self, train, epoch, num_batch, **kwargs):
        for m in self.modules: 
            if m.weight is not None:
                if torch.any(torch.isnan(m.weight)): 
                    print (f'Epoch/Batch ({epoch}/{num_batch}): Invalid Parameter, terminating training.')
                    print(m,o)
                    return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}
                
class ActStats:
    def __init__(self):
        pass
    def __call__(self, m, i, o):
        d = o.data
        self.mean,self.std = d.mean().item(),d.std().item()

def lsuv_module(m, model, xb):
    stats = ActStats()
    h = Hook(m, stats)

    if hasattr(m, 'bias'): 
        while model(xb) is not None and abs(stats.mean)  > 1e-3: m.bias -= stats.mean
    if hasattr(m, 'weight'):
        while model(xb) is not None and abs(stats.std-1) > 1e-3: m.weight.data /= stats.std

    h.remove()
    return stats.mean, stats.std

def lsuv_init(learn):
    "initialize model parameters with LSUV - https://arxiv.org/abs/1511.06422"
    modules = [m for m in flatten_model(learn.model) if hasattr(m, 'weight') and m.weight is not None]
    mdl = learn.model.cuda()
    xb, yb = learn.data.one_batch()
    for m in modules: print(lsuv_module(m, learn.model, xb.cuda()))
    for m in modules: 
        if hasattr(m, 'weight'): assert not torch.any(torch.isnan(m.weight))
        if hasattr(m, 'bias'): assert not torch.any(torch.isnan(m.bias))
    del modules
    del mdl
    del xb
    del yb
    gc.collect()
            
def get_img_pred_masks(learner, dl, thresh=0.5):
    model = learner.model.eval()
    preds = []
    images = []
    masks = []
    for image, mask in tqdm_notebook(dl):
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
    hard_preds = torch.sigmoid(logits) > thresh
    m1 = hard_preds.view(-1).float() 
    m2 = targets.view(-1).float()
    intersection = (m1 * m2).sum()
    return (2. * intersection) / (m1.sum() + m2.sum() + 1e-6)

def eval_preds(preds, target, thresh=0.5):
    iflat = np.array(preds).reshape(-1)
    tflat = np.array(target).reshape(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection) / (iflat.sum() + tflat.sum() + 1e-6))

def dice_loss(logits, target, smooth=1.):
    logits = torch.sigmoid(logits)
    iflat = logits.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, logits, target):
        logits = logits.squeeze(1)
        if not (target.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), logits.size()))

        max_val = (-logits).clamp(min=0)
        loss = logits - logits * target + max_val + \
            ((-max_val).exp() + (-logits - max_val).exp()).log()

        invprobs = F.logsigmoid(-logits * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()
    
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(1 - dice_loss(input, target))
        return loss.mean()    
    
    
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

# def random_rotate_3D_transform2(image_voxel, mask_voxel, angle=30, p=0.3):
#     """rotate by +-angle"""
#     H, W = mask_voxel.shape[-1], mask_voxel.shape[-2]
#     if np.random.random() < p:
#         angle = np.random.randint(-angle, angle, size=1)
#         M = cv2.getRotationMatrix2D((H / 2, W / 2), angle, 1)
#         image_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in image_voxel])
#         mask_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in mask_voxel])
#     return image_voxel, mask_voxel

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
    
    
## visualize model with netron
# import torch.onnx
# dummy_input = torch.randn(1,1,128,256,256).half().cuda()

# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]
# torch.onnx.export(learn.model, dummy_input, "/tmp/learn_model.onnx",
#                   verbose=False, input_names=input_names, output_names=output_names)

# import netron

# netron.start('atlas_brain_mr_models/best_of_ATLAS_Brain_MR_Baseline_1.pth', port=41501, host='localhost')