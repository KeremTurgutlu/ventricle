from fastai.vision import *
from fastai import *
from fastai.data_block import *
from fastai.callbacks import *
from fastai.script import *
from fastai.distributed import *
from pathlib import Path

from unet import VolumetricUnet, dice_loss, dice_score, predict, plot_predictions, eval_preds

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import FloatTensor as FT

import sys
sys.path.extend(['../'])
from Vent_Seg_Preprocessing.helper import normalize, extract_brain
import numpy as np

train_raw_images_path = Path("/data/public/Segmentation_Dataset/MR_Dataset/train/raw_images")
validation_raw_images_path = Path("/data/public/Segmentation_Dataset/MR_Dataset/validation/raw_images")
test1_raw_images_path = Path("/data/public/Segmentation_Dataset/MR_Dataset/test1/raw_images")
test2_raw_images_path = Path("/data/public/Segmentation_Dataset/MR_Dataset/test2/raw_images")

data_path = Path("/data/public/Segmentation_Dataset/MR_Dataset")

trn_img = sorted(list(Path(data_path/"train/normalized_raw_images").iterdir()))
trn_mask = sorted(list(Path(data_path/"train/brain_masks").iterdir()))

valid_img = sorted(list(Path(data_path/"validation/normalized_raw_images").iterdir()))
valid_mask = sorted(list(Path(data_path/"validation/brain_masks").iterdir()))

test1_img = sorted(list(Path(data_path/"test1/normalized_raw_images").iterdir()))
test1_mask = sorted(list(Path(data_path/"test1/brain_masks").iterdir()))

test2_img = sorted(list(Path(data_path/"test2/normalized_raw_images").iterdir()))
test2_mask = sorted(list(Path(data_path/"test2/brain_masks").iterdir()))

set_of_trn_images = set([str(fp).split("/")[-1] for fp in trn_img])
set_of_trn_masks = set([str(fp).split("/")[-1] for fp in trn_mask])

set_of_valid_images = set([str(fp).split("/")[-1] for fp in valid_img])
set_of_valid_masks = set([str(fp).split("/")[-1] for fp in valid_mask])

set_of_test1_images = set([str(fp).split("/")[-1] for fp in test1_img])
set_of_test1_masks = set([str(fp).split("/")[-1] for fp in test1_mask])

set_of_test2_images = set([str(fp).split("/")[-1] for fp in test2_img])
set_of_test2_masks = set([str(fp).split("/")[-1] for fp in test2_mask])

assert set_of_trn_masks == set_of_trn_images
assert set_of_trn_images == set_of_trn_masks
assert set_of_valid_masks == set_of_valid_images
assert set_of_valid_images == set_of_valid_masks
assert set_of_test1_images == set_of_test1_masks
assert set_of_test2_images == set_of_test2_masks

print(f"Size of Train Dataset Images: {len(trn_img)} Images")
print(f"Size of Validation Dataset Images: {len(valid_img)} Images")
print(f"Size of Train Dataset Masks: {len(trn_mask)} Images")
print(f"Size of Validation Dataset Masks: {len(valid_mask)} Images")
print(f"Size of Test 1 Dataset Images: {len(test1_img)} Images")
print(f"Size of Test 1 Dataset Masks: {len(test1_mask)} Images")
print(f"Size of Test 2 Dataset Images: {len(test2_img)} Images")
print(f"Size of Test 2 Dataset Masks: {len(test2_mask)} Images")

for i in range(len(trn_img)):
    assert str(trn_img[i]).split("/")[-1] == str(trn_mask[i]).split("/")[-1]

for i in range(len(valid_img)):
    assert str(valid_img[i]).split("/")[-1] == str(valid_mask[i]).split("/")[-1]

for i in range(len(test1_img)):
    assert str(test1_img[i]).split("/")[-1] == str(test1_mask[i]).split("/")[-1]

for i in range(len(test2_img)):
    assert str(test2_img[i]).split("/")[-1] == str(test2_mask[i]).split("/")[-1]
    
    
class MRI_3D_Dataset(object):
    def __init__(self, img_fnames, ventricles, transform=None):
        self.img_fnames = img_fnames
        self.ventricles = ventricles
        self.transform = transform

    def __getitem__(self, index):
        image_voxel = np.load(self.img_fnames[index]).astype(np.float32)
        mask_voxel = np.load(self.ventricles[index]).astype(np.float32)
        if self.transform:
            image_voxel, mask_voxel = self.transform(image_voxel, mask_voxel)
        return FT(image_voxel[None,:]), FT(mask_voxel) 

    def __len__(self):
        return len(self.img_fnames)    

    
train_ds = MRI_3D_Dataset(trn_img, trn_mask)
valid_ds = MRI_3D_Dataset(valid_img, valid_mask)
test1_ds = MRI_3D_Dataset(test1_img, test1_mask)
test2_ds = MRI_3D_Dataset(test2_img, test2_mask)

from models import *

    
@call_parse
def main(gpu:Param("GPU to run on", str)=None):
    
    """Distrubuted training of a given experiment.
    Fastest speed is if you run as follows:
        python ../../fastai/fastai/launch.py --gpus=1234567 ./multi_gpu_training.py
       
    trains with effective batch size = n_gpu*bs 
    """
    
    # Init
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
    
    data = DataBunch.create(train_ds=train_ds, valid_ds=valid_ds, bs=1)
    
    # default
#     unet = VolumetricUnet2(in_c=1, out_c=4, n_layers=3, c=1, block_type=conv_relu_bn_drop,
#                            norm_type='group', actn='prelu', p=0.)

    # wider
#     unet = VolumetricUnet(in_c=1, out_c=8, n_layers=3, c=1, block_type=conv_relu_bn_drop,
#                            norm_type='group', actn='prelu', p=0.)

    # deeper
    unet = VolumetricUnet(in_c=1, out_c=8, n_layers=4, c=1, block_type=conv_relu_bn_drop,
                           norm_type='group', actn='prelu', p=0.)

    MODEL_NAME = 'MR_Brain_Scratch_Baseline_9'

    early_stop_cb = partial(EarlyStoppingCallback, monitor='dice_score', mode='max', patience=5)

    save_model_cb = partial(SaveModelCallback, monitor='dice_score', mode='max', every='improvement',
                            name=f'best_of_{MODEL_NAME}')

    reduce_lr_cb = partial(ReduceLROnPlateauCallback, monitor='dice_score', mode='max', patience=0, factor=0.8)

    csv_logger_cb = partial(CSVLogger, filename=f'logs/{MODEL_NAME}')

    callback_fns = [early_stop_cb, save_model_cb, reduce_lr_cb, csv_logger_cb]
    callbacks = [TerminateOnNaNCallback()]

    learn = Learner(data=data, model=unet, callbacks=callbacks, callback_fns=callback_fns)
    learn.loss_func = dice_loss
    learn.metrics = [dice_score]
    

    learn.to_fp16()
    learn.to_distributed(gpu)
    learn.fit_one_cycle(100, max_lr=5e-1)
    
    
