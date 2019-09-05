from fastai.vision import *
from fastai import *
from fastai.data_block import *

from pathlib import Path

from unet import VolumetricUnet, dice_loss, dice_score

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import FloatTensor as FT

torch.cuda.set_device(0)

unet = VolumetricUnet(in_channel=1, out_channel=4, num_classes=1)

data_path = Path("/data/public/Segmentation_Dataset/MR_Dataset_Atlas")
trn_img = sorted(list(Path(data_path/"train/skull_stripped").iterdir()))
trn_mask = sorted(list(Path(data_path/"train/ventricle_atlas").iterdir()))
valid_img = sorted(list(Path(data_path/"validation/skull_stripped").iterdir()))
valid_mask = sorted(list(Path(data_path/"validation/ventricle_masks").iterdir()))

print(f"Size of Train Dataset: {len(trn_img)} Images")
print(f"Size of Validation Dataset: {len(valid_img)} Images")


class MRI_3D_Dataset(object):
    def __init__(self, images, ventricles, transform=None):
        self.images = images
        self.ventricles = ventricles
        self.transform = transform
    
    def __getitem__(self, index):
        image_voxel = np.load(self.images[index]).astype(np.float32)
        mask_voxel = np.load(self.ventricles[index]).astype(np.float32)
        if self.transform:
            image_voxel, mask_voxel = self.transform(image_voxel, mask_voxel)
        return FT(image_voxel[None,:]), FT(mask_voxel) 

    def __len__(self):
        return len(self.images)
    
train_ds = MRI_3D_Dataset(trn_img, trn_mask)
valid_ds = MRI_3D_Dataset(valid_img, valid_mask)   

data = DataBunch.create(train_ds=train_ds, valid_ds=valid_ds, bs=3)

learner = Learner(data=data, model=unet)
learner.loss_func = dice_loss
learner.metrics = [dice_score]
learner.to_fp16()

for i in range(len(trn_img)):
    assert str(trn_img[i]).split("/")[-1] == str(trn_mask[i]).split("/")[-1]

for i in range(len(valid_img)):
    assert str(valid_img[i]).split("/")[-1] == str(valid_mask[i]).split("/")[-1]
    
#learner.fit_one_cycle(5, 3e-2)
#learner.save('DL_Ventricle_MR_ATLAS_5_epochs_3e-2_lr')

learner.load('DL_Ventricle_MR_ATLAS_5_epochs_3e-2_lr')
learner.fit_one_cycle(1, 3e-2)
learner.save('DL_Ventricle_MR_ATLAS_6_epochs_3e-2_lr')
learner.fit_one_cycle(1, 3e-2)
learner.save('DL_Ventricle_MR_ATLAS_7_epochs_3e-2_lr')
learner.fit_one_cycle(1, 3e-2)
learner.save('DL_Ventricle_MR_ATLAS_8_epochs_3e-2_lr')
learner.fit_one_cycle(1, 3e-2)
learner.save('DL_Ventricle_MR_ATLAS_9_epochs_3e-2_lr')
learner.fit_one_cycle(1, 3e-2)
learner.save('DL_Ventricle_MR_ATLAS_10_epochs_3e-2_lr')


