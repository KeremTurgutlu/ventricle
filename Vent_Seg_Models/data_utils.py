from fastai.vision import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import FloatTensor as FT
import sys
sys.path.extend(['../'])
from Vent_Seg_Preprocessing.helper import normalize, extract_brain


__all__ = ['MRI_3D_Dataset', 'data_dict', 'tl_brain_mr_model_dict']


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
    
##########################
#### BRAIN DATA ###
#########################

def get_notl_brain_mr_data():
    
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

    for i in range(len(trn_img)): assert str(trn_img[i]).split("/")[-1] == str(trn_mask[i]).split("/")[-1]
    for i in range(len(valid_img)): assert str(valid_img[i]).split("/")[-1] == str(valid_mask[i]).split("/")[-1]
    for i in range(len(test1_img)): assert str(test1_img[i]).split("/")[-1] == str(test1_mask[i]).split("/")[-1]
    for i in range(len(test2_img)): assert str(test2_img[i]).split("/")[-1] == str(test2_mask[i]).split("/")[-1]

    return (trn_img, trn_mask), (valid_img, valid_mask), (test1_img, test1_mask), (test2_img, test2_mask)


def get_notl_brain_ct_data():

    data_path = Path("/data/public/Segmentation_Dataset/CT_Dataset")
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
    
    for i in range(len(trn_img)): assert str(trn_img[i]).split("/")[-1] == str(trn_mask[i]).split("/")[-1]
    for i in range(len(valid_img)): assert str(valid_img[i]).split("/")[-1] == str(valid_mask[i]).split("/")[-1]
    for i in range(len(test1_img)): assert str(test1_img[i]).split("/")[-1] == str(test1_mask[i]).split("/")[-1]
    for i in range(len(test2_img)): assert str(test2_img[i]).split("/")[-1] == str(test2_mask[i]).split("/")[-1]

    return (trn_img, trn_mask), (valid_img, valid_mask), (test1_img, test1_mask), (test2_img, test2_mask)


def get_atlas_brain_mr_data():

    data_path = Path("/data/public/Segmentation_Dataset/MR_Dataset_Atlas")
    trn_img = sorted(list(Path(data_path/"train/normalized_raw_images").iterdir()))
    trn_mask = sorted(list(Path(data_path/"train/brain_atlas").iterdir()))
    valid_img = sorted(list(Path(data_path/"validation/normalized_raw_images").iterdir()))
    valid_mask = sorted(list(Path(data_path/"validation/brain_masks").iterdir()))

    set_of_trn_images = set([str(fp).split("/")[-1] for fp in trn_img])
    set_of_trn_masks = set([str(fp).split("/")[-1] for fp in trn_mask])
    set_of_valid_images = set([str(fp).split("/")[-1] for fp in valid_img])
    set_of_valid_masks = set([str(fp).split("/")[-1] for fp in valid_mask])
    trn_img = list(filter(lambda x: str(x).split("/")[-1] in set_of_trn_masks, trn_img))
    
    for i in range(len(trn_img)): assert str(trn_img[i]).split("/")[-1] == str(trn_mask[i]).split("/")[-1]
    for i in range(len(valid_img)): assert str(valid_img[i]).split("/")[-1] == str(valid_mask[i]).split("/")[-1]
    
    return (trn_img, trn_mask), (valid_img, valid_mask), None, None


##########################
#### VENTRICLE DATA ###
#########################

def get_notl_ventricle_mr_data():
    
    data_path = Path("../../data/Segmentation_Dataset/MR_Dataset")

    trn_img = sorted(list(Path(data_path/"train/skull_stripped_v2").iterdir()))
    trn_mask = sorted(list(Path(data_path/"train/ventricle_masks").iterdir()))

    valid_img = sorted(list(Path(data_path/"validation/skull_stripped_v2").iterdir()))
    valid_mask = sorted(list(Path(data_path/"validation/ventricle_masks").iterdir()))

    test1_img = sorted(list(Path(data_path/"test1/skull_stripped_v2").iterdir()))
    test1_mask = sorted(list(Path(data_path/"test1/ventricle_masks").iterdir()))

    test2_img = sorted(list(Path(data_path/"test2/skull_stripped_v2").iterdir()))
    test2_mask = sorted(list(Path(data_path/"test2/ventricle_masks").iterdir()))

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

    for i in range(len(trn_img)): assert str(trn_img[i]).split("/")[-1] == str(trn_mask[i]).split("/")[-1]
    for i in range(len(valid_img)): assert str(valid_img[i]).split("/")[-1] == str(valid_mask[i]).split("/")[-1]
    for i in range(len(test1_img)): assert str(test1_img[i]).split("/")[-1] == str(test1_mask[i]).split("/")[-1]
    for i in range(len(test2_img)): assert str(test2_img[i]).split("/")[-1] == str(test2_mask[i]).split("/")[-1]

    return (trn_img, trn_mask), (valid_img, valid_mask), (test1_img, test1_mask), (test2_img, test2_mask)

def get_notl_ventricle_ct_data():

    data_path = Path("../../data/Segmentation_Dataset/CT_Dataset")
    trn_img = sorted(list(Path(data_path/"train/skull_stripped_v2").iterdir()))
    trn_mask = sorted(list(Path(data_path/"train/ventricle_masks").iterdir()))

    valid_img = sorted(list(Path(data_path/"validation/skull_stripped_v2").iterdir()))
    valid_mask = sorted(list(Path(data_path/"validation/ventricle_masks").iterdir()))

    test1_img = sorted(list(Path(data_path/"test1/skull_stripped_v2").iterdir()))
    test1_mask = sorted(list(Path(data_path/"test1/ventricle_masks").iterdir()))

    test2_img = sorted(list(Path(data_path/"test2/skull_stripped_v2").iterdir()))
    test2_mask = sorted(list(Path(data_path/"test2/ventricle_masks").iterdir()))

    set_of_trn_images = set([str(fp).split("/")[-1] for fp in trn_img])
    set_of_trn_masks = set([str(fp).split("/")[-1] for fp in trn_mask])
    set_of_valid_images = set([str(fp).split("/")[-1] for fp in valid_img])
    set_of_valid_masks = set([str(fp).split("/")[-1] for fp in valid_mask])
    set_of_test1_images = set([str(fp).split("/")[-1] for fp in test1_img])
    set_of_test1_masks = set([str(fp).split("/")[-1] for fp in test1_mask])
    set_of_test2_images = set([str(fp).split("/")[-1] for fp in test2_img])
    set_of_test2_masks = set([str(fp).split("/")[-1] for fp in test2_mask])
    
    for i in range(len(trn_img)): assert str(trn_img[i]).split("/")[-1] == str(trn_mask[i]).split("/")[-1]
    for i in range(len(valid_img)): assert str(valid_img[i]).split("/")[-1] == str(valid_mask[i]).split("/")[-1]
    for i in range(len(test1_img)): assert str(test1_img[i]).split("/")[-1] == str(test1_mask[i]).split("/")[-1]
    for i in range(len(test2_img)): assert str(test2_img[i]).split("/")[-1] == str(test2_mask[i]).split("/")[-1]

    return (trn_img, trn_mask), (valid_img, valid_mask), (test1_img, test1_mask), (test2_img, test2_mask)

def get_atlas_ventricle_mr_data():

    data_path = Path("../../data/Segmentation_Dataset/MR_Dataset_Atlas")
    trn_img = sorted(list(Path(data_path/"train/skull_stripped_v2").iterdir()))
    trn_mask = sorted(list(Path(data_path/"train/ventricle_atlas").iterdir()))
    valid_img = sorted(list(Path(data_path/"validation/skull_stripped_v2").iterdir()))
    valid_mask = sorted(list(Path(data_path/"validation/ventricle_masks").iterdir()))

    set_of_trn_images = set([str(fp).split("/")[-1] for fp in trn_img])
    set_of_trn_masks = set([str(fp).split("/")[-1] for fp in trn_mask])
    set_of_valid_images = set([str(fp).split("/")[-1] for fp in valid_img])
    set_of_valid_masks = set([str(fp).split("/")[-1] for fp in valid_mask])
    trn_img = list(filter(lambda x: str(x).split("/")[-1] in set_of_trn_masks, trn_img))
    
    for i in range(len(trn_img)): assert str(trn_img[i]).split("/")[-1] == str(trn_mask[i]).split("/")[-1]
    for i in range(len(valid_img)): assert str(valid_img[i]).split("/")[-1] == str(valid_mask[i]).split("/")[-1]
    
    return (trn_img, trn_mask), (valid_img, valid_mask), None, None


data_dict = {
    'notl_brain_mr': get_notl_brain_mr_data,
    'notl_brain_ct': get_notl_brain_ct_data,
    'atlas_brain_mr': get_atlas_brain_mr_data,
    'notl_ventricle_mr': get_notl_ventricle_mr_data,
    'notl_ventricle_ct': get_notl_ventricle_ct_data,
    'atlas_ventricle_mr': get_atlas_ventricle_mr_data,
}   

##########################
#### TRANSFER LEARNING ###
#########################
 
tl_brain_mr_model_dict = {
    'TL_Brain_MR_Baseline_1': 'best_of_ATLAS_Brain_MR_Baseline_1',
    'TL_Brain_MR_Baseline_2': 'best_of_ATLAS_Brain_MR_Baseline_2',
    'TL_Brain_MR_Baseline_3': 'best_of_ATLAS_Brain_MR_Baseline_3',
    'TL_Brain_MR_Baseline_4': 'best_of_ATLAS_Brain_MR_Baseline_4',
    'TL_Brain_MR_Baseline_5': 'best_of_ATLAS_Brain_MR_Baseline_5',
    'TL_Brain_MR_Baseline_6': 'best_of_ATLAS_Brain_MR_Baseline_6',
    'TL_Brain_MR_Baseline_7': 'best_of_ATLAS_Brain_MR_Baseline_7',
    'TL_Brain_MR_Baseline_8': 'best_of_ATLAS_Brain_MR_Baseline_8',
    'TL_Brain_MR_Baseline_9': 'best_of_ATLAS_Brain_MR_Baseline_9',
    'TL_Brain_MR_Baseline_10': 'best_of_ATLAS_Brain_MR_Baseline_10',
    'TL_Brain_MR_Baseline_11': 'best_of_ATLAS_Brain_MR_Baseline_11'
}

tl_brain_ct_model_dict = {
    'TL_Brain_CT_Baseline_1': 'best_of_ATLAS_Brain_MR_Baseline_1',
    'TL_Brain_CT_Baseline_2': 'best_of_ATLAS_Brain_MR_Baseline_2',
    'TL_Brain_CT_Baseline_3': 'best_of_ATLAS_Brain_MR_Baseline_3',
    'TL_Brain_CT_Baseline_4': 'best_of_ATLAS_Brain_MR_Baseline_4',
    'TL_Brain_CT_Baseline_5': 'best_of_ATLAS_Brain_MR_Baseline_5',
    'TL_Brain_CT_Baseline_6': 'best_of_ATLAS_Brain_MR_Baseline_6',
    'TL_Brain_CT_Baseline_7': 'best_of_ATLAS_Brain_MR_Baseline_7',
    'TL_Brain_CT_Baseline_8': 'best_of_ATLAS_Brain_MR_Baseline_8',
    'TL_Brain_CT_Baseline_9': 'best_of_ATLAS_Brain_MR_Baseline_9',
    'TL_Brain_CT_Baseline_10': 'best_of_ATLAS_Brain_MR_Baseline_10',
    'TL_Brain_CT_Baseline_11': 'best_of_ATLAS_Brain_MR_Baseline_11'
}





















