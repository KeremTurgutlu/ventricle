import copy
import math
import sys
sys.path.extend(['/home/dalala/PiCare-develop/Code'])

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import helper
from medimage import MedImage
from file_categorizer import Categorizer


if __name__ == '__main__':
    print("Gathering DICOM scans")
    scans = Categorizer.categorize_files_by_scan("/data/public/PICARE_BMETS_Raw_DICOM_Files")
    
    mishaped = []
    errors = []
    i = 1
    
    for scan in scans:
        try:
            print(f"{i}/{len(scans)}")
            medim_obj = MedImage.create_from_files(scan)
            file_name = medim_obj.name.split("/")[-1]
            
            image, brain_atlas = medim_obj.compute_mask("{Brain}", "full", interp=True)
            _, ventricle_atlas = medim_obj.compute_mask("{Ventricles}", "full", interp=True)
            
            raw_image = copy.deepcopy(image)
            raw_image = helper.z_pad(raw_image)
            raw_image = helper.xy_pad(raw_image)
            np.save("/data/public/Segmentation_Dataset/MR_Dataset_Atlas/train/raw_images/{name}".format(name=file_name), raw_image.data)
            
            image.data = helper.normalize(image.data, brain_atlas.data)
            image.data = helper.extract_brain(image.data, brain_atlas.data)
            skull_stripped = helper.z_pad(image)
            skull_stripped = helper.xy_pad(skull_stripped)
            
            np.save("/data/public/Segmentation_Dataset/MR_Dataset_Atlas/train/skull_stripped/{name}".format(name=file_name), skull_stripped.data)
            
            ventricle_atlas = helper.z_pad(ventricle_atlas)
            ventricle_atlas = helper.xy_pad(ventricle_atlas)
            
            np.save("/data/public/Segmentation_Dataset/MR_Dataset_Atlas/train/ventricle_atlas/{name}".format(name=file_name), ventricle_atlas.data)
            
            brain_atlas = helper.z_pad(brain_atlas)
            brain_atlas = helper.xy_pad(brain_atlas)
            
            np.save("/data/public/Segmentation_Dataset/MR_Dataset_Atlas/train/brain_atlas/{name}".format(name=file_name), brain_atlas.data)
            
            if raw_image.data.shape != (128, 256, 256):
                mishaped.append(file_name)
            i += 1
        except:
            errors.append(file_name)
            i += 1
            

    np.save("/data/public/Segmentation_Dataset/MR_Dataset_Atlas/mishaped.npy",np.array(mishaped))
    np.save("/data/public/Segmentation_Dataset/MR_Dataset_Atlas/errors.npy", np.array(errors))
            