import copy
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
    scans = Categorizer.categorize_files_by_scan("/data/public/Testing_CT_Raw_DICOM_Files")
    
    mishaped = []
    errors = []
    i = 1
    
    for scan in scans:
        try:
            print(f"{i}/{len(scans)}")
            medim_obj = MedImage.create_from_files(scan)
            file_name = medim_obj.name.split("/")[-1]
      
            image, brain_mask = medim_obj.compute_mask("{Brain}", "full", interp=True)
            _, ventricle_mask = medim_obj.compute_mask("{Ventricles}", "full", interp=True)
            _, brain_atlas = medim_obj.compute_mask("{BrainAtlas}", "full", interp=True)
            _, ventricle_atlas = medim_obj.compute_mask("{VentriclesAtlas}", "full", interp=True)
            
    
            raw_image = copy.deepcopy(image)
            raw_image = helper.z_pad(raw_image)
            raw_image = helper.xy_pad(raw_image)


            image.data = helper.normalize(image.data, brain_mask.data)
            image.data = helper.extract_brain(image.data, brain_mask.data)
            skull_stripped = helper.z_pad(image)
            skull_stripped = helper.xy_pad(skull_stripped)


            ventricle_mask = helper.z_pad(ventricle_mask)
            ventricle_mask = helper.xy_pad(ventricle_mask)

            brain_mask = helper.z_pad(brain_mask)
            brain_mask = helper.xy_pad(brain_mask)

            ventricle_atlas = helper.z_pad(ventricle_atlas)
            ventricle_atlas = helper.xy_pad(ventricle_atlas)

            brain_atlas = helper.z_pad(brain_atlas)
            brain_atlas = helper.xy_pad(brain_atlas)

       
            path = "/data/public/Segmentation_Dataset/CT_Dataset/test2"
            np.save("{p}/raw_images/{n}".format(n=file_name, p=path), raw_image.data)
            np.save("{p}/skull_stripped/{n}".format(n=file_name, p=path), skull_stripped.data)
            np.save("{p}/brain_masks/{n}".format(n=file_name, p=path), brain_mask.data)
            np.save("{p}/brain_atlas/{n}".format(n=file_name, p=path), brain_atlas.data)
            np.save("{p}/ventricle_masks/{n}".format(n=file_name, p=path), ventricle_mask.data)
            np.save("{p}/ventricle_atlas/{n}".format(n=file_name, p=path), ventricle_atlas.data)

            if raw_image.data.shape != (128, 256, 256):
                mishaped.append(file_name)
            i += 1
        except:
            errors.append(file_name)
            i += 1
            

np.save("/data/public/Segmentation_Dataset/CT_Dataset/mishaped_Testing_CT.npy",np.array(mishaped))
np.save("/data/public/Segmentation_Dataset/CT_Dataset/errors_Testing_CT.npy", np.array(errors))
            