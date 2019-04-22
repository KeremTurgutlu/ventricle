import copy
import sys
sys.path.extend(['/home/dalala/PiCare-develop/Code'])

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import helper
from medimage import MedImage
from file_categorizer import Categorizer

MR_Val = {'ANON61382','ANON55375','ANON85534','ANON54218','ANON24182','ANON14135','ANON49037','ANON66932','ANON10465',
       'ANON39801','ANON14447','ANON42229','ANON99458','ANON36946', 'ANON16732'}

MR_test_1 = {'ANON78381', 'ANON38662','ANON78219','ANON65248','ANON98217','ANON22366', 'ANON53486','ANON80073',
        'ANON93045','ANON26348','ANON72855','ANON60446','ANON28622','ANON60751','ANON41567'}

MR_blacklist = {'ANON74311', 'ANON13670', 'ANON74311', 'ANON32586'}


if __name__ == '__main__':
    print("Gathering DICOM scans")
    scans = Categorizer.categorize_files_by_scan("/data/public/PICARE_SEGMENTATION_BRAINVENT_MR_V1")
    
    mishaped = []
    errors = []
    i = 1
    
    for scan in scans:
        try:
            print(f"{i}/{len(scans)}")
            medim_obj = MedImage.create_from_files(scan)
            file_name = medim_obj.name.split("/")[-1]
            name = file_name.split("_")[0]
            destination = None
            if name in MR_blacklist:
                i += 1
                continue

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

            if name in MR_Val:
                destination = "validation"
            elif name in MR_test_1:
                destination = "test1"
            else:
                destination = "train"

            path = "/data/public/Segmentation_Dataset/MR_Dataset/{destination}".format(destination=destination)

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
            

np.save("/data/public/Segmentation_Dataset/MR_Dataset/mishaped.npy",np.array(mishaped))
np.save("/data/public/Segmentation_Dataset/MR_Dataset/errors.npy", np.array(errors))
            