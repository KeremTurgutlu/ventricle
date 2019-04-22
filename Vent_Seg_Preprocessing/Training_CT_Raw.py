import copy
import sys
sys.path.extend(['/home/dalala/PiCare-develop/Code'])

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import helper
from medimage import MedImage
from file_categorizer import Categorizer

CT_val = {'ANON85656', 'ANON24135', 'ANON45434', 'ANON53464', 'ANON50198', 'ANON86095' , 'ANON47701', 'ANON21818', 'ANON13928', 'ANON45164', 'ANON57908', 'ANON10634', 'ANON37574', 'ANON13983', 'ANON39193', 'ANON52842', 'ANON83901', 'ANON34509', 'ANON14150', 'ANON70712', 'ANON36668', 'ANON86933', 'ANON69869', 'ANON55750'}

CT_test_1 = {'ANON95021', 'ANON17272', 'ANON45950', 'ANON71219', 'ANON84614', 'ANON22673', 'ANON65837', 'ANON51808', 'ANON24224'}

CT_blacklist = {'ANON13673', 'ANON29029', 'ANON64388', 'ANON51627', 'ANON58382', 'ANON80734', 'ANON50481', 'ANON25906',
            'ANON66568', 'ANON25334', 'ANON10495', 'ANON38793', 'ANON53534', 'ANON34438', 'ANON58827', 'ANON40613',
            'ANON13191', 'ANON66387', 'ANON86317', 'ANON93987', 'ANON95193', 'ANON12076'}


if __name__ == '__main__':
    print("Gathering DICOM scans")
    scans = Categorizer.categorize_files_by_scan("/data/public/Training_CT_Raw_DICOM_Files")
    
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
            if name in CT_blacklist:
                i += 1
                continue

            image, brain_mask = medim_obj.compute_mask("{Brain}", "full", interp=True)
            _, ventricle_mask = medim_obj.compute_mask("{Ventricles}", "full", interp=True)
            _, brain_atlas = medim_obj.compute_mask("{BrainAtlas}", "full", interp=True)
            _, ventricle_atlas = medim_obj.compute_mask("{VentriclesAtlas}", "full", interp=True)
            
            print(np.sum(brain_atlas.data))
            print(np.sum(ventricle_atlas.data))
            print(np.sum(brain_mask.data))
            
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

            if name in CT_val:
                destination = "validation"
            elif name in CT_test_1:
                destination = "test1"
            else:
                destination = "train"

            path = "/data/public/Segmentation_Dataset/CT_Dataset/{destination}".format(destination=destination)

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
            

np.save("/data/public/Segmentation_Dataset/CT_Dataset/mishaped.npy",np.array(mishaped))
np.save("/data/public/Segmentation_Dataset/CT_Dataset/errors.npy", np.array(errors))
            