# load nodules used in script
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import helperFunctions
import os

# set path of MEDimage class
sys.path.extend(['/home/nanot/MEDomicsLab-develop/Code/'])
from MEDimage.MEDimage import MEDimage

# for each patient, create 2 MEDimage objects for brain and ventricle ROIs:
# - read patient dicoms and RTs
# - save MEDimage object
    # - image
    # - mask
    # - modified_images, an array with 2 images (normalized, normalized skull stripped)
# MEDimage objects are saved in folder 'output_script1'


## MR Dataset
inputDataPath = Path('/data/public/PICARE_SEGMENTATION_BRAINVENT_MR_V1/')

roiB = '{Brain}'
roiV = '{Ventricles}'

for f in os.listdir(inputDataPath): # each patient

    patientFolder = Path(os.path.join(inputDataPath, f))
    
    if os.path.isdir(patientFolder):
        
        # 1. ROI for brain
        # create MEDimage class for patient, this includes raw images and masks
        brainMEDimage = MEDimage(path_patient=patientFolder, name_roi=roiB, compute_radiomics_features=False, save_modified_image=False)
        
        # get raw and mask images
        [rawImage, brainMask] = brainMEDimage.get_image_and_mask()
        
        # normalize images, should we make this a MEDimage class method?
        normImage = helperFunctions.makeNormImage(rawImage)
        
        # brain only image "skull stripped", should we make this a MEDimage class method?
        normSkullStrippedImage = normImage
        normSkullStrippedImage.data = normSkullStrippedImage.data*brainMask.data
        
        # add norm and striped image to object 
        brainMEDimage.modified_image = [normImage, normSkullStrippedImage]
        
        # save
        filename = 'brain_' + f
        brainMEDimage.save_MEDimage_object(path_save='output_script1', name_save=filename)
        
        
        # 2. ROI for ventricle
        # create MEDimage class for patient, this includes raw images and masks
        ventMEDimage = MEDimage(path_patient=patientFolder, name_roi=roiV, compute_radiomics_features=False, save_modified_image=False)
        
        # get raw and mask images
        [rawImage, ventMask] = ventMEDimage.get_image_and_mask()
        
        # normalize images, should we make this a MEDimage class method?
        normImage = helperFunctions.makeNormImage(rawImage)
        
        # ventricle only image, should we make this a MEDimage class method?
        normVentImage = normImage
        normVentImage.data = normVentImage.data*ventMask.data
        
        # add norm and striped image to object 
        ventMEDimage.modified_image = [normImage, normVentImage]
        
        # save
        filename = 'ventricle_' + f
        ventMEDimage.save_MEDimage_object(path_save='output_script1', name_save=filename)
        
    else:
        print('Error in inputDataPath.')



        
        