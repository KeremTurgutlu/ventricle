import numpy as np
import math

def extract_brain(image_voxel, brain_voxel):
    return image_voxel * brain_voxel

def weighted_avg_and_std(values, weights=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, math.sqrt(variance)

def normalize(image_voxel, weights=None):
    ave, std = weighted_avg_and_std(image_voxel, weights=weights)
    image_voxel = (image_voxel - ave) / std
    return image_voxel


def z_pad(image):
    depth, height, width = image.data.shape
    if depth == 128: return image
    if depth > 128:
        target = depth - 128
        if target % 2 == 0:
            n_pad_up, n_pad_down = int(target / 2), int(target / 2)
        else:
            n_pad_up, n_pad_down = int(target // 2) + 1, int(target // 2)
        image.data = image.data[n_pad_up:n_slices-n_pad_down]
    else:
        target = 128 - depth
        if target % 2 == 0:
            n_pad_up, n_pad_down = int(target / 2), int(target / 2)
        else:
            n_pad_up, n_pad_down = int(target // 2) + 1, int(target // 2)
        up_voxel = np.zeros((n_pad_up, height, width))
        down_voxel = np.zeros((n_pad_down, height, width))
        image.data = np.concatenate([up_voxel, image.data, down_voxel])
    return image

def xy_pad(image):
    depth, height, width = image.data.shape
    diff_h = int(abs((height - 256)/2))
    diff_w = int(abs((width - 256)/2))
    if height > 256:
        if height % 2 == 0:
            image.data = image.data[:,diff_h:height-diff_h,:]
        else:
            image.data = image.data[:,diff_h+1:height-diff_h,:]
    else:
        if height % 2 == 0:
            image.data = np.pad(image.data, ((0,0),(diff_h,diff_h),(0,0)), 'constant')
        else:
            image.data = np.pad(image.data, ((0,0),(diff_h+1,diff_h),(0,0)), 'constant')
    if width > 256:
        if width % 2 == 0:
            image.data = image.data[:,:,diff_w:width-diff_w]
        else:
            image.data = image.data[:,:,diff_w+1:width-diff_w]
    else:
        if width % 2 == 0:
            image.data = np.pad(image.data, ((0,0),(0,0),(diff_w,diff_w)), 'constant')
        else:
            image.data = np.pad(image.data, ((0,0),(0,0),(diff_w+1,diff_w)), 'constant')
    return image