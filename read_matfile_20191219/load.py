""" Utils on generators / lists of ids to transform from strings to cropped images and masks """

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, normalize, hwc_to_chw
import scipy.io as sio


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (os.path.splitext(f)[0] for f in os.listdir(dir) if not f.startswith('.'))


def to_cropped_imgs(ids, dir, suffix, crop_size=(1000, 1000)):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        # im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        mat = sio.loadmat(dir + id + suffix)    # import matfile
        mat = mat['data'].astype(np.float32)       # change type

        mat = np.expand_dims(mat, axis=2)       # add axis
        yield mat

def get_imgs_and_masks(ids, dir_img, dir_cor, dir_gt, scale):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, dir_img, '.mat', scale) # change jpg to mat
    imgs_cor = to_cropped_imgs(ids, dir_cor, '_cor.mat', scale) # change jpg to mat

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_cor_switched = map(hwc_to_chw, imgs_cor)
    # imgs_normalized = map(normalize, imgs_switched)

    gt = to_cropped_imgs(ids, dir_gt, '_leveling.mat', scale)
    gt_switched = map(hwc_to_chw, gt)

    return zip(imgs_switched, imgs_cor_switched, gt_switched)

def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
