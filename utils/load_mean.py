""" Utils on generators / lists of ids to transform from strings to cropped images and masks """

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, normalize_sar, normalize_cor, hwc_to_chw
import scipy.io as sio


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (os.path.splitext(f)[0] for f in os.listdir(dir) if not f.startswith('.'))

def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        # im = resize_and_crop(Image.open(dir + id + suffix), scale=scale) # original

        # MATLAB file
        im = resize_and_crop(np.load(dir + id + suffix), scale)    # import
        yield im

# 畳み込みによるフィルタリング
def convolve2d(img, kernel):
    #部分行列の大きさを計算
    sub_shape = tuple(np.subtract(img.shape, kernel.shape) + 1)

    #関数名が長いのでいったん省略
    strd = np.lib.stride_tricks.as_strided

    #部分行列の行列を作成
    submatrices = strd(img,kernel.shape + sub_shape,img.strides * 2)

    #部分行列とカーネルのアインシュタイン和を計算
    convolved_matrix = np.einsum('ij,ijkl->kl', kernel, submatrices)

    return convolved_matrix

def gaussian_kernel(n : int) -> np.ndarray:
    '''(n,n)のガウス行列を作る'''

    #[nC0, nC1, ..., nCn]を作成
    combs = [1]
    for i in range(1,n):
        ratio = (n-i)/(i)
        combs.append(combs[-1]*ratio)
    combs = np.array(combs).reshape(1,n)/(2**(n-1))

    #縦ベクトルと横ベクトルの積でガウス行列を作る
    result = combs.T.dot(combs)
    return result

def get_imgs_and_masks(ids, dir_img, dir_cor, dir_gt, scale):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, dir_img, '.npy', scale) # change jpg to mat
    imgs_cor = to_cropped_imgs(ids, dir_cor, '_cor.npy', scale) # change jpg to mat

    # need to transform from HWC to CHW
    imgs_normalized = map(normalize_sar, imgs)
    #
    # 平均化処理を入れるならこの行．corは無しでも問題ないか．
    # https://qiita.com/secang0/items/f3a3ff629988dc660d87
    # ↑が参考になるか
    imgs_filtered = map(convolve2d, imgs_normalized, gaussian_kernel(5))
    #
    imgs_switched = map(hwc_to_chw, imgs_filtered)

    imgs_cor_normalized = map(normalize_cor, imgs_cor)
    imgs_cor_switched = map(hwc_to_chw, imgs_cor_normalized)

    gt = to_cropped_imgs(ids, dir_gt, '_leveling.npy', scale)
    gt_switched = map(hwc_to_chw, gt)

    return zip(imgs_switched, imgs_cor_switched, gt_switched)

def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
