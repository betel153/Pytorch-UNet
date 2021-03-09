import os

import numpy as np
from PIL import Image

import scipy.io as sio

dir_gt = '/home/shimosato/dl001/shimosato/dataset/unet/test/gt/'

def resize_and_crop(img, scale=1, final_height=None):
    if scale != 1:
        w = img.shape[0]
        h = img.shape[1]
        newW = int(w * scale)
        newH = int(h * scale)

        reimg = np.empty((newW, newH)).astype(np.float32)
        for i in range(newW):
                for j in range(newW):
                    reimg[i][j] = img[2*i+1][2*j+1]
        return reimg
    else:
        return img

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

ids = get_ids(dir_gt)
num = np.empty(0, dtype=np.int)
for id in ids:
    gt = np.load(dir_gt + id + '.npy')
    gt_mask = np.where(gt != 0)  # mask
    if len(gt_mask[0]):
        num = np.append(num, len(gt_mask[0]))

ave = np.mean(num)
max = np.max(num)
min = np.min(num)
med = np.median(num)

print(ave)
print(max)
print(min)
print(med)

import csv
with open('leveling_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(num)
