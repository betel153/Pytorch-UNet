import numpy as np
import scipy.io as sio
import os
import datetime

dt_now = datetime.datetime.now()

base_dir = '/home/shimosato/dataset/unet/'
# crop_size = (1024, 1024)
crop_size = (512, 512)

# input folder
# sar_in_dir = base_dir + 'datafolder/sar/'
# cor_in_dir = base_dir + 'datafolder/cor/'
# gt_in_dir = base_dir + 'datafolder/gt/'
# xy_in_dir = base_dir + 'datafolder/xy/'
input_dir = base_dir + 'datafolder/'

# output folder
sar_out_dir = base_dir + 'train2/sar/'
cor_out_dir = base_dir + 'train2/cor/'
gt_out_dir = base_dir + 'train2/gt/'

# sar_out_dir = base_dir + 'test/sar/'
# cor_out_dir = base_dir + 'test/cor/'
# gt_out_dir = base_dir + 'test/gt/'

ids = []
for f in os.listdir(input_dir):
    if not f.startswith('.'):
        if not '_' in f:
            ids.append(os.path.splitext(f)[0])

for id in ids:
    print(id)
    sar = sio.loadmat(input_dir + id + '.mat')    # import matfile
    cor = sio.loadmat(input_dir + id + '_cor.mat')    # import matfile
    gt = sio.loadmat(input_dir + id + '_leveling.mat')    # import matfile
    xy = sio.loadmat(input_dir + id + '_xy.mat')    # import matfile

    sar = sar['data'].astype(np.float32)       # change type
    cor = cor['data'].astype(np.float32)       # change type
    gt = gt['data'].astype(np.float32)       # change type
    xy = xy['data'].astype(np.int32)       # change type

    h, w = sar.shape

    for i, center in enumerate(xy):
        sar_output = []
        c_h, c_w = center  # GT's center
        # decide top and left
        top = c_h - crop_size[0] // 2
        left = c_w - crop_size[1] // 2
        # decide bottom and right
        bottom = top + crop_size[0]
        right = left + crop_size[1]

        if right > w:
            left -= right - w
            left -= np.random.randint(0, left)
            right = left + crop_size[1]
        if bottom > h:
            top -= right - h
            top -= np.random.randint(0, top)
            bottom = top + crop_size[1]
        # print('top:',top,'left:',left,'bottom:',bottom,'right:',right)

        np.save(sar_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M'), sar[top:bottom, left:right])
        np.save(cor_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_cor', cor[top:bottom, left:right])
        np.save(gt_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_leveling', gt[top:bottom, left:right])
