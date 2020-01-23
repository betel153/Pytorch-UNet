import numpy as np
import scipy.io as sio
import os
import datetime
import h5py

dt_now = datetime.datetime.now()

base_dir = '/home/shimosato/dataset/unet/'
# crop_size = (1024, 1024)
crop_size = (2048, 2048)

# input folder
input_dir = base_dir + 'datafolder/'

# output folder
sar_out_dir = base_dir + 'train2048/sar/'
cor_out_dir = base_dir + 'train2048/cor/'
gt_out_dir = base_dir + 'train2048/gt/'

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
    # sar = sio.loadmat(input_dir + id + '.mat')    # import matfile
    # cor = sio.loadmat(input_dir + id + '_cor.mat')    # import matfile
    # gt = sio.loadmat(input_dir + id + '_leveling.mat')    # import matfile
    # xy = sio.loadmat(input_dir + id + '_xy.mat')    # import matfile

    # sar = sar['data'].astype(np.float32)       # change type
    # cor = cor['data'].astype(np.float32)       # change type
    # gt = gt['data'].astype(np.float32)       # change type
    # xy = xy['data'].astype(np.int32)       # change type

    # Read HDF5 Format File
    with h5py.File(input_dir + id + '.mat', 'r') as f:
        sar = np.transpose(f['data'][()].astype(np.float32), axes=[1, 0])
    with h5py.File(input_dir + id + '_cor.mat', 'r') as f:
        cor = np.transpose(f['data'][()].astype(np.float32), axes=[1, 0])
    with h5py.File(input_dir + id + '_leveling.mat', 'r') as f:
        gt = np.transpose(f['data'][()].astype(np.float32), axes=[1, 0])
    with h5py.File(input_dir + id + '_xy.mat', 'r') as f:
        xy = np.transpose(f['data'][()].astype(np.int32), axes=[1, 0])

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
