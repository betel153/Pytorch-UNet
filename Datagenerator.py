import numpy as np
import scipy.io as sio
import os
import datetime
import h5py

dt_now = datetime.datetime.now()

base_dir = '/home/shimosato/dataset/unet/'
# crop_size = (1024, 1024)
crop_size = (4096, 4096)

# input folder
# input_dir = base_dir + 'datafolder/'
input_dir = base_dir + 'test/raw/'

# output folder
sar_out_dir = base_dir + 'train_visual_4096/sar/'
cor_out_dir = base_dir + 'train_visual_4096/cor/'
gt_out_dir = base_dir + 'train_visual_4096/gt/'

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
        xy -= 1 # MATLABの時点で水準点のXY座標が全て1ずつずれているため補正
    h, w = sar.shape

    leveling_dot = []

    for index in xy:
        if index.all() and index[0] < h and index[1] < w:
            leveling_dot.append(index)

    # leveling_dot_ag = []
    # for m_x, m_y in leveling_dot:
    #     for x in range(-1, 2):
    #         for y in range(-1, 2):
    #             leveling_dot_ag.append([m_x + x, m_y + y])
    #             gt[m_x + x, m_y + y] = gt[m_x, m_y]

    print('　Creating ...')
    # for i, center in enumerate(leveling_dot_ag): # Training
    for i, center in enumerate(leveling_dot): # Test
        sar_output = []
        c_h, c_w = center  # GT's center
        # decide top and left
        top = c_h - np.random.randint(0, crop_size[0] // 2)
        left = c_w - np.random.randint(0, crop_size[1] // 2)
        # decide bottom and right
        bottom = top + crop_size[0]
        right = left + crop_size[1]

        count = 0
        while True:
            if left < 0:
                left =  - left
                left += np.random.randint(0, w - right)
                right = left + crop_size[1]
            if right > w:
                left -= right - w
                left -= np.random.randint(0, left)
                right = left + crop_size[1]
            if top < 0:
                top = - top
                top += np.random.randint(0, h - bottom)
                bottom = top + crop_size[1]
            if bottom > h:
                top -= right - h
                top -= np.random.randint(0, top)
                bottom = top + crop_size[1]
            if c_h < top or c_h > bottom or c_w < left or c_w > right:
                top = c_h - np.random.randint(crop_size[0] // 4, crop_size[0] // 2)
                left = c_w - np.random.randint(crop_size[1] // 4, crop_size[1] // 2)
                # decide bottom and right
                bottom = top + crop_size[0]
                right = left + crop_size[1]
            if left >= 0 and right <= w and top >= 0 and bottom <= h and c_h >= top and c_h <= bottom and c_w >= left and c_w <= right:
                # print('Save now: ', id + '_' + str(i))
                gt_test = np.zeros(gt.shape)
                gt_test[c_h, c_w] = gt[c_h, c_w]
                np.save(sar_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M'), sar[top:bottom, left:right])
                np.save(cor_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_cor', cor[top:bottom, left:right])
                np.save(gt_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_leveling', gt[top:bottom, left:right])
                # np.save(gt_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_leveling', gt_test[top:bottom, left:right])
                break
            if count > 20:
                print('Error: ', id + '_' + str(i))
                break
            count += 1
        # print('top:',top,'left:',left,'bottom:',bottom,'right:',right)

        # gt_test = np.zeros(gt.shape)
        # gt_test[c_h-1, c_w-1] = gt[c_h-1, c_w-1]

        # np.save(sar_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M'), sar[top:bottom, left:right])
        # np.save(cor_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_cor', cor[top:bottom, left:right])
        # np.save(gt_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_leveling', gt[top:bottom, left:right])
        # np.save(gt_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_leveling', gt_test[top:bottom, left:right])
