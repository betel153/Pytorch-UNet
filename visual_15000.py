# 1. visual_15000.py            ←
# 2. visual_predict.py
# 3. visual_npy_concat.py
# 4. Datagenerator_visual.py

# RAW画像からcrop_sizeに準じてクロップしてnpyで保存

import numpy as np
import scipy.io as sio
import os
import datetime
import h5py

dt_now = datetime.datetime.now()

base_dir = '/home/shimosato/dataset/unet/'
crop_size = (12288,12288)
split = 3 # 各辺

# input folder
input_dir = base_dir + 'test/raw/'

# output folder
sar_out_dir = base_dir + 'visual_15000/sar/'
cor_out_dir = base_dir + 'visual_15000/cor/'
gt_out_dir = base_dir + 'visual_15000/gt/'

ids = []
for f in os.listdir(input_dir):
    if not f.startswith('.'):
        if not '_' in f:
            ids.append(os.path.splitext(f)[0])

for id in ids:
    print(id)

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

    count = 1
    for i in range(split):
        for j in range(split):
            # print("sar[",int(i * crop_size[0] / split),":",int((i+1) * crop_size[0] / split),",",int(j * crop_size[0] / split),":",int((j+1) * crop_size[0] / split),"]")
            np.save(sar_out_dir + id + '_' + str(count), sar[int(i * crop_size[0] / split):int((i+1) * crop_size[0] / split), int(j * crop_size[0] / split):int((j+1) * crop_size[0] / split)])
            np.save(cor_out_dir + id + '_' + str(count) + '_cor', cor[int(i * crop_size[0] / split):int((i+1) * crop_size[0] / split), int(j * crop_size[0] / split):int((j+1) * crop_size[0] / split)])
            np.save(gt_out_dir + id + '_' + str(count) + '_leveling', gt[int(i * crop_size[0] / split):int((i+1) * crop_size[0] / split), int(j * crop_size[0] / split):int((j+1) * crop_size[0] / split)])
            count += 1
