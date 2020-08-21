import numpy as np
import scipy.io as sio
import os
import datetime
import h5py

dt_now = datetime.datetime.now()

base_dir = '/home/shimosato/dataset/unet/'
crop_size = (1024, 1024)
#crop_size = (2048, 2048)

# input folder
input_dir = base_dir + 'datafolder/'
# input_dir = base_dir + 'test/raw/'

# output folder
sar_out_dir = base_dir + 'train/sar/'
cor_out_dir = base_dir + 'train/cor/'
gt_out_dir = base_dir + 'train/gt/'

# sar_out_dir = base_dir + 'test/sar/'
# cor_out_dir = base_dir + 'test/cor/'
# gt_out_dir = base_dir + 'test/gt/'

ids = []
for f in os.listdir(input_dir):
    if not f.startswith('.'):
        if not '_' in f:
            ids.append(os.path.splitext(f)[0])

flg_g = 0
cnt_g = 0
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

    flg = 0
    cnt = 0
    for i, ps in enumerate(xy):
        ps_col = ps - 1
        if ps_col[0] > sar.shape[0] or ps_col[1] > sar.shape[1] or ps_col[0] == 0 or ps_col[1] == 0:
            continue
        # print("xy :", ps)
        # print("gt :", gt[tuple(ps_col)])
        # print("sar:", sar[tuple(ps_col)])
        # print("")
        cnt += 1
        if sar[tuple(ps_col)] != 0:
            flg += 1
    print("xy ：", xy.shape[0])
    print("flg：", flg)
    print("cnt：", cnt)
    print("PS含有率：" ,flg / cnt)
    flg_g += flg
    cnt_g += cnt
print("flg_g：", flg_g)
print("cnt_g：", cnt_g)
print("PS含有率：", flg_g / cnt)
