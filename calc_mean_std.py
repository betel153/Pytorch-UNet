import numpy as np
import scipy.io as sio
import os
import datetime
import h5py

base_dir = '/home/shimosato/dataset/unet/'

# input folder
input_dir = base_dir + 'train2048/cor/'

ids = []
for f in os.listdir(input_dir):
    if not f.startswith('.'):
        ids.append(os.path.splitext(f)[0])

i=0
for id in ids:
    print(id)
    sar = np.load(input_dir + id + '.npy')

    if not i:
        sar_all = sar
    else:
        sar_all = np.dstack([sar_all, sar])
    i=1

print(input_dir)
print('mean: ', sar_all.mean())
print('std: ', sar_all.std())
