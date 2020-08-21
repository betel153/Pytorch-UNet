import numpy as np
import scipy.io as sio
import os
import datetime
import h5py

base_dir = '/home/shimosato/dataset/unet/'

# input folder
input_dir = base_dir + 'train/cor/'

ids = []
for f in os.listdir(input_dir):
    if not f.startswith('.'):
        ids.append(os.path.splitext(f)[0])

# i=0
# for id in ids:
#     print(id)
#     sar = np.load(input_dir + id + '.npy')

#     if not i:
#         sar_all = sar
#     else:
#         sar_all = np.dstack([sar_all, sar])
#     i=1

tmp = np.empty((1024,1024,0), dtype=np.float32)
for id in ids:
    print(id)
    sar = np.load(input_dir + id + '.npy')
    tmp = np.dstack([tmp, sar])

filename = 'calc_mean_std_cor'
with open(filename + '.log', mode='w') as f:
    f.write('Mean ' + str(tmp.mean()) + '\n')
    f.write('Std  ' + str(tmp.std()))
