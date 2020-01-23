import numpy as np
import scipy.io as sio
import os
import datetime
import h5py

base_dir = '/home/shimosato/dataset/unet/'

# input folder
input_dir = '/home/shimosato/dataset/unet/train/'

# output folder
sar_out_dir = base_dir + 'train_re512/sar/'
cor_out_dir = base_dir + 'train_re512/cor/'
gt_out_dir = base_dir + 'train_re512/gt/'


scale = 0.5

ids = []
for f in os.listdir(input_dir + 'sar/'):
    if not f.startswith('.'):
        ids.append(os.path.splitext(f)[0])

for id in ids:
    print(id)

    # Read HDF5 Format File
    sar = np.load(input_dir + 'sar/' + id + '.npy')
    cor = np.load(input_dir + 'cor/' + id + '_cor.npy')
    gt = np.load(input_dir + 'gt/' + id + '_leveling.npy')

    w = sar.shape[0]
    h = sar.shape[1]
    newW = int(w * scale)
    newH = int(h * scale)

    resar = np.empty((newW, newH)).astype(np.float32)
    recor = np.empty((newW, newH)).astype(np.float32)
    regt = np.empty((newW, newH)).astype(np.float32)
    for i in range(newW):
            for j in range(newW):
                resar[i][j] = sar[2*i+1][2*j+1]
                recor[i][j] = sar[2*i+1][2*j+1]
                regt[i][j] = sar[2*i+1][2*j+1]

    np.save(sar_out_dir + id, resar)
    np.save(cor_out_dir + id + '_cor', recor)
    np.save(gt_out_dir + id + '_leveling',regt)
