# 1. visual_15000.py
# 2. visual_predict.py
# 3. visual_npy_concat.py       ←
# 4. Datagenerator_visual.py

# 元々あるsar, cor, gtと推論で出力されるOUT.npy（sar）をconcatして元に近い解像度にする
# sarディレクトリは一つのRAWのみにする

import numpy as np
import os

dir = '/home/shimosato/dataset/unet/visual_15000/'

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (os.path.splitext(f)[0] for f in os.listdir(dir) if not f.startswith('.'))

sar = []
cor = []
gt = []
out = []

# SAR
for i, fn in enumerate(list(get_ids(dir + 'sar/'))):
    print('FileName:', fn)
    sar.append(np.load(dir + 'sar/' + fn +'.npy'))
    cor.append(np.load(dir + 'cor/' + fn +'_cor.npy'))
    gt.append(np.load(dir + 'gt/' + fn +'_leveling.npy'))
    out.append(np.load(dir + 'out/' + fn +'_OUT.npy'))

for i, fnn in enumerate(sar):
    sar[i] = sar[i][20:4076, 20:4076]
    cor[i] = cor[i][20:4076, 20:4076]
    gt[i] = gt[i][20:4076, 20:4076]
    out[i] = out[i][20:4076, 20:4076]

# 以下9分割用のハードコード
sar = np.block([[sar[0], sar[1], sar[2]], [sar[3], sar[4], sar[5]], [sar[6], sar[7], sar[8]]])
cor = np.block([[cor[0], cor[1], cor[2]], [cor[3], cor[4], cor[5]], [cor[6], cor[7], cor[8]]])
gt = np.block([[gt[0], gt[1], gt[2]], [gt[3], gt[4], gt[5]], [gt[6], gt[7], gt[8]]])
out = np.block([[out[0], out[1], out[2]], [out[3], out[4], out[5]], [out[6], out[7], out[8]]])

np.save(dir + 'concat/sar/' + fn +'.npy', sar)
np.save(dir + 'concat/cor/' + fn +'_cor.npy', cor)
np.save(dir + 'concat/gt/' + fn +'_leveling.npy', gt)
np.save(dir + 'concat/out/' + fn +'_OUT.npy', out)
