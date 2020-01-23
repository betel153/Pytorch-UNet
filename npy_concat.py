import numpy as np

sar1 = np.load('ibaraki2016IW2_OUT.npy')
sar2 = np.load('ibaraki2016IW2_OUT2.npy')
sar3 = np.load('ibaraki2016IW2_OUT3.npy')
sar4 = np.load('ibaraki2016IW2_OUT4.npy')
sar5 = np.load('ibaraki2016IW2_OUT5.npy')
sar6 = np.load('ibaraki2016IW2_OUT6.npy')
sar7 = np.load('ibaraki2016IW2_OUT7.npy')
sar8 = np.load('ibaraki2016IW2_OUT8.npy')
sar9 = np.load('ibaraki2016IW2_OUT9.npy')

sar = np.block([[sar1, sar2, sar3], [sar4, sar5, sar6], [sar7, sar8, sar9]])

np.save('ibaraki2016IW2_concat.npy', sar)
