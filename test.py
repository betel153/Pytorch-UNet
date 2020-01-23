import numpy as np

a = np.zeros([2, 1, 3, 3])
a[0, 0, 1, 1] = 1
a[1, 0, 2, 2] = 1
print(a)
b = np.nonzero(a)
print(b)
c = np.transpose(b)
print(a[b])
test=0
