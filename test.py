import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def img_show(img: np.ndarray, cmap='gray', vmin=0, vmax=255, interpolation='none') -> None:
    '''np.arrayを引数とし、画像を表示する。'''

    # dtypeをuint8にする
    img = np.clip(img, vmin, vmax).astype(np.uint8)

    # 画像を表示
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
               interpolation=interpolation)
    plt.show()
    plt.close()


def convolve2d(img, kernel):
    # 部分行列の大きさを計算
    sub_shape = tuple(np.subtract(img.shape, kernel.shape) + 1)

    # 関数名が長いのでいったん省略
    strd = np.lib.stride_tricks.as_strided

    # 部分行列の行列を作成
    submatrices = strd(img, kernel.shape + sub_shape, img.strides * 2)

    # 部分行列とカーネルのアインシュタイン和を計算
    convolved_matrix = np.einsum('ij,ijkl->kl', kernel, submatrices)

    return convolved_matrix


def gaussian_kernel(n: int) -> np.ndarray:
    '''(n,n)のガウス行列を作る'''

    # [nC0, nC1, ..., nCn]を作成
    combs = [1]
    for i in range(1, n):
        ratio = (n-i)/(i)
        combs.append(combs[-1]*ratio)
    combs = np.array(combs).reshape(1, n)/(2**(n-1))

    # 縦ベクトルと横ベクトルの積でガウス行列を作る
    result = combs.T.dot(combs)
    return result


# img = plt.imread('tiger.jpg')[1200:1500, 1400:1700]
img = np.load(
    '/home/shimosato/dl001/shimosato/dataset/unet/data_num/x5/sar/aichi2017IW1_105_202008250108.npy')
# img = (0.298912 * img[..., 0] + 0.586611 *
#        img[..., 1] + 0.114478 * img[..., 2])
# img_show(img)
plt.imsave('figure01.jpg', img)
# plt.imsave('figure02.jpg', convolve2d(img, kernel))
# img_gaussian = ndimage.filters.gaussian_filter(img, 12)
# pass
plt.imsave('figure02.jpg', ndimage.filters.gaussian_filter(img, 5))
