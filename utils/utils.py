import random

import numpy as np


def hwc_to_chw(img):
    img = np.expand_dims(img, axis=2)       # add axis
    return np.transpose(img, axes=[2, 0, 1])

def normalize_sar(x):
    # if x.shape[0] == 512:
    #     mean = 0.28402385
    #     std = 9.549969
    # elif x.shape[0] == 2048:
    #     mean = 0.12748308
    #     std = 9.599052
    # else:
    #     mean = 0.14989512
    #     std = 9.466791
    mean = 0.16424856
    std = 7.7834034
    return (x - mean) / std
    # return x # stop normalize

def normalize_cor(x):
    # if x.shape[0] == 512:
    #     mean = 0.002839577
    #     std = 2.3237224
    # elif x.shape[0] == 2048:
    #     mean = 0.001480923
    #     std = 2.4691918
    # else:
    #     mean = 0.045178086
    #     std = 2.6548002
    mean = 0.01181804
    std = 1.5279412
    return (x - mean) / std
    # return x

def resize_and_crop(img, scale=1, final_height=None):
    if scale != 1:
        w = img.shape[0]
        h = img.shape[1]
        newW = int(w * scale)
        newH = int(h * scale)

        reimg = np.empty((newW, newH)).astype(np.float32)
        for i in range(newW):
                for j in range(newW):
                    reimg[i][j] = img[2*i+1][2*j+1]
        return reimg
    else:
        return img

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t) # この時点でbには本体とcor,levelingの画素データが代入
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    # length = len(dataset)
    # n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset, 'val': []}
    # return {'train': dataset[:-n], 'val': dataset[-n:]}
    # return {'train': dataset[:-1], 'val': dataset[-1:]} % test mode


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
