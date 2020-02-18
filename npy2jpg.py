import argparse
import logging
import os

import numpy as np
from PIL import Image

from unet import UNet
from utils import plot_img_and_mask
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_GT_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def MinMaxScaler(sar):
    return (sar - np.amin(sar)) / (np.amax(sar) - np.amin(sar))

def sigmoid(sar):
    return 1 / (1 + np.exp(-sar))

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        fn = os.path.splitext(fn)[0]
        # img = Image.open(fn)

        # NPY file
        sar = np.load(fn + '.npy')

        # MAT file
        # with h5py.File(fn + '.mat', 'r') as f:
        #     sar = np.transpose(f['data'][()].astype(np.float32), axes=[1, 0])

        # sar = MinMaxScaler(sar)
        sar = sigmoid(sar)

        out_fn = out_files[i]
        result = mask_to_image(sar)
        plt.imshow((sar * 255).astype(np.uint8), "gray")  # グレースケールで
        plt.colorbar()  # カラーバーの表示
        plt.show()
        # result.save(out_files[i] + '.jpg')

        logging.info("Mask saved to {}".format(out_files[i]))
