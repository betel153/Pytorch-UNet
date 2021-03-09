# 1. visual_15000.py
# 2. visual_predict.py          ←
# 3. visual_npy_concat.py
# 4. Datagenerator_visual.py

# 分割したnpyファイルをcpuで推論し，出力される画像をnpyで保存

import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from unet import UNet_fusenet
from utils import plot_img_and_mask
from utils import resize_and_crop, normalize_sar, normalize_cor, hwc_to_chw, dense_crf
from collections import OrderedDict # Use Old dataset
import scipy.io as sio
from sklearn import metrics
import re

dir_test = '/home/shimosato/dataset/unet/visual_15000/'

def predict_img(net,
                sar,
                sar_cor,
                device,
                scale_factor=1,
                out_threshold=0.5,
                use_dense_crf=False):
    net.eval()
    # sar_height = sar.shape[1]

    # img = resize_and_crop(sar, scale=scale_factor)
    # img = normalize(img)
    # img = hwc_to_chw(sar)

    sar_torch = torch.from_numpy(sar).unsqueeze(0)
    sar_cor_torch = torch.from_numpy(sar_cor).unsqueeze(0)
    # X = torch.cat([sar_torch, sar_cor_torch], dim=1)
    # X = X.to(device=device)

    sar_torch = sar_torch.to(device=device)
    sar_cor_torch = sar_cor_torch.to(device=device)

    with torch.no_grad():
        probs = net(sar_torch, sar_cor_torch)

        # if net.n_classes > 1:
        #     probs = F.softmax(output, dim=1)
        # else:
        #     probs = torch.sigmoid(output)

        # jpg
        # probs = MinMaxScaler(output)

        probs = probs.squeeze(0)

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(sar_height),
        #         transforms.ToTensor()
        #     ]
        # )

        # probs = tf(probs.cpu())

        full_mask = probs.squeeze().cpu().numpy()

    if use_dense_crf:
        full_mask = dense_crf(np.array(sar).astype(np.uint8), full_mask)

    # return full_mask > out_threshold
    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def MinMaxScaler(sar):
    return (sar - torch.min(sar)) / (torch.max(sar) - torch.min(sar))

# Use Old Dataset
def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (os.path.splitext(f)[0] for f in os.listdir(dir) if not f.startswith('.'))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    all_train_eval = []
    all_pred_eval = []
    a = 0
    a2 = 0
    b = 0
    b2 = 0
    c = 0
    c2 = 0
    d = 0
    d2 = 0
    e = 0
    e2 = 0
    test = 0
    test2 =0
    args = get_args()
    args.input = list(get_ids(dir_test + 'sar/'))
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet_fusenet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # Use Old dataset
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(fix_model_state_dict(state_dict))
    # net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    zero_count = 0
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))
        print('FileName:', fn)

        # fn = os.path.splitext(fn)[0]

        # Load npy file
        sar = np.load(dir_test + 'sar/' + fn + '.npy')
        sar_cor = np.load(dir_test + 'cor/' + fn + '_cor.npy')

        # Normalize
        sar_norm = normalize_sar(sar)
        sar_cor_norm = normalize_cor(sar_cor)

        # HWC to CHW
        sar_in = hwc_to_chw(sar_norm)
        sar_cor_in = hwc_to_chw(sar_cor_norm)

        # sar = sio.loadmat(fn + '.mat')    # import matfile
        # sar_cor = sio.loadmat(fn + '_cor.mat')    # import matfile
        # sar = sar['data'].astype(np.float32)       # change type
        # sar_cor = sar_cor['data'].astype(np.float32)       # change type

        mask = predict_img(net=net,
                           sar=sar_in,
                           sar_cor=sar_cor_in,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=False,
                           device=device)

        if not args.no_save:
            # out_fn = out_files[i]
            # result = mask_to_image(mask)
            # result.save(out_files[i] + '.jpg')

            np.save(dir_test + 'out/' + out_files[i], mask)
            logging.info("Mask saved to {}".format(out_files[i]))
