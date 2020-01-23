import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from utils import get_ids, split_train_val, get_imgs_and_masks, batch
from torch.utils.tensorboard import SummaryWriter

dir_sar = '/home/shimosato/dataset/carvana-image-masking-challenge/sar/train/'
dir_cor = '/home/shimosato/dataset/carvana-image-masking-challenge/sar/cor/'
dir_gt = '/home/shimosato/dataset/carvana-image-masking-challenge/sar/gt/'
dir_checkpoint = 'checkpoints/'

def train_net(net,
              device,
              epochs=20,
              batch_size=1,
              lr=0.1,
              val_percent=0.15,
              save_cp=True,
              img_scale=1):
    ids = get_ids(dir_sar)

    iddataset = split_train_val(ids, val_percent)

    writer = SummaryWriter(comment=f'Learning_rate_{lr}_Batch_size_{batch_size}_SCALE_{img_scale}')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(iddataset["train"])}
        Validation size: {len(iddataset["val"])}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    n_train = len(iddataset['train'])
    n_val = len(iddataset['val'])
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.MSELoss()        # MSE Loss

    for epoch in range(epochs):
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_sar, dir_cor, dir_gt, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_sar, dir_cor, dir_gt, img_scale)

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            epoch_loss = 0
            for i, b in enumerate(batch(train, batch_size)):
                imgs_sar = np.array([i[0] for i in b]).astype(np.float32)
                imgs_cor = np.array([i[1] for i in b]).astype(np.float32)
                gt = np.array([i[2] for i in b]).astype(np.float32)
                gt_mask = np.where(gt != 0, 1, 0)  # mask

                imgs_sar = torch.from_numpy(imgs_sar)
                imgs_cor = torch.from_numpy(imgs_cor)
                imgs = torch.cat([imgs_sar, imgs_cor], dim=1)
                gt = torch.from_numpy(gt)
                gt_mask = torch.from_numpy(gt_mask)

                imgs = imgs.to(device=device)
                gt = gt.to(device=device)
                gt_mask = gt_mask.to(device=device)

                pred_def = net(imgs)
                pred_def_mask = pred_def * gt_mask
                loss = criterion(pred_def_mask, gt)
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(batch_size)
            writer.add_scalar('Loss/Train', epoch_loss, epoch+1)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        val_score = eval_net(net, val, device, n_val)
        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))
            writer.add_scalar('Loss/Validation', val_score, epoch+1)
        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))
            writer.add_scalar('Dice/Validation', val_score, epoch+1)
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=15.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


def pretrain_checks():
    imgs = [f for f in os.listdir(dir_sar) if not f.startswith('.')]
    masks = [f for f in os.listdir(dir_gt) if not f.startswith('.')]
    if len(imgs) != len(masks):
        logging.warning(f'The number of images and masks do not match ! '
                        f'{len(imgs)} images and {len(masks)} masks detected in the data folder.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    pretrain_checks()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=2, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

try:
    train_net(net=net,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              device=device,
              img_scale=args.scale,
              val_percent=args.val / 100)
except KeyboardInterrupt:
    torch.save(net.state_dict(), 'INTERRUPTED.pth')
    logging.info('Saved interrupt')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
