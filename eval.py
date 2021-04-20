import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from sklearn import metrics
import re
from utils import get_ids, split_train_val, get_imgs_and_masks, batch
from utils import resize_and_crop, normalize_sar, normalize_cor, hwc_to_chw, dense_crf
from dice_loss import dice_coeff
from scipy import ndimage

dir_test = '/home-local/shimosato/datasets/test/'
# dir_test = '/home/shimosato/dl001/shimosato/dataset/unet/test/'


def eval_net(net, dataset, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    for i, b in tqdm(enumerate(dataset), total=n_val, desc='Validation round', unit='img'):
        imgs_sar = b[0]
        imgs_cor = b[1]
        gt = b[2]
        gt_mask = np.where(gt != 0, 1, 0)  # mask

        imgs_sar = torch.from_numpy(imgs_sar).unsqueeze(0)
        imgs_cor = torch.from_numpy(imgs_cor).unsqueeze(0)
        imgs = torch.cat([imgs_sar, imgs_cor], dim=1)
        gt = torch.from_numpy(gt).unsqueeze(0)
        gt_mask = torch.from_numpy(gt_mask)

        imgs = imgs.to(device=device)
        gt = gt.to(device=device)
        gt_mask = gt_mask.to(device=device)

        # mask_pred = net(imgs_sar).squeeze(dim=0)
        pred_def = net(imgs).squeeze(dim=0)
        pred_def_mask = pred_def * gt_mask

        # mask_pred = (mask_pred > 0.5).float()
        if net.module.n_classes > 1:
            tot += F.cross_entropy(pred_def_mask.unsqueeze(dim=0),
                                   gt.unsqueeze(dim=0)).item()
        else:
            tot += F.mse_loss(pred_def_mask, gt.squeeze(dim=1)).item()

    return tot / n_val


def eval_net_test(net, dataset, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    zero_count = 0
    leveling_ave = 0
    d = 0
    d2 = 0

    # for i, b in tqdm(enumerate(dataset), total=n_val, desc='Validation round', unit='img'):
    for i, fn in enumerate(list(get_ids(dir_test + 'sar/'))):
        # Load npy file
        sar = np.load(dir_test + 'sar/' + fn + '.npy')
        sar_cor = np.load(dir_test + 'cor/' + fn + '_cor.npy')

        # Normalize
        sar_norm = normalize_sar(sar)
        sar_cor_norm = normalize_cor(sar_cor)

        # HWC to CHW
        sar_in = hwc_to_chw(sar_norm)
        sar_cor_in = hwc_to_chw(sar_cor_norm)

        sar_torch = torch.from_numpy(sar_in).unsqueeze(0)
        sar_cor_torch = torch.from_numpy(sar_cor_in).unsqueeze(0)

        sar_torch = sar_torch.to(device=device)
        sar_cor_torch = sar_cor_torch.to(device=device)

        with torch.no_grad():
            pred_def = net(sar_torch, sar_cor_torch)
            pred_def = pred_def.squeeze(0)
            full_mask = pred_def.squeeze().cpu().numpy()

        gt = np.load(dir_test + 'gt/' + fn + '_leveling.npy')
        gt_mask = np.where(gt != 0, 1, 0)  # mask

        # Nonzero method
        train_mask = sar[gt.nonzero()]
        pred_mask = full_mask[gt.nonzero()]
        gt = gt[gt.nonzero()]
        leveling_ave += gt.shape[0]

        if not i:
            train_ar = train_mask
            pred_ar = pred_mask
            gt_ar = gt
        else:
            train_ar = np.append(train_ar, train_mask)
            pred_ar = np.append(pred_ar, pred_mask)
            gt_ar = np.append(gt_ar, gt)

        if not train_mask.any():
            zero_count += 1
            continue

        # train_rmse = np.sqrt(metrics.mean_squared_error(train_mask, gt))
        pred_rmse = np.sqrt(metrics.mean_squared_error(pred_mask, gt))

        # d += train_rmse
        d2 += pred_rmse

        # if train_rmse == 0 and pred_rmse == 0:
        if pred_rmse == 0:
            zero_count += 1

    divide_num = i+1-zero_count
    # train_rmse = d / (divide_num)
    pred_rmse = d2 / (divide_num)

    # mask_pred = (mask_pred > 0.5).float()
    # if net.module.n_classes > 1:
    #     tot += F.cross_entropy(pred_def_mask.unsqueeze(dim=0), gt.unsqueeze(dim=0)).item()
    # else:
    #     tot += F.mse_loss(pred_def_mask, gt.squeeze(dim=1)).item()

    return pred_rmse


def eval_net_test_smoothing(net, dataset, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    zero_count = 0
    leveling_ave = 0
    d = 0
    d2 = 0

    # for i, b in tqdm(enumerate(dataset), total=n_val, desc='Validation round', unit='img'):
    for i, fn in enumerate(list(get_ids(dir_test + 'sar/'))):
        # Load npy file
        sar = np.load(dir_test + 'sar/' + fn + '.npy')
        sar = ndimage.filters.gaussian_filter(sar, 5)  # gaussian_filter
        sar_cor = np.load(dir_test + 'cor/' + fn + '_cor.npy')

        # Normalize
        sar_norm = normalize_sar(sar)
        sar_cor_norm = normalize_cor(sar_cor)

        # HWC to CHW
        sar_in = hwc_to_chw(sar_norm)
        sar_cor_in = hwc_to_chw(sar_cor_norm)

        sar_torch = torch.from_numpy(sar_in).unsqueeze(0)
        sar_cor_torch = torch.from_numpy(sar_cor_in).unsqueeze(0)

        sar_torch = sar_torch.to(device=device)
        sar_cor_torch = sar_cor_torch.to(device=device)

        with torch.no_grad():
            pred_def = net(sar_torch, sar_cor_torch)
            pred_def = pred_def.squeeze(0)
            full_mask = pred_def.squeeze().cpu().numpy()

        gt = np.load(dir_test + 'gt/' + fn + '_leveling.npy')
        gt_mask = np.where(gt != 0, 1, 0)  # mask

        # Nonzero method
        train_mask = sar[gt.nonzero()]
        pred_mask = full_mask[gt.nonzero()]
        gt = gt[gt.nonzero()]
        leveling_ave += gt.shape[0]

        if not i:
            train_ar = train_mask
            pred_ar = pred_mask
            gt_ar = gt
        else:
            train_ar = np.append(train_ar, train_mask)
            pred_ar = np.append(pred_ar, pred_mask)
            gt_ar = np.append(gt_ar, gt)

        if not train_mask.any():
            zero_count += 1
            continue

        # train_rmse = np.sqrt(metrics.mean_squared_error(train_mask, gt))
        pred_rmse = np.sqrt(metrics.mean_squared_error(pred_mask, gt))

        # d += train_rmse
        d2 += pred_rmse

        # if train_rmse == 0 and pred_rmse == 0:
        if pred_rmse == 0:
            zero_count += 1

    divide_num = i+1-zero_count
    # train_rmse = d / (divide_num)
    pred_rmse = d2 / (divide_num)

    # mask_pred = (mask_pred > 0.5).float()
    # if net.module.n_classes > 1:
    #     tot += F.cross_entropy(pred_def_mask.unsqueeze(dim=0), gt.unsqueeze(dim=0)).item()
    # else:
    #     tot += F.mse_loss(pred_def_mask, gt.squeeze(dim=1)).item()

    return pred_rmse
