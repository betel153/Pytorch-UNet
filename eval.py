import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from dice_loss import dice_coeff


def eval_net(net, dataset, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    for i, b in tqdm(enumerate(dataset), total=n_val, desc='Validation round', unit='img'):
        imgs_sar = b[0]
        imgs_cor = b[1]
        gt = b[2]
        gt_mask = np.where(gt != 0, 1, 0) # mask

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
            tot += F.cross_entropy(pred_def_mask.unsqueeze(dim=0), gt.unsqueeze(dim=0)).item()
        else:
            tot += F.mse_loss(pred_def_mask, gt.squeeze(dim=1)).item()

    return tot / n_val
