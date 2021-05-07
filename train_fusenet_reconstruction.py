import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net_test
from unet import UNet_fusenet
from utils import get_ids, split_train_val, get_imgs_and_masks, batch
from torch.utils.tensorboard import SummaryWriter
import datetime
import optuna
dt_now = datetime.datetime.now()

# dir_sar = '/home/shimosato/dl001/shimosato/dataset/unet/data_num/x5/sar/'
# dir_cor = '/home/shimosato/dl001/shimosato/dataset/unet/data_num/x5/cor/'
# dir_gt = '/home/shimosato/dl001/shimosato/dataset/unet/data_num/x5/gt/'
dir_sar = '/home-local/shimosato/datasets/x5/sar/'
dir_cor = '/home-local/shimosato/datasets/x5/cor/'
dir_gt = '/home-local/shimosato/datasets/x5/gt/'
# dir_sar = '/home-local/shimosato/datasets/test/sar/'
# dir_cor = '/home-local/shimosato/datasets/test/cor/'
# dir_gt = '/home-local/shimosato/datasets/test/gt/'
dir_checkpoint = 'checkpoints/'

dir_test = '/home-local/shimosato/datasets/test/'
# dir_test = '/home/shimosato/dl001/shimosato/dataset/unet/test/'

# leveling_loss (alpha) vs reconstruction_loss (1-alpha)
alpha = 0.5


def gaussian(x, y, mus, S):
    filter = np.ones([mus.shape[0], 1, mus.shape[2], mus.shape[3]])
    tmp = 0
    for i, ch in enumerate(np.nonzero(mus)[0]):
        Z = 0
        if not i:
            tmp = ch
            chlist = np.array(ch)
            tmpch = ch
        else:
            if tmp == ch:
                chlist = np.append(chlist, i)
                tmpch = ch
            else:
                mu2 = np.transpose(np.nonzero(mus))[chlist, 2:4]
                if len(mu2.shape) == 1:
                    mu2 = mu2[None, :]
                for mu in mu2:
                    x_norm = (np.array([x, y]) -
                              mu[:, None, None]).transpose(1, 2, 0)
                    Z += np.exp(- x_norm[:, :, None, :] @ np.linalg.inv(S)[None, None, :, :]
                                @ x_norm[:, :, :, None] / 2.0) / (2*np.pi*np.sqrt(np.linalg.det(S)))
                Z = -(Z - np.max(Z))[:, :, 0, 0]
                Z /= np.max(Z)
                filter[tmpch, 0, :, :] = Z

                tmp = ch
                chlist = np.array(ch)
    return filter


def train_net(net,
              device,
              writer,
              trial_num,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.15,
              save_cp=True,
              img_scale=1):
    def objective(trial):
        global trial_num
        trial_num += 1

        lr = trial.suggest_loguniform('lr', 1e-07, 1e-04)
        alpha = trial.suggest_uniform('alpha', 0.1, 3.0)
        beta = trial.suggest_uniform('beta', 0.1, 3.0)

        ids = get_ids(dir_sar)

        iddataset = split_train_val(ids, val_percent)

        # writer = SummaryWriter(
        #     comment=f'_Learning_rate_{lr}_Batch_size_{batch_size}')

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {len(iddataset["train"])}
            Validation size: {len(iddataset["val"])}
            Checkpoints:     {save_cp}
            Device:          {device.type}
            Trial id:        {trial_num}
        ''')

        best_valid_accuracy = 10000.0
        n_train = len(iddataset['train'])
        n_val = len(iddataset['val'])
        optimizer = optim.Adam(net.parameters(), lr=lr)
        if net.module.n_classes > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            # criterion = nn.BCEWithLogitsLoss()
            criterion = nn.MSELoss()        # MSE Loss

        loss_min = 100000
        for epoch in range(epochs):
            net.train()

            # reset the generators
            train = get_imgs_and_masks(
                iddataset['train'], dir_sar, dir_cor, dir_gt, img_scale)
            val = get_imgs_and_masks(
                iddataset['val'], dir_sar, dir_cor, dir_gt, img_scale)

            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                epoch_loss = 0
                epoch_loss_leveling = 0
                epoch_loss_reconstruction = 0
                # epoch_loss_input = 0
                for i, b in enumerate(batch(train, batch_size)):
                    imgs_sar = np.array([i[0] for i in b]).astype(np.float32)
                    imgs_cor = np.array([i[1] for i in b]).astype(np.float32)
                    gt = np.array([i[2] for i in b]).astype(np.float32)
                    gt_mask = np.where(gt != 0, 1, 0)  # mask

                    x = y = np.arange(0, 1024, 1)
                    X, Y = np.meshgrid(x, y)
                    S = np.array([[300, 0], [0, 300]])
                    Z = gaussian(X, Y, gt, S)

                    imgs_sar = torch.from_numpy(imgs_sar)
                    imgs_cor = torch.from_numpy(imgs_cor)
                    gt = torch.from_numpy(gt)
                    gt_mask = torch.from_numpy(gt_mask)
                    Z = torch.from_numpy(Z)

                    imgs_sar = imgs_sar.to(device=device)
                    imgs_cor = imgs_cor.to(device=device)
                    gt = gt.to(device=device)
                    gt_mask = gt_mask.to(device=device)
                    Z = Z.to(device=device)

                    # Check input mse_loss
                    # loss_input = criterion(imgs_sar * gt_mask, gt)
                    # epoch_loss_input += loss_input.item()

                    pred_def = net(imgs_sar, imgs_cor)
                    pred_def_mask = pred_def * gt_mask
                    gt_index = torch.nonzero(gt, as_tuple=True)
                    # loss = criterion(pred_def_mask, gt)

                    loss = 0
                    leveling_loss = criterion(pred_def[gt_index], gt[gt_index])
                    reconstruction_loss = criterion(imgs_sar * Z, pred_def * Z)
                    loss = alpha * leveling_loss + \
                        beta * reconstruction_loss
                    epoch_loss += loss.item()
                    epoch_loss_leveling += leveling_loss.item()
                    epoch_loss_reconstruction += reconstruction_loss.item()
                    pbar.set_postfix(**{'Total loss (batch)': loss.item(), 'leveling loss': leveling_loss.item(
                    ), 'reconstruction loss': reconstruction_loss.item()})
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(batch_size)
                logging.info('Train Loss: {}'.format(epoch_loss))
                # writer.add_scalar('Loss/Train', epoch_loss /
                #                   len(iddataset["train"]), epoch+1)
                # writer.add_scalar('Leveling Loss/Train', epoch_loss_leveling /
                #                   len(iddataset["train"]), epoch+1)
                # writer.add_scalar('Reconstruction Loss/Train', epoch_loss_reconstruction /
                #                   len(iddataset["train"]), epoch+1)
                writer.add_scalars('Loss/Train', {'trial_{:02d}_lr_{:.2e}_alpha_{:.2e}_beta_{:.2e}'.format(
                    trial_num, lr, alpha, beta): epoch_loss / len(iddataset["train"])}, epoch+1)
                writer.add_scalars('Leveling Loss/Train', {'trial_{:02d}_lr_{:.2e}_alpha_{:.2e}_beta_{:.2e}'.format(
                    trial_num, lr, alpha, beta): epoch_loss_leveling / len(iddataset["train"])}, epoch+1)
                writer.add_scalars('Reconstruction Loss/Train', {'trial_{:02d}_lr_{:.2e}_alpha_{:.2e}_beta_{:.2e}'.format(
                    trial_num, lr, alpha, beta): epoch_loss_reconstruction / len(iddataset["train"])}, epoch+1)
                writer.flush()

            val_score = eval_net_test(net, val, device, n_val)
            # writer.add_scalar('Loss/Test', val_score, epoch+1)

            writer.add_scalars('Loss/Test', {'trial_{:02d}_lr_{:.2e}_alpha_{:.2e}_beta_{:.2e}'.format(
                trial_num, lr, alpha, beta): val_score}, epoch+1)
            writer.flush()

            if val_score <= best_valid_accuracy:
                best_valid_accuracy = val_score

            # if net.module.n_classes > 1:
            #     logging.info('Validation cross entropy: {}'.format(val_score))
            #     writer.add_scalar('Loss/Validation', val_score, epoch+1)
            # else:
            #     logging.info('Validation MSE: {}'.format(val_score))
            #     writer.add_scalar('Loss/Validation', val_score, epoch+1)

            if save_cp:
                try:
                    data_lr_Bs = dt_now.strftime(
                        '%Y%m%d%H%M')+'_trial_' + str(trial_num) + '_Lr_'+str(lr)+'_Bs_'+str(batch_size)+'_Alpha_'+str(round(alpha, 2))+'_Beta_'+str(round(beta, 2))+'/'
                    os.mkdir(dir_checkpoint + data_lr_Bs)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                # torch.save(net.state_dict(), dir_checkpoint + data_lr_Bs + f'CP_epoch{epoch + 1}.pth')
                if loss_min >= epoch_loss:
                    torch.save(net.module.state_dict(), dir_checkpoint +
                               data_lr_Bs + f'CP_epoch{epoch + 1}.pth')
                    logging.info(f'Checkpoint {epoch + 1} saved !')
                    loss_min = epoch_loss
                else:
                    logging.info(f'Skip Checkpoint {epoch + 1} !')
        # print('Input MSE_Loss:',epoch_loss_input)

        global best_accuracy
        if best_valid_accuracy <= best_accuracy:
            best_accuracy = best_valid_accuracy
            global best_num
            best_num = trial_num

        print('best_valid_accuracy of this trial: {:.3f}'.format(
            best_valid_accuracy))
        print('best_accuracy of trials : {:.3f}'.format(best_accuracy))
        print('best_num of trials: {:.3f}'.format(best_num))
        print('Finished Training')

        return val_score
    return objective


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
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
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
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
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
    net = UNet_fusenet(n_channels=1, n_classes=1)
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

    # Multi-GPU
    net = nn.DataParallel(net).cuda()  # make parallel
    # faster convolutions, but more memory
    torch.backends.cudnn.benchmark = True

    # current_time = dt_now.strftime('%b%d-%Y-%H-%M-%S')
    # log_dir = os.path.join('runs', current_time)
    # writer = SummaryWriter(log_dir=log_dir)
    writer = SummaryWriter(
        log_dir='runs_optuna_reconstruction/2021_0429_20/')
    trial_num = -1  # trial number of optuna
    best_accuracy = 10000.0  # best valid accuracy of optuna trials
    best_num = -1  # best trial number of optuna

try:
    # 最適化のセッションを作る
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(),
        study_name='optuna_reconstruction_lr_and_alpha_beta',
        storage='sqlite:///optuna_database/optuna_reconstruction_lr_and_alpha_beta.db',
        load_if_exists=True
    )

    # 50 回試行する
    study.optimize(
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  writer=writer,
                  trial_num=trial_num),
        n_trials=50)

    print('Best params : {}'.format(study.best_params))
    print('Best value  : {}'.format(study.best_value))
    print('Best trial  : {}'.format(study.best_trial))

    df = study.trials_dataframe()
    print(df)

    df_records = df.to_dict(orient='records')

    for i in range(len(df_records)):
        df_records[i]['datetime_start'] = str(
            df_records[i]['datetime_start'])
        df_records[i]['datetime_complete'] = str(
            df_records[i]['datetime_complete'])
        df_records[i]['duration'] = str(
            df_records[i]['duration'])
        value = df_records[i].pop('value')
        value_dict = {'value': value}
        writer.add_hparams(df_records[i], value_dict)

    writer.flush()
    writer.close()

except KeyboardInterrupt:
    torch.save(net.state_dict(), 'INTERRUPTED_reconst.pth')
    logging.info('Saved interrupt')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
