import numpy as np
import scipy.io as sio
import os
import datetime
import h5py
import cv2
from PIL import Image
import matplotlib.pyplot as plt

dt_now = datetime.datetime.now()

dir_test = '/home/shimosato/dataset/unet/train_visual/'
base_dir = '/home/shimosato/dataset/unet/'

# input folder
input_dir = base_dir + 'test/raw/'

# output folder
sar_out_dir = base_dir + 'train_visual/sar/'
cor_out_dir = base_dir + 'train_visual/cor/'
gt_out_dir = base_dir + 'train_visual/gt/'

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def mask_to_colorimage(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def MinMaxScaler(sar):
    return (sar - np.amin(sar)) / (np.amax(sar) - np.amin(sar))

def sigmoid(sar):
    return 1 / (1 + np.exp(-sar))

def normalize(values, lower, upper):
    return (values - lower) / (upper - lower)

def denormalize(values, lower=0, upper=1):
    return values * (upper - lower) + lower

def to_grayscale(values, lower, upper):
    normalized = normalize(values, lower, upper)
    denormalized = np.clip(denormalize(normalized, 0, 255), 0, 255)
    return denormalized.astype(np.uint8)

def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (os.path.splitext(f)[0] for f in os.listdir(dir) if not f.startswith('.'))

for i, fn in enumerate(list(get_ids(dir_test + 'sar/'))):
    print('FileName:', fn)

    # Load npy file
    sar = np.load(dir_test + 'sar/' + fn + '.npy')
    sar_out = np.load(dir_test + 'out/' + fn + '_OUT.npy')
    gt = np.load(dir_test + 'gt/' + fn + '_leveling.npy')
    gt_mask = np.where(gt != 0, 1, 0)  # mask
    xy = list(zip(*np.where(gt != 0)))

    leveling_dot = []
    for index in xy:
        leveling_dot.append(index)

    leveling_dot_ag = []
    for m_x, m_y in leveling_dot:
        for x in range(-30, 31):
            for y in range(-30, 31):
                leveling_dot_ag.append([m_x + x, m_y + y])
                # gt[m_x + x, m_y + y] = gt[m_x, m_y]
        for x in range(-30, -19):
            for y in range(-30, 31):
                if m_x + x >= 0 and m_y + y >= 0 and m_x + x < sar.shape[0] and m_y + y < sar.shape[1]:
                    gt[m_x + x, m_y + y] = gt[m_x, m_y]
                if m_x + y >= 0 and m_y + x >= 0 and m_x + y < sar.shape[0] and m_y + x < sar.shape[1]:
                    gt[m_x + y, m_y + x] = gt[m_x, m_y]
        for x in range(20, 31):
            for y in range(-30, 31):
                if m_x + x >= 0 and m_y + y >= 0 and m_x + x < sar.shape[0] and m_y + y < sar.shape[1]:
                    gt[m_x + x, m_y + y] = gt[m_x, m_y]
                if m_x + y >= 0 and m_y + x >= 0 and m_x + y < sar.shape[0] and m_y + x < sar.shape[1]:
                    gt[m_x + y, m_y + x] = gt[m_x, m_y]

    sar_score = MinMaxScaler(sar)
    # sar_score = sigmoid(sar)
    result = mask_to_image(sar_score)
    result.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_in.png')
    print("SAR Input saved")

    # sar2_score = MinMaxScaler(sar_out)
    # sar2_score = zscore(sar_out)
    sar2_score = sigmoid(sar_out)

    result2 = mask_to_image(sar2_score)
    result2.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_out.png')
    print("SAR Output saved")

    # GT
    gt_minmax = MinMaxScaler(gt)

    gt_minmax_3d = np.repeat(gt_minmax[:, :, np.newaxis], 3, axis=2)
    gt_minmax_3d[:, :, (0, 2)] = 0
    gt_type = (gt_minmax_3d * 255).astype(np.uint8)
    gt_type_replace = np.where(gt_type==gt_type[0,0,:], 0, gt_type)
    gt_img = Image.fromarray(gt_type_replace)
    gt_img.putalpha(255)
    gt_img.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt.png')
    print("GT saved")

    with Image.open(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_in.png') as img:
        sar_gray = img.convert("L").convert("RGB")
        sar_3d = np.asarray(sar_gray, np.uint8)
    with Image.open(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_out.png') as img:
        sar2_gray = img.convert("L").convert("RGB")
        sar2_3d = np.asarray(sar2_gray, np.uint8)

    gt_cv2 = cv2.imread(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt.png', cv2.IMREAD_UNCHANGED)
    color_lower = np.array([0, 0, 0, 255])                 # 抽出する色の下限(BGR形式)
    color_upper = np.array([0, 0, 0, 255])                 # 抽出する色の上限(BGR形式)
    gt_cv2_mask = cv2.inRange(gt_cv2, color_lower, color_upper)    # 範囲からマスク画像を作成
    gt_cv2_bool = cv2.bitwise_not(gt_cv2, gt_cv2, mask=gt_cv2_mask)      # 元画像とマスク画像の演算(背景を白くする)
    cv2.imwrite(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt_alpha.png', gt_cv2_bool)                         # 画像保存

    sar_syn = sar_3d * (1 - gt_cv2_bool[:,:,3:]/255) + gt_cv2_bool[:,:,:3] * (gt_cv2_bool[:,:,3:]/255)
    sar_syn_result = Image.fromarray(sar_syn.astype(np.uint8))
    sar_syn_result.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_syn_in.png')

    sar2_syn = sar2_3d * (1 - gt_cv2_bool[:,:,3:]/255) + gt_cv2_bool[:,:,:3] * (gt_cv2_bool[:,:,3:]/255)
    sar2_syn_result = Image.fromarray(sar2_syn.astype(np.uint8))
    sar2_syn_result.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_syn_out.png')

    check = []
    # for m_x, m_y in leveling_dot:
    #     if m_x >= top and m_x <= bottom and m_y >= left and m_y <= right:
    #         check.append([m_x - top, m_y - left, gt[m_x, m_y], t[m_x - top, m_y - left]])
