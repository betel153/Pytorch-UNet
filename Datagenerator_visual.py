# 1. visual_15000.py
# 2. visual_predict.py
# 3. visual_npy_concat.py
# 4. Datagenerator_visual.py    ←


import numpy as np
import scipy.io as sio
import os
import datetime
import h5py
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import requests  # LINE notify
import sys       # LINE notify

dt_now = datetime.datetime.now()

dir_test = '/home/shimosato/dl001/shimosato/dataset/unet/visual_15000/concat/'

# inner_size = 100
# thickness = 20
inner_size = 70
thickness = 10

save_option = 1 # 1: Save  0:NoSave

# output folder
sar_out_dir = dir_test + 'sar/'
cor_out_dir = dir_test + 'cor/'
gt_out_dir = dir_test + 'gt/'

def line_ntfy(mess):
    url = "https://notify-api.line.me/api/notify"
    token = '4aXg0FwLCIbJ0dm3Z4ahi1UZcuoKxjpo8Jn09nlGh25'
    headers = {"Authorization": "Bearer " + token}
    payload = {"message": str(mess)}
    requests.post(url, headers=headers, params=payload)

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def mask_to_heatmap(data, mode, dpi, ab=None):
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.xaxis.tick_top
    ax.set_aspect('equal')
    ax.axis("off")
    if not ab:
        ab = abs(data).max()
    heatmap = ax.pcolor(data, cmap=plt.cm.cool, norm=Normalize(vmin=-ab, vmax=ab))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(heatmap, cax=cax)
    # fig.set_clim(-3,3)
    plt.savefig('image_' + mode + '.png', dpi=dpi)
    return ab

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
    xy = list(zip(*np.where(gt != 0)))

    leveling_dot = []
    # 特定のGTのみ発火させるための措置
    # leveling_dot = [(10896, 3285), (10842, 3686), (11151, 3598), (11421, 3724)]
    # leveling_dot = [(9569, 10249), (10209, 10218), (10009, 10927)]
    # leveling_dot = [(6522, 9020), (7078, 9107)]
    leveling_dot = [(2936, 10751), (3025, 10777), (3275, 11146)]

    # ↓通常はこれ
    # for index in xy:
    #     leveling_dot.append(index)

    sar_bak = sar.copy()
    sar_out_bak = sar_out.copy()

    # GT場所確認用
    sar_for_gt = np.zeros_like(sar)

    for m_x, m_y in leveling_dot:
        for x in range(-(int(inner_size / 2) + thickness), -(int(inner_size / 2) - 1)):
            for y in range(-(int(inner_size / 2) + thickness), int(inner_size / 2) + 1 + thickness):
                if m_x + x >= 0 and m_y + y >= 0 and m_x + x < sar.shape[0] and m_y + y < sar.shape[1]:
                    gt[m_x + x, m_y + y] = gt[m_x, m_y]
                if m_x + y >= 0 and m_y + x >= 0 and m_x + y < sar.shape[0] and m_y + x < sar.shape[1]:
                    gt[m_x + y, m_y + x] = gt[m_x, m_y]
        for x in range(int(inner_size / 2), int(inner_size / 2) + 1 + thickness):
            for y in range(-(int(inner_size / 2) + thickness), int(inner_size / 2) + 1 + thickness):
                if m_x + x >= 0 and m_y + y >= 0 and m_x + x < sar.shape[0] and m_y + y < sar.shape[1]:
                    gt[m_x + x, m_y + y] = gt[m_x, m_y]
                if m_x + y >= 0 and m_y + x >= 0 and m_x + y < sar.shape[0] and m_y + x < sar.shape[1]:
                    gt[m_x + y, m_y + x] = gt[m_x, m_y]
        for x in range(-(int(inner_size / 2) - 1), int(inner_size / 2)):
            for y in range(-(int(inner_size / 2) - 1), int(inner_size / 2)):
                if m_x + x >= 0 and m_y + y >= 0 and m_x + x < sar.shape[0] and m_y + y < sar.shape[1]:
                    sar[m_x + x, m_y + y] = gt[m_x, m_y]
                    sar_out[m_x + x, m_y + y] = gt[m_x, m_y]
                    sar_for_gt[m_x + x, m_y + y] = gt[m_x, m_y]

    gt_mask = np.where(gt != 0, 1, 0)  # mask

    # sar_inout_concat = np.stack([sar, sar_out])
    # IMG option
    # sar_score = (sar - np.amin(sar_cor_concat)) / (np.amax(sar_cor_concat) - np.amin(sar_cor_concat))
    # sar_score = MinMaxScaler(sar)

    # sar_inout_score = sigmoid(sar_inout_concat)
    # 分割
    # sar_score, sar2_score = np.split(sar_inout_score, 2, 0)
    # sar_score = np.squeeze(sar_score)
    # sar2_score = np.squeeze(sar2_score)
    sar_score = np.squeeze(sar)
    sar2_score = np.squeeze(sar_out)
    sar_for_gt_score = np.squeeze(sar_for_gt)

    # お試し　平均x:3104 , y:10948
    sar_score = sar_score[1080:5128, 8924:12972]
    sar2_score = sar2_score[1080:5128, 8924:12972]

    if save_option:
        # result = mask_to_image(sar_score)
        # result.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_in.png')

        ab = 10
        ab = mask_to_heatmap(sar_score, 'input_dpi350_'+ str(ab), 350, ab)
        print("SAR Input saved")

        # sar2_score = (sar_out - np.amin(sar_cor_concat)) / (np.amax(sar_cor_concat) - np.amin(sar_cor_concat))
        # sar2_score = MinMaxScaler(sar_out)
        # sar2_score = zscore(sar_out)
        # sar2_score = sigmoid(sar_out)

        # result2 = mask_to_image(sar2_score)
        # result2.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_out.png')

        ab = 10
        ab = mask_to_heatmap(sar2_score, 'output_dpi350'+ str(ab), 350, ab)
        print("SAR Output saved")

        ab = mask_to_heatmap(sar_for_gt_score, 'gt_dpi350'+ str(ab), 350)
        print("SAR GT saved")

        line_ntfy('SAR End dpi=350')

        # ab = mask_to_heatmap(sar_score, 'input_dpi1000', 1000)
        # print("SAR Input saved")
        # ab = mask_to_heatmap(sar2_score, 'output_dpi1000', 1000, ab)
        # print("SAR Output saved")

        # line_ntfy('SAR End dpi=1000')

        # # GT
        # # gt_minmax = MinMaxScaler(gt)
        # gt_minmax = gt_mask

        # gt_minmax_3d = np.repeat(gt_minmax[:, :, np.newaxis], 3, axis=2)
        # gt_minmax_3d[:, :, (0, 2)] = 0
        # gt_type = (gt_minmax_3d * 255).astype(np.uint8)
        # gt_type_replace = np.where(gt_type==gt_type[0,0,:], 0, gt_type)
        # gt_img = Image.fromarray(gt_type_replace)
        # gt_img.putalpha(255)
        # gt_img.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt.png')
        # print("GT saved")

        # with Image.open(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_in.png') as img:
        #     sar_gray = img.convert("L").convert("RGB")
        #     sar_3d = np.asarray(sar_gray, np.uint8)
        # with Image.open(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_out.png') as img:
        #     sar2_gray = img.convert("L").convert("RGB")
        #     sar2_3d = np.asarray(sar2_gray, np.uint8)

        # gt_cv2 = cv2.imread(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt.png', cv2.IMREAD_UNCHANGED)
        # color_lower = np.array([0, 0, 0, 255])                 # 抽出する色の下限(BGR形式)
        # color_upper = np.array([0, 0, 0, 255])                 # 抽出する色の上限(BGR形式)
        # gt_cv2_mask = cv2.inRange(gt_cv2, color_lower, color_upper)    # 範囲からマスク画像を作成
        # gt_cv2_bool = cv2.bitwise_not(gt_cv2, gt_cv2, mask=gt_cv2_mask)      # 元画像とマスク画像の演算(背景を白くする)
        # cv2.imwrite(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt_alpha.png', gt_cv2_bool)                         # 画像保存

        # sar_syn = sar_3d * (1 - gt_cv2_bool[:,:,3:]/255) + gt_cv2_bool[:,:,:3] * (gt_cv2_bool[:,:,3:]/255)
        # sar_syn_result = Image.fromarray(sar_syn.astype(np.uint8))
        # sar_syn_result.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_syn_in.png')

        # sar2_syn = sar2_3d * (1 - gt_cv2_bool[:,:,3:]/255) + gt_cv2_bool[:,:,:3] * (gt_cv2_bool[:,:,3:]/255)
        # sar2_syn_result = Image.fromarray(sar2_syn.astype(np.uint8))
        # sar2_syn_result.save(gt_out_dir + 'jpg/' + fn + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_syn_out.png')

# check = []
# import csv # csvモジュールをインポート

# for m_x, m_y in leveling_dot:
#     # if m_x >= top and m_x <= bottom and m_y >= left and m_y <= right:
#     check.append([m_y, m_x, gt[m_x, m_y], sar_bak[m_x, m_y], sar_out_bak[m_x, m_y]])
# # print(check)
# with open("ibaraki_leveling.csv", "w", encoding="Shift_jis") as f: # 文字コードをShift_JISに指定
#     writer = csv.writer(f, lineterminator="\n") # writerオブジェクトの作成 改行記号で行を区切る
#     writer.writerows(check) # csvファイルに書き込み
# print('End')
# line_ntfy('End')
