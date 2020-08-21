import numpy as np
import scipy.io as sio
import os
import datetime
import h5py
import cv2
from PIL import Image
import matplotlib.pyplot as plt

dt_now = datetime.datetime.now()

base_dir = '/home/shimosato/dataset/unet/'
# crop_size = (4096, 4096)
crop_size = (2048, 2048)

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

ids = []
for f in os.listdir(input_dir):
    if not f.startswith('.'):
        if not '_' in f:
            ids.append(os.path.splitext(f)[0])

for id in ids:
    print(id)

    # Read HDF5 Format File
    with h5py.File(input_dir + id + '.mat', 'r') as f:
        sar = np.transpose(f['data'][()].astype(np.float32), axes=[1, 0])
    with h5py.File(input_dir + id + '_cor.mat', 'r') as f:
        cor = np.transpose(f['data'][()].astype(np.float32), axes=[1, 0])
    with h5py.File(input_dir + id + '_leveling.mat', 'r') as f:
        gt = np.transpose(f['data'][()].astype(np.float32), axes=[1, 0])
    with h5py.File(input_dir + id + '_xy.mat', 'r') as f:
        xy = np.transpose(f['data'][()].astype(np.int32), axes=[1, 0])
        xy -= 1 # MATLABの時点で水準点のXY座標が全て1ずつずれているため補正
    h, w = gt.shape

    leveling_dot = []

    for index in xy:
        if index.all() and index[0] < h and index[1] < w:
            leveling_dot.append(index)

    leveling_dot_ag = []
    for m_x, m_y in leveling_dot:
        for x in range(-30, 31):
            for y in range(-30, 31):
                leveling_dot_ag.append([m_x + x, m_y + y])
                # gt[m_x + x, m_y + y] = gt[m_x, m_y]
        for x in range(-30, -19):
            for y in range(-30, 31):
                gt[m_x + x, m_y + y] = gt[m_x, m_y]
                gt[m_x + y, m_y + x] = gt[m_x, m_y]
        for x in range(20, 31):
            for y in range(-30, 31):
                gt[m_x + x, m_y + y] = gt[m_x, m_y]
                gt[m_x + y, m_y + x] = gt[m_x, m_y]

    print('　Creating ...')
    for i, center in enumerate(leveling_dot_ag):
        sar_output = []
        c_h, c_w = center  # GT's center
        # decide top and left
        top = c_h - np.random.randint(0, crop_size[0] // 2)
        left = c_w - np.random.randint(0, crop_size[1] // 2)
        # decide bottom and right
        bottom = top + crop_size[0]
        right = left + crop_size[1]

        count = 0
        while True:
            if left < 0:
                left =  - left
                left += np.random.randint(0, w - right)
                right = left + crop_size[1]
            if right > w:
                left -= right - w
                left -= np.random.randint(0, left)
                right = left + crop_size[1]
            if top < 0:
                top = - top
                top += np.random.randint(0, h - bottom)
                bottom = top + crop_size[1]
            if bottom > h:
                top -= right - h
                top -= np.random.randint(0, top)
                bottom = top + crop_size[1]
            if c_h < top or c_h > bottom or c_w < left or c_w > right:
                top = c_h - np.random.randint(crop_size[0] // 4, crop_size[0] // 2)
                left = c_w - np.random.randint(crop_size[1] // 4, crop_size[1] // 2)
                # decide bottom and right
                bottom = top + crop_size[0]
                right = left + crop_size[1]
            if left >= 0 and right <= w and top >= 0 and bottom <= h and c_h >= top and c_h <= bottom and c_w >= left and c_w <= right:
                # print('Save now: ', id + '_' + str(i))
                gt_test = np.zeros(gt.shape)
                gt_test[c_h, c_w] = gt[c_h, c_w]
                sar_score = sar

                # SAR
                print("最小：", np.min(sar[top:bottom, left:right]))
                print("中央：", np.median(sar[top:bottom, left:right]))
                print("平均：", np.mean(sar[top:bottom, left:right]))
                print("最大：", np.max(sar[top:bottom, left:right]))
                result = mask_to_image(sar[top:bottom, left:right])
                result.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar.jpg')

                # sar[top:bottom, left:right] = np.clip(sar[top:bottom, left:right], -40, 40)
                # sar[top:bottom, left:right] = np.clip(sar[top:bottom, left:right], -30, 30)
                sar_score[top:bottom, left:right] = zscore(sar[top:bottom, left:right])

                # sar[top:bottom, left:right] = denormalize(sar[top:bottom, left:right], lower=-5, upper=8)
                print("After: clip and zscore")
                print("最小：", np.min(sar_score[top:bottom, left:right]))
                print("中央：", np.median(sar_score[top:bottom, left:right]))
                print("平均：", np.mean(sar_score[top:bottom, left:right]))
                print("最大：", np.max(sar_score[top:bottom, left:right]))
                result = mask_to_image(sar_score[top:bottom, left:right])
                result.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_zscore.jpg')

                # gt2 = to_grayscale(sar[top:bottom, left:right], -5, 5)
                # sar[top:bottom, left:right] = sigmoid(sar[top:bottom, left:right])
                sar_score[top:bottom, left:right] = MinMaxScaler(sar[top:bottom, left:right])
                sar_score2 = MinMaxScaler(sar[top:bottom, left:right])
                print("After: sigmoid")
                print("最小：", np.min(sar_score[top:bottom, left:right]))
                print("中央：", np.median(sar_score[top:bottom, left:right]))
                print("平均：", np.mean(sar_score[top:bottom, left:right]))
                print("最大：", np.max(sar_score[top:bottom, left:right]))
                result = mask_to_image(sar_score[top:bottom, left:right])
                result.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_minmax.jpg')
                # plt.imshow((sar[top:bottom, left:right] * 255).astype(np.uint8), "gray")  # グレースケールで
                # # plt.imshow(gt2, "gray")  # グレースケールで
                # plt.colorbar()  # カラーバーの表示
                # plt.show()
                # result.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '.jpg')

                # GT
                # gt_score = []
                # gt_score_minmax = []
                print("GT")
                print("最小：", np.min(gt[top:bottom, left:right]))
                print("中央：", np.median(gt[top:bottom, left:right]))
                print("平均：", np.mean(gt[top:bottom, left:right]))
                print("最大：", np.max(gt[top:bottom, left:right]))

                sar_R = np.repeat(gt[top:bottom, left:right][:, :, np.newaxis], 3, axis=2)
                sar_R[:, :, (1, 2)] = 0
                result = mask_to_colorimage(sar_R)
                # result.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt.jpg')

                # gt[top:bottom, left:right] = np.clip(gt[top:bottom, left:right], -40, 40)
                # gt[top:bottom, left:right] = np.clip(gt[top:bottom, left:right], -30, 30)
                gt_score = zscore(gt[top:bottom, left:right])
                # gt[top:bottom, left:right] = denormalize(gt[top:bottom, left:right], lower=-5, upper=8)
                print("After: clip and zscore")
                print("最小：", np.min(gt_score))
                print("中央：", np.median(gt_score))
                print("平均：", np.mean(gt_score))
                print("最大：", np.max(gt_score))

                sar_gt_R = np.repeat(gt_score[:, :, np.newaxis], 3, axis=2)
                sar_gt_R[:, :, (0, 1)] = 0
                result = mask_to_colorimage(sar_gt_R)
                # result.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt_zscore.jpg')

                # gt2 = to_grayscale(gt[top:bottom, left:right], -5, 5)
                # gt[top:bottom, left:right] = sigmoid(gt[top:bottom, left:right])
                gt_score_minmax = MinMaxScaler(gt[top:bottom, left:right])
                print("After: sigmoid")
                print("最小：", np.min(gt_score_minmax))
                print("中央：", np.median(gt_score_minmax))
                print("平均：", np.mean(gt_score_minmax))
                print("最大：", np.max(gt_score_minmax))

                sar_gt_R = np.repeat(gt_score_minmax[:, :, np.newaxis], 3, axis=2)
                sar_gt_R[:, :, (0, 2)] = 0

                result = (sar_gt_R * 255).astype(np.uint8)
                t2 = np.where(result==result[0,0,:], 0, result)
                result = Image.fromarray(t2)
                result.putalpha(255)
                result.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt_minmax.png')

                # sar_R = np.repeat(sar_score2[:, :, np.newaxis], 3, axis=2)
                # sar_R = sar_R.astype(np.uint8)
                # sar_vi_result2 = Image.fromarray(sar_R)
                # sar_vi_result2.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt_minmax_original.jpg')
                # sar_R = np.array(Image.open(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_minmax.jpg'))
                with Image.open(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_minmax.jpg') as img:
                    gray = img.convert("L").convert("RGB")
                    array = np.asarray(gray, np.uint8)
                img = cv2.imread(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt_minmax.png', cv2.IMREAD_UNCHANGED)
                color_lower = np.array([0, 0, 0, 255])                 # 抽出する色の下限(BGR形式)
                color_upper = np.array([0, 0, 0, 255])                 # 抽出する色の上限(BGR形式)
                img_mask = cv2.inRange(img, color_lower, color_upper)    # 範囲からマスク画像を作成
                img_bool = cv2.bitwise_not(img, img, mask=img_mask)      # 元画像とマスク画像の演算(背景を白くする)
                cv2.imwrite(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_sar_alpha.png', img_bool)                         # 画像保存

                sar_vi = array * (1 - img_bool[:,:,3:]/255) + img_bool[:,:,:3] * (img_bool[:,:,3:]/255)
                # sar_vi = array * 0.7 + t2 * 0.3
                sar_vi_result = Image.fromarray(sar_vi.astype(np.uint8))
                sar_vi_result.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt_minmax_2.jpg')

                # result = mask_to_image(gt[top:bottom, left:right])
                # plt.imshow((gt[top:bottom, left:right] * 255).astype(np.uint8), "gray")  # グレースケールで
                # # plt.imshow(gt2, "gray")  # グレースケールで
                # plt.colorbar()  # カラーバーの表示
                # plt.show()
                # result.save(gt_out_dir + 'jpg/' + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_gt.jpg')

                # check = []
                # for m_x, m_y in leveling_dot:
                #     if m_x >= top and m_x <= bottom and m_y >= left and m_y <= right:
                #         check.append([m_x - top, m_y - left, gt[m_x, m_y], t[m_x - top, m_y - left]])

                np.save(sar_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M'), sar[top:bottom, left:right])
                np.save(cor_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_cor', cor[top:bottom, left:right])
                np.save(gt_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_leveling', gt[top:bottom, left:right])
                # np.save(gt_out_dir + id + '_' + str(i) + '_' + dt_now.strftime('%Y%m%d%H%M') + '_leveling', gt_test[top:bottom, left:right])
                break
            if count > 20:
                print('Error: ', id + '_' + str(i))
                break
            count += 1
