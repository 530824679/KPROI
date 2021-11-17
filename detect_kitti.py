# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/11/1 下午2:03
# @Author  : Jingyi Wang
# @Email   : Wangjingyi70@gmail.com
# @File    : detect.py
# @Software: PyCharm
# Description : None
# --------------------------------------
import json
import os
import cv2
import numpy as np

import os
import sys
import time
import yaml                                         # yaml配置模块
import torch
import logging                                      # 日志模块
import argparse
import os.path as osp
from tqdm import tqdm
from pathlib import Path                            # 路径操作模块
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from valid import validate
from dataset.dataloader import create_train_dataloader, create_val_dataloader
from utils.misc import AverageMeter, ProgressMeter
from loss.loss import KeyPointsMSELoss
from loss.optimizer import create_optimizer, create_lr_scheduler
from models.kpnet import build_model
from models.basenet import parameters_num
from utils.general import to_python_float, get_latest_run, check_file, colorstr, increment_dir
from utils.evaluate import calc_acc
from utils.torch_utils import select_device, save_checkpoint, get_saved_state
from utils.postprocess import *
from utils.visualize import *
from utils.kitti_utils import *

mean_rgb = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std_rgb = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
def normalize_image(image):
    return (image / 255. - mean_rgb) / std_rgb


def predict_keypoints_from_roi(model, img_roi):
    model.eval()
    with torch.no_grad():
        heatmaps = model(torch.FloatTensor(img_roi))
        preds, _ = get_max_preds(heatmaps.detach().cpu().numpy())
        preds = preds[0,: 4,:] *4 + 2
        return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--config', type=str, default='./config/hyp.yaml',
                        help='the path of the pretrained weights')
    parser.add_argument('--device', type=str, default='cpu', help='number of select device')
    opt = parser.parse_args()

    hyp = check_file(opt.config)
    assert len(hyp), '--hyp file must be specified'

    # 载入初始超参
    with open(hyp, encoding='UTF-8') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    device = select_device(opt.device, batch_size=hyp['batch_size'])
    model = build_model(hyp['pretrained'], hyp['num_keypoints'], is_train=False)
    model.load_state_dict(torch.load("./saved_weights/Epoch20_RGB.pth"))

    test_image = hyp["test_image_kitti"]
    test_label = hyp['test_label_kitti']
    calib_path = "F:\\Kitti\\object\\training\\calib"
    input_size = np.array(hyp["input_size"])
    heatmap_size = np.array(hyp["heatmap_size"])

    frame_cnt = 0
    for filename in os.listdir(test_label):
        if not filename.endswith(".txt"):
            continue

        frame_cnt += 1

        image_path = os.path.join(test_image, '{}.png'.format(os.path.splitext(filename)[0]))
        image_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_size = np.array([image.shape[1], image.shape[0]])

        with open(os.path.join(test_label, filename), 'r') as label_file:

            # Find all labels with shape "LShape" in one image
            for line in label_file:
                values = line.split()
                if values[0] != "Car" and values[0] != "Van" and values[0] != "Truck":
                    continue

                box3d = Box3D(line)

                with open(os.path.join(calib_path, '{}.txt'.format(os.path.splitext(filename)[0])), 'r') as calib_file:
                    # Find all labels with shape "LShape" in one image
                    calib_lines = calib_file.readlines()
                    P2 = np.array([float(calib_lines[2].split()[i]) for i in range(1, 13)]).reshape((3, 4))
                    R0_rect = np.array([float(calib_lines[4].split()[i]) for i in range(1, 10)]).reshape((3, 3))
                    R0_rect = np.column_stack((np.row_stack((R0_rect, np.zeros(3))), np.zeros(4)))
                    R0_rect[3][3] = 1

                box3d_pts = project_to_image(box3d.in_camera_coordinate(), P2)
                image_bgr = draw_projected_box3d(image_bgr, box3d_pts)

                if filename == '000057.txt':
                    print("Box 3D:")
                    for i in range(8):
                        print(box3d_pts[:, i])

                continue

                # Find the ROI region and add 16 pixels for each edge
                left_most = int(max(round(float(values[4])) - 16, 0))
                right_most = int(min(round(float(values[6])) + 16, image_size[0] - 1))
                width_roi = right_most - left_most

                top_most = int(max(round(float(values[5])) - 16, 0))
                bottom_most = int(min(round(float(values[7])) + 16, image_size[1] - 1))
                height_roi = bottom_most - top_most

                if height_roi < 50 + 16 * 2:
                    continue

                # Crop and resize the input image
                img_roi = image[top_most:bottom_most, left_most:right_most]
                img_roi = cv2.resize(img_roi, tuple(input_size))

                # Apply norm for the cropped image
                img_roi = normalize_image(img_roi)
                img_roi = img_roi.transpose(2, 0, 1).reshape([1, 3, input_size[1], input_size[0]])

                # Predict!
                keypoints = predict_keypoints_from_roi(model, img_roi)
                keypoints[:, 0] = keypoints[:, 0] / input_size[0] * width_roi + left_most
                keypoints[:, 1] = keypoints[:, 1] / input_size[1] * height_roi + top_most

                image_bgr = draw_3D_on_cv_image(image_bgr, keypoints)

        cv2.imshow(":)", image_bgr)
        cv2.waitKey(50)
        # cv2.imwrite(os.path.join("Test", os.path.splitext(filename)[0] + ".png"), image_bgr)
    print("888")