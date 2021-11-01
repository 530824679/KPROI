# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/10/30 下午12:54
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : visualize.py
# @Software: PyCharm
# Description : None
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import math
import copy
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from utils.postprocess import get_max_preds
plt.switch_backend('agg')

def plot_lr_scheduler(optimizer, scheduler, num_epochs=300, save_dir='', lr_type=''):
    # Plot LR simulating training for full num_epochs
    optimizer, scheduler = copy.copy(optimizer), copy.copy(scheduler)  # do not modify originals
    y = []
    for _ in range(num_epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, num_epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'LR_{}.png'.format(lr_type)), dpi=200)

def save_batch_image_with_keypoints(batch_image, batch_keypoints, batch_keypoints_vis, filename, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_keypoints[k]
            joints_vis = batch_keypoints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(filename, ndarr)


def draw_3D_on_cv_image(image, preds):
    preds = preds.astype(int)
    bottom_left = np.array([preds[2][0], preds[3][1]]).astype(int)
    top_right = np.array([preds[3][0], preds[2][1]]).astype(int)
    cv2.line(image, tuple(preds[0]), tuple(preds[1]), (255, 127, 0), 1, cv2.LINE_AA)

    if preds[1][0] < preds[2][0]:
        cv2.line(image, tuple(bottom_left), tuple(preds[1]), (255, 64, 0), 1, cv2.LINE_AA)
        cv2.line(image, tuple(preds[2]), tuple(preds[0]), (255, 64, 0), 1, cv2.LINE_AA)
    elif preds[1][0] > preds[3][0]:
        cv2.line(image, tuple(preds[3]), tuple(preds[1]), (255, 64, 0), 1, cv2.LINE_AA)
        cv2.line(image, tuple(top_right), tuple(preds[0]), (255, 64, 0), 1, cv2.LINE_AA)

    cv2.line(image, tuple(bottom_left), tuple(preds[2]), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, tuple(bottom_left), tuple(preds[3]), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, tuple(top_right), tuple(preds[2]), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, tuple(top_right), tuple(preds[3]), (255, 0, 0), 1, cv2.LINE_AA)

    return image

def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True, draw_3D=False):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_keypoints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size * heatmap_height, (num_keypoints+1) * heatmap_width, 3), dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_keypoints):
            cv2.circle(resized_image, (int(preds[i][j][0]), int(preds[i][j][1])), 1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image, (int(preds[i][j][0]), int(preds[i][j][1])), 1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

        if draw_3D:
            resized_image = draw_3D_on_cv_image(resized_image, preds[i])

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)

def save_debug_images(hyp, input, meta, target, joints_pred, output, prefix):
    if not hyp['debug']:
        return

    if hyp['save_batch_images_gt']:
        save_batch_image_with_keypoints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if hyp['save_batch_images_pred']:
        save_batch_image_with_keypoints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if hyp['save_heatmaps_gt']:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if hyp['save_heatmaps_pred']:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )