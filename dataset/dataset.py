# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/10/30 下午1:04
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset.py
# @Software: PyCharm
# Description : None
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import math
import copy
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class KeypointsDataset(Dataset):
    def __init__(self, hyp, mode='train', aug_transforms=None, hflip_prob=0., num_samples=None):
        self.data_dir = hyp['data_dir']
        self.sigma = 2

        self.num_keypoints = 4
        self.num_edges = 2
        self.num_out_channels = hyp['num_keypoints']

        # [Width, Height]
        self.image_size = np.array([1664, 512])
        self.input_size = np.array(hyp['input_size'])
        self.heatmap_size = np.array(hyp['heatmap_size'])

        self.hflip_prob = hflip_prob
        self.mean_rgb = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std_rgb = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'testing' if self.is_test else 'training'

        self.aug_transforms = aug_transforms
        self.images_dir = os.path.join(self.data_dir, sub_folder, 'images')
        self.labels_dir = os.path.join(self.data_dir, sub_folder, 'labels')
        split_txt_path = os.path.join(self.data_dir, 'ImageSets', '{}.txt'.format(mode))
        self.sample_list = [x.strip() for x in open(split_txt_path).readlines()]

        if num_samples is not None:
            self.sample_list = self.sample_list[:num_samples]
        self.num_samples = len(self.sample_list)

        self.data_db = []
        # Find all labels in dir with json format
        file_cnt = 0
        for sample in self.sample_list:
            file_cnt += 1
            if file_cnt % 500 == 0:
                print("{} files are processed!".format(file_cnt))

            self.get_label(sample)
        print("data_db len is %d", len(self.data_db))

    def __len__(self):
        return len(self.data_db)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_image(index)
        else:
            return self.load_image_and_label(index)

    def normalize_image(self, image):
        return (image / 255. - self.mean_rgb) / self.std_rgb

    def load_image(self, index):
        """Load only image for the testing phase"""
        filename = self.sample_list[index]
        image_path, image = self.get_image(filename)

        image = self.normalize_image(image)

        return image_path, image.transpose(2, 0, 1)

    def load_image_and_label(self, index):
        """Load images and targets for the training and validation phase"""
        db_rec = copy.deepcopy(self.data_db[index])
        image_path, image = self.get_image(db_rec['filename'])

        # Apply the augmentation for the raw image
        if self.aug_transforms:
            image = self.aug_transforms(image=image)['image']

        image = self.normalize_image(image)

        keypoints = db_rec['keypoints']
        keypoints_vis = db_rec['keypoints_vis']

        # Find the ROI region and add 16 pixels for each edge
        left_most = int(max(keypoints[:, 0].min() - 16, 0))
        right_most = int(min(keypoints[:, 0].max() + 16, self.image_size[0] - 1))
        width_roi = right_most - left_most

        top_most = int(max(keypoints[:, 1].min() - 16, 0))
        bottom_most = int(min(keypoints[:, 1].max() + 16, self.image_size[1] - 1))
        height_roi = bottom_most - top_most

        # Crop and resize the input image
        image = image[top_most:bottom_most, left_most:right_most]
        image = cv2.resize(image, tuple(self.input_size))

        # Apply the same transform to the coordinates of the ground truth
        # Translation
        keypoints[:, 0] -= left_most
        keypoints[:, 1] -= top_most

        # Scale
        keypoints[:, 0] *= self.input_size[0] / width_roi
        keypoints[:, 1] *= self.input_size[1] / height_roi

        heatmap = np.zeros((self.num_out_channels,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                           dtype=np.float32)

        # Generate the target keypoints heatmap
        target, target_weight = self.generate_target_keypoints(keypoints, keypoints_vis)
        heatmap[0:self.num_keypoints] = target

        # Generate the target edge heatmap
        bottom_left = np.array([keypoints[2][0], keypoints[3][1]])
        heatmap[self.num_keypoints] = self.generate_target_edge(keypoints[1], bottom_left)
        heatmap[self.num_keypoints + 1] = self.generate_target_edge(bottom_left, keypoints[3])

        heatmap = torch.from_numpy(heatmap)
        target_weight = torch.from_numpy(target_weight)

        return image, heatmap, target_weight

    def get_image(self, filename):
        image_path = os.path.join(self.images_dir, '{}.png'.format(filename))
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if image is None:
            logger.error('=> fail to read {}'.format(filename))
            raise ValueError('Fail to read {}'.format(filename))

        return image_path, image

    def get_label(self, filename):
        label_path = os.path.join(self.labels_dir, '{}.json'.format(filename))
        with open(label_path, 'r') as json_file:
            data = json.load(json_file)

            # Find all labels with shape "LShape" in one image
            for obj in data['objects']:
                if obj['shape'] != "LShape":
                    continue

                # Sequence: su -> sl -> ul -> lr
                keypoints = np.zeros((self.num_keypoints, 2), dtype=float)
                keypoints_vis = np.ones([self.num_keypoints], dtype=np.int8)

                keypoints[0] = obj['su']
                keypoints[1] = obj['sl']
                keypoints[2] = obj['ul']
                keypoints[3] = obj['lr']

                # Coordinate of labels less than zero
                if (keypoints < 0).sum() > 0:
                    continue
                # Coordinate of labels out of the image
                if (keypoints[:, 0] >= self.image_size[0]).sum() > 0:
                    continue
                if (keypoints[:, 1] >= self.image_size[1]).sum() > 0:
                    continue

                # Height of the car less than 50, too small
                if keypoints[3][1] - keypoints[2][1] < 50:
                    continue

                # Append the found object to dataset
                self.data_db.append({
                    'filename': filename,
                    'keypoints': keypoints,
                    'keypoints_vis': keypoints_vis,
                })

                if len(self.data_db) % 500 == 0:
                    print("{} boxes are loaded!".format(len(self.data_db)))

    def generate_target_edge(self, start, stop):
        try:
            target_edge = np.zeros([self.heatmap_size[1], self.heatmap_size[0]], dtype=float)

            normal_distrib = [1, 0.7, 0.4, 0.2, 0.1, 0.05]
            feat_stride = self.input_size / self.heatmap_size

            start = np.floor(start / feat_stride).astype(int)
            stop = np.floor(stop / feat_stride).astype(int)

            start = np.minimum(start, self.heatmap_size - 1)
            stop = np.minimum(stop, self.heatmap_size - 1)

            # Using DDA algorithm to generate the line and push back to the queue
            dx = stop[0] - start[0]
            dy = stop[1] - start[1]
            k = dy / dx

            queue = []

            if abs(k) <= 1:
                if start[0] > stop[0]:
                    tmp = start
                    start = stop
                    stop = tmp

                for cx in range(start[0], stop[0]):
                    cy = round(start[1] + k * (cx - start[0]))
                    queue.append((cx, cy, 0))
                    target_edge[cy][cx] = 1
            else:
                if start[1] > stop[1]:
                    tmp = start
                    start = stop
                    stop = tmp

                for cy in range(start[1], stop[1]):
                    cx = round(start[0] + (1 / k) * (cy - start[1]))
                    queue.append((cx, cy, 0))
                    target_edge[cy][cx] = 1

            queue.append((stop[0], stop[1], 0))
            target_edge[stop[1]][stop[0]] = 1

            # u r d l
            dx = [0, 1, 0, -1]
            dy = [-1, 0, 1, 0]

            while len(queue) > 0:
                cx, cy, depth = queue.pop(0)
                for i in range(4):
                    cx += dx[i]
                    cy += dy[i]

                    if cx >= 0 and cy >= 0 and cx < self.heatmap_size[0] and cy < self.heatmap_size[1]:
                        if target_edge[cy][cx] == 0:
                            if depth < 5:
                                queue.append((cx, cy, depth + 1))
                                target_edge[cy][cx] = normal_distrib[depth + 1]

                    cx -= dx[i]
                    cy -= dy[i]

            return target_edge

        except IndexError:
            print("?????????")


    def generate_target_keypoints(self, keypoints, keypoints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_keypoints, 1), dtype=np.float32)
        target_weight[:, 0] = keypoints_vis[:]

        target = np.zeros((self.num_keypoints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_keypoints):
            feat_stride = self.input_size / self.heatmap_size
            mu_x = int(keypoints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(keypoints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
