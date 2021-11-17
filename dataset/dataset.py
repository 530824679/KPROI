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
from utils.kitti_utils import *

logger = logging.getLogger(__name__)

def dist(x1, y1, x2, y2, x3, y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = (dx*dx + dy*dy)**.5

    return dist

class KeypointsDataset(Dataset):
    def __init__(self, hyp, mode='train', aug_transforms=None, hflip_prob=0., num_samples=None):
        self.sigma = 2
        self.hyp = hyp
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

        self.aug_transforms = aug_transforms

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.test_mode = False
        self.data_db = []

        if hyp["use_dataset"] == "kitti_train":
            self.images_dir = hyp["train_image_kitti"]
            self.labels_dir = hyp["train_label_kitti"]
            self.calibs_dir = hyp["train_calib_kitti"]

            for filename in os.listdir(self.labels_dir):
                if not filename.endswith(".txt"):
                    continue

                # image_path = os.path.join(self.images_dir, '{}.png'.format(os.path.splitext(filename)[0]))
                # image_bgr = cv2.imread(image_path)
                # image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                # image_size = np.array([image.shape[1], image.shape[0]])

                with open(os.path.join(self.labels_dir, filename), 'r') as label_file:

                    # Find all labels with shape "LShape" in one image
                    for line in label_file:
                        values = line.split()
                        if values[0] != "Car" and values[0] != "Van" and values[0] != "Truck":
                            continue

                        # Truncated between 0. to 1.
                        if float(values[1]) > 0.5:
                            continue

                        # Occlusion between 0 to 2, and 3 means unknown
                        if int(values[2]) >= 2:
                            continue

                        box3d = Box3D(line)

                        with open(os.path.join(self.calibs_dir, '{}.txt'.format(os.path.splitext(filename)[0])),
                                  'r') as calib_file:
                            # Find all labels with shape "LShape" in one image
                            calib_lines = calib_file.readlines()
                            P2 = np.array([float(calib_lines[2].split()[i]) for i in range(1, 13)]).reshape((3, 4))
                            R0_rect = np.array([float(calib_lines[4].split()[i]) for i in range(1, 10)]).reshape((3, 3))
                            R0_rect = np.column_stack((np.row_stack((R0_rect, np.zeros(3))), np.zeros(4)))
                            R0_rect[3][3] = 1

                        box3d_pts = project_to_image(box3d.in_camera_coordinate(), P2)
                        box3d_pts = box3d_pts.T

                        if box3d_pts[:, 1].max() - box3d_pts[:, 1].min() < 16:
                            continue

                        # Append the found object to dataset
                        self.data_db.append({
                            'filename': os.path.splitext(filename)[0],
                            'keypoints': box3d_pts,
                            'keypoints_vis': np.ones([self.num_keypoints], dtype=np.int8),
                            "ob_angle": float(values[3])
                        })


        else:
            if self.mode == "train":
                self.images_dir = hyp["train_image"]
                self.labels_dir = hyp["train_label"]

            if self.mode == "test":
                self.test_mode = True
                self.images_dir = hyp["test_image"]
                self.labels_dir = hyp["test_label"]

            index_txt = os.path.join(self.images_dir, 'index.txt')
            self.filename_list = [x.strip() for x in open(index_txt).readlines()]

            if num_samples is not None:
                self.filename_list = self.filename_list[:num_samples]
            self.num_files = len(self.filename_list)

            # Find all labels in dir with json format
            file_cnt = 0
            for filename in self.filename_list:
                file_cnt += 1
                self.append_dataset(filename)

                if file_cnt % 500 == 0:
                    print("{} files are processed!".format(file_cnt))

                if len(self.data_db) % 500 == 0:
                    print("{} boxes are loaded!".format(len(self.data_db)))

            print("Dataset length is %d, number of files is %d", len(self.data_db), self.num_files)

    def __len__(self):
        return len(self.data_db)

    def __getitem__(self, index):
        if self.hyp["use_dataset"] == "kitti_train":
            return self.load_image_and_label_kitti(index)
        if self.test_mode:
            return self.load_image(index)
        else:
            return self.load_image_and_label(index)

# ----- Init the dataset ----- #

    def append_dataset(self, filename):
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
                if keypoints[3][1] - keypoints[2][1] < 25:
                    continue

                # Append the found object to dataset
                self.data_db.append({
                    'filename': filename,
                    'keypoints': keypoints,
                    'keypoints_vis': keypoints_vis,
                })

# ----- Get item methods ----- #

    def get_image(self, filename):
        image_path = os.path.join(self.images_dir, '{}.png'.format(filename))
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if image is None:
            logger.error('=> fail to read {}'.format(filename))
            raise ValueError('Fail to read {}'.format(filename))

        return image_path, image

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

        # Apply the augmentation for the cropped image
        if self.aug_transforms:
            image = self.aug_transforms(image=image)['image']
        # Apply norm for the cropped image
        image = self.normalize_image(image)

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

        # Left or right target edge
        if keypoints[1][0] < keypoints[2][0]:
            heatmap[self.num_keypoints] = self.generate_target_edge(keypoints[1], bottom_left)
        elif keypoints[1][0] > keypoints[3][0]:
            heatmap[self.num_keypoints] = self.generate_target_edge(keypoints[1], keypoints[3])

        heatmap[self.num_keypoints + 1] = self.generate_target_edge(bottom_left, keypoints[3])

        heatmap = torch.from_numpy(heatmap)
        target_weight = torch.from_numpy(target_weight)

        return image.transpose(2, 0, 1), heatmap, target_weight

    def load_image_and_label_kitti(self, index):
        """Load images and targets for the training and validation phase"""
        db_rec = copy.deepcopy(self.data_db[index])
        image_path, image = self.get_image(db_rec['filename'])
        image_size = np.array([image.shape[1], image.shape[0]])
        ob_angle = db_rec["ob_angle"]

        box3d_pts = db_rec['keypoints']
        keypoints_vis_fake = db_rec['keypoints_vis']

        left_most = box3d_pts[:, 0].min()
        right_most = box3d_pts[:, 0].max()
        width_roi = right_most - left_most

        # Find the ROI region and add 1/10 of width for each edge
        left_most = int(max(left_most - 0.1 * width_roi, 0))
        right_most = int(min(right_most + 0.1 * width_roi, image_size[0] - 1))
        width_roi = right_most - left_most

        top_most = box3d_pts[:, 1].min()
        bottom_most = box3d_pts[:, 1].max()
        height_roi = bottom_most - top_most

        top_most = int(max(top_most - 0.1 * height_roi, 0))
        bottom_most = int(min(bottom_most + 0.1 * height_roi, image_size[1] - 1))
        height_roi = bottom_most - top_most

        # Crop and resize the input image
        image = image[top_most:bottom_most, left_most:right_most]
        image = cv2.resize(image, tuple(self.input_size))

        # Apply the augmentation for the cropped image
        if self.aug_transforms:
            image = self.aug_transforms(image=image)['image']
        # Apply norm for the cropped image
        image = self.normalize_image(image)

        f_area = np.linalg.norm(np.cross(box3d_pts[5] - box3d_pts[4], box3d_pts[0] - box3d_pts[4]))
        b_area = np.linalg.norm(np.cross(box3d_pts[6] - box3d_pts[7], box3d_pts[3] - box3d_pts[7]))

        # Sequence: su -> sl -> ul -> lr
        keypoints = np.zeros((self.num_keypoints, 2), dtype=float)

        heatmap = np.zeros((self.num_out_channels,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                           dtype=np.float32)

        # Apply the same transform to the coordinates of the ground truth
        # Translation
        box3d_pts[:, 0] -= left_most
        box3d_pts[:, 1] -= top_most

        # Scale
        box3d_pts[:, 0] *= self.input_size[0] / width_roi
        box3d_pts[:, 1] *= self.input_size[1] / height_roi

        if ob_angle < 0:
            if ob_angle > -1.52:
                # We can see rear and right surface
                keypoints[0] = box3d_pts[5]
                keypoints[1] = box3d_pts[1]
                heatmap[self.num_keypoints] = self.generate_target_edge(box3d_pts[1], box3d_pts[2])
            elif ob_angle < -1.62:
                # We can see rear and left surface
                keypoints[0] = box3d_pts[4]
                keypoints[1] = box3d_pts[0]
                heatmap[self.num_keypoints] = self.generate_target_edge(box3d_pts[0], box3d_pts[3])
            else:
                # We can only see rear surface
                # Assign -1 means not generate on the heatmap
                keypoints[0] = -np.ones(2) * 666
                keypoints[1] = -np.ones(2) * 666
            keypoints[2] = box3d_pts[7]
            keypoints[3] = box3d_pts[2]
            heatmap[self.num_keypoints + 1] = self.generate_target_edge(box3d_pts[3], box3d_pts[2])
        else:
            if ob_angle > 1.62:
                # We can see front and left surface
                keypoints[0] = box3d_pts[7]
                keypoints[1] = box3d_pts[3]
                heatmap[self.num_keypoints] = self.generate_target_edge(box3d_pts[0], box3d_pts[3])
            elif ob_angle < 1.52:
                # We can see front and right surface
                keypoints[0] = box3d_pts[6]
                keypoints[1] = box3d_pts[2]
                heatmap[self.num_keypoints] = self.generate_target_edge(box3d_pts[1], box3d_pts[2])
            else:
                # We can only see front surface
                # Assign -1 means not generate on the heatmap
                keypoints[0] = -np.ones(2) * 666
                keypoints[1] = -np.ones(2) * 666
            keypoints[2] = box3d_pts[5]
            keypoints[3] = box3d_pts[0]
            heatmap[self.num_keypoints + 1] = self.generate_target_edge(box3d_pts[1], box3d_pts[0])

        # Generate the target keypoints heatmap
        target, target_weight = self.generate_target_keypoints(keypoints, keypoints_vis_fake)
        heatmap[0:self.num_keypoints] = target
        heatmap = torch.from_numpy(heatmap)
        target_weight = torch.from_numpy(target_weight)

        return image.transpose(2, 0, 1), heatmap, target_weight

# ----- Generate target edges and keypoints ----- #

    def generate_target_edge(self, start0, stop0):
        target_edge = np.zeros([self.heatmap_size[1], self.heatmap_size[0]], dtype=float)

        normal_distrib = [1, 0.7, 0.4, 0.2, 0.1, 0.05]
        feat_stride = self.input_size / self.heatmap_size

        start = start0.copy() / feat_stride
        stop = stop0.copy() / feat_stride

        for i in range(self.heatmap_size[0]):
            for j in range(self.heatmap_size[1]):
                target_edge[j][i] = 1.2 ** (- dist(start[0], start[1], stop[0], stop[1], i, j) ** 2)

        return target_edge

        # old method
        start = np.floor(start / feat_stride).astype(int)
        stop = np.floor(stop / feat_stride).astype(int)

        # start = np.minimum(start, self.heatmap_size - 1)
        # stop = np.minimum(stop, self.heatmap_size - 1)



        if start[0] < 0 or start[1] < 0 or start[0] >= self.heatmap_size[0] or start[1] >= self.heatmap_size[1]:
            tmp = start
            start = stop
            stop = tmp

        # Using DDA algorithm to generate the line and push back to the queue
        dx = stop[0] - start[0]
        dy = stop[1] - start[1]
        k = dy / (dx+1)

        queue = []

        if abs(k) <= 1:
            if start[0] > stop[0]:
                tmp = start
                start = stop
                stop = tmp

            for cx in range(start[0], stop[0]):
                cy = int(round(start[1] + k * (cx - start[0])))
                if cx >= 0 and cy >= 0 and cx < self.heatmap_size[0] and cy < self.heatmap_size[1]:
                    queue.append((cx, cy, 0))
                    target_edge[cy][cx] = 1
                else:
                    break
        else:
            if start[1] > stop[1]:
                tmp = start
                start = stop
                stop = tmp

            for cy in range(start[1], stop[1]):
                cx = int(round(start[0] + (1 / k) * (cy - start[1])))
                if cx >= 0 and cy >= 0 and cx < self.heatmap_size[0] and cy < self.heatmap_size[1]:
                    queue.append((cx, cy, 0))
                    target_edge[cy][cx] = 1
                else:
                    break

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
