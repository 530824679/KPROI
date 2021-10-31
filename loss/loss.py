# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/10/30 上午9:49
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : loss.py
# @Software: PyCharm
# Description : None
# --------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class KeyPointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(KeyPointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, weights):
        batch_size = output.size(0)
        num_keypoints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_keypoints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_keypoints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_keypoints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(weights[:, idx]),
                    heatmap_gt.mul(weights[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_keypoints