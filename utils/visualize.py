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
import copy
import numpy as np
import matplotlib.pyplot as plt

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