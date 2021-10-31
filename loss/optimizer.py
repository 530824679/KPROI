# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/10/30 下午12:30
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : optimizer.py
# @Software: PyCharm
# Description : None
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from utils.visualize import plot_lr_scheduler

def create_optimizer(hyp, model):
    optimizer = None
    if hyp['optimizer_type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyp['lr'],
            momentum=hyp['momentum'],
            weight_decay=hyp['weight_decay'],
            nesterov=False)
    elif hyp['optimizer_type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyp['lr'])
    else:
        assert False, "Unknown optimizer type"

    return optimizer

def create_lr_scheduler(hyp, optimizer):
    """Create learning rate scheduler for training process"""
    if hyp['lr_type'] == 'multi_step':
        def multi_step_scheduler(i):
            if i < hyp['lr_step'][0]:
                factor = 1.
            elif i < hyp['lr_step'][1]:
                factor = 0.1
            else:
                factor = 0.01

            return factor

        lr_scheduler = LambdaLR(optimizer, multi_step_scheduler)

    elif hyp['lr_type'] == 'cosin':
        lf = lambda x: (((1 + math.cos(x * math.pi / hyp['num_epochs'])) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
    else:
        raise ValueError

    plot_lr_scheduler(optimizer, lr_scheduler, hyp['end_epoch'], save_dir=hyp['logs_dir'], lr_type=hyp['lr_type'])

    return lr_scheduler