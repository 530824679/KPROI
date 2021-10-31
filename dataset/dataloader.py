# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/10/30 下午12:58
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataloader.py
# @Software: PyCharm
# Description : None
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import albumentations as album
from torch.utils.data import DataLoader
from dataset.dataset import KeypointsDataset

def create_train_dataloader(hyp):
    """Create dataloader for training"""
    train_aug_transforms = album.Compose([
        album.RandomBrightnessContrast(p=0.5),
        album.GaussNoise(p=0.5)
    ], p=1.)

    train_sampler = None
    train_dataset = KeypointsDataset(hyp, mode='train', aug_transforms=train_aug_transforms, hflip_prob=hyp['hflip'])
    train_dataloader = DataLoader(train_dataset, batch_size=hyp['batch_size'], shuffle=hyp['is_shuffle'], pin_memory=True, num_workers=hyp['num_workers'], sampler=train_sampler)

    return train_dataloader, train_sampler

def create_val_dataloader(hyp):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = KeypointsDataset(hyp, mode='val', aug_transforms=None, hflip_prob=0.)
    val_dataloader = DataLoader(val_dataset, batch_size=hyp['batch_size'], shuffle=False, pin_memory=True, num_workers=hyp['num_workers'], sampler=val_sampler)

    return val_dataloader


def create_test_dataloader(hyp):
    """Create dataloader for testing phase"""
    test_sampler = None
    test_dataset = KeypointsDataset(hyp, mode='test', aug_transforms=None, hflip_prob=0.)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=hyp['num_workers'], sampler=test_sampler)

    return test_dataloader