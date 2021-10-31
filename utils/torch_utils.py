# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/10/30 下午4:25
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : torch_utils.py
# @Software: PyCharm
# Description : None
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os                               # 与操作系统进行交互的模块
import copy
import torch
import logging                          # 日志功能生成模块
import platform                         # 提供获取操作系统相关信息的模块
logger = logging.getLogger(__name__)

def get_saved_state(model, optimizer, lr_scheduler, epoch, hyp):
    """Get the information to save with checkpoints"""
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    utils_state_dict = {
        'epoch': epoch,
        'hyp': hyp,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scheduler': copy.deepcopy(lr_scheduler.state_dict())
    }

    return model_state_dict, utils_state_dict

def save_checkpoint(checkpoints_dir, saved_fn, model_state_dict, utils_state_dict, epoch):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""
    model_save_path = os.path.join(checkpoints_dir, 'Model_{}_epoch_{}.pth'.format(saved_fn, epoch))
    utils_save_path = os.path.join(checkpoints_dir, 'Utils_{}_epoch_{}.pth'.format(saved_fn, epoch))

    torch.save(model_state_dict, model_save_path)
    torch.save(utils_state_dict, utils_save_path)

    print('save a checkpoint at {}'.format(model_save_path))

def select_device(device='', batch_size=None):
    """
    用于选择模型训练的设备 并输出日志信息
    :params device: 输入的设备  device = 'cpu' or '0' or '0,1,2,3'
    :params batch_size: 一个批次的图片个数
    """
    # s: 之后要加入logger日志的显示信息
    s = f'3DOD 🚀 torch {torch.__version__} '  # string
    # 如果device输入为cpu  cpu=True  device.lower(): 将device字符串全部转为小写字母
    cpu = device.lower() == 'cpu'
    if cpu:
        # 如果cpu=True 就强制(force)使用cpu 令torch.cuda.is_available() = False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        # 如果输入device不为空  device=GPU  直接设置 CUDA environment variable = device 加入CUDA可用设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        # 检查cuda的可用性 如果不可用则终止程序
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    # 输入device为空 自行根据计算机情况选择相应设备  先看GPU 没有就CPU
    # 如果cuda可用 且 输入device != cpu 则 cuda=True 反正cuda=False
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        # n: 所有可用的gpu设备数量  device count
        n = torch.cuda.device_count()
        # 检查是否有gpu设备 且 batch_size是否可以能被显卡数目整除  check batch_size is divisible by device_count
        if n > 1 and batch_size:
            # 如果不能则关闭程序
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        # 定义等长的空格
        space = ' ' * len(s)

        # 满足所有条件 s加上所有显卡的信息
        for i, d in enumerate(device.split(',') if device else range(n)):
            # p: 每个可用显卡的相关属性
            p = torch.cuda.get_device_properties(i)
            # 显示信息s加上每张显卡的属性信息
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        # cuda不可用显示信息s就加上CPU
        s += 'CPU\n'

    # 将显示信息s加入logger日志文件中
    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    # 如果cuda可用就返回第一张显卡的的名称 如: GeForce RTX 2060 反之返回CPU对应的名称
    return torch.device('cuda:0' if cuda else 'cpu')