# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/10/30 下午12:54
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : valid.py
# @Software: PyCharm
# Description : None
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml                                         # yaml配置模块
import pprint
import logging                                      # 日志模块
import argparse
import torch
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tqdm import tqdm

from models.kpnet import build_model
from dataset.dataloader import create_val_dataloader
from loss.loss import KeyPointsMSELoss
from utils.general import check_file
from utils.torch_utils import select_device
from utils.misc import AverageMeter
from utils.general import to_python_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate(val_dataloader, model, device, hyp):
    # define loss function
    losses = AverageMeter('Loss', ':.4e')
    criterion = KeyPointsMSELoss(hyp['use_target_weight']).cuda()

    # switch to train mode
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_dataloader)):
            images, targets, target_weight = batch_data

            batch_size = images.size(0)
            #for k in targets.keys():
                #targets[k] = targets[k].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True).float()
            outputs = model(images)
            total_loss = criterion(outputs, targets, 0)

            reduced_loss = total_loss.data
            losses.update(to_python_float(reduced_loss), batch_size)

    return losses.avg

def main(hyp, device):
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # load model arch
    torch.cuda.set_device(device)
    model = build_model(None, hyp['num_keypoints'], is_train=False)
    model = model.cuda(device)

    # load ckpt file
    logger.info('=> loading model from {}'.format(opt['val_model_ckpt']))
    model.load_state_dict(torch.load(opt['val_model_ckpt']))

    # load valid dataset
    val_dataloader = create_val_dataloader(hyp)
    print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))

    # evaluate
    val_loss = validate(val_dataloader, model, device, hyp)
    print('val_loss: {:.4e}'.format(val_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validate keypoints network')
    parser.add_argument('--config', type=str, default='./config/hyp.yaml', help='the path of the pretrained weights')
    parser.add_argument('--device', type=str, default='0', help='number of select device')
    opt = parser.parse_args()

    hyp = check_file(opt.config)
    assert len(hyp), '--hyp file must be specified'

    # 载入初始超参
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    device = select_device(opt.device, batch_size=hyp['batch_size'])

    # 调用validate()函数，开始验证
    main(hyp, device)