# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/10/30 下午12:54
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : train.py
# @Software: PyCharm
# Description : None
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(train_dataloader, model, optimizer, lr_scheduler, epoch, hyp, device, logger, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('Accuracy', ':.4e')

    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses, accuracy], prefix="Train - Epoch: [{}/{}]".format(epoch, hyp['end_epoch']))

    criterion = KeyPointsMSELoss(hyp['use_target_weight'])
    num_iters_per_epoch = len(train_dataloader)

    # switch to train mode
    model.train()
    start_time = time.time()
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        data_time.update(time.time() - start_time)
        inputs, targets, target_weight = batch_data
        global_step = num_iters_per_epoch * epoch + batch_idx + 1

        batch_size = inputs.size(0)
        targets = targets.to(device, non_blocking=True)
        target_weight = target_weight.to(device, non_blocking=True)
        #for k in targets.keys():
        #    targets[k] = targets[k].to(device, non_blocking=True)
        #    target_weight[k] = target_weight[k].cuda(non_blocking=True)
        inputs = inputs.to(device, non_blocking=True).float()
        outputs = model(inputs)

        # compute loss
        total_loss = criterion(outputs, targets, target_weight)

        # compute gradient and perform backpropagation
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure accuracy and record loss
        reduced_loss = total_loss.data
        losses.update(to_python_float(reduced_loss), batch_size)

        _, avg_acc, cnt, pred = calc_acc(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
        accuracy.update(avg_acc, cnt)

        # measure elapsed time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)

        #if tb_writer is not None:
        #    if (global_step % hyp['ckpt_freq']) == 0:
        #        tb_writer.add_scalars('Train', total_loss, global_step)

        if batch_idx % hyp['print_freq'] == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, batch_idx, num_iters_per_epoch, batch_time=batch_time,
                      speed=inputs.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=accuracy)
            print(msg)

        # Log message
        if logger is not None:
            if (global_step % hyp['ckpt_freq']) == 0:
                logger.info(progress.get_message(batch_idx))

        start_time = time.time()

def main(hyp, device, tb_writer=None):
    # create model
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    model = build_model(hyp['pretrained'], hyp['num_keypoints'], is_train=False)
    model = model.to(device)

    num_parameters = parameters_num(model)
    logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if hyp['resume']:
        checkpoints = hyp['resume'] if isinstance(hyp['resume'], str) else get_latest_run()
        assert os.path.isfile(checkpoints), 'ERROR: --resume checkpoint does not exist'
        model.load_state_dict(torch.load(checkpoints))
        if logger is not None:
            logger.info('Resuming training from %s' % checkpoints)

    # create optimizer
    optimizer = create_optimizer(hyp, model)
    lr_scheduler = create_lr_scheduler(hyp, optimizer)

    # Create dataloader
    logger.info(">>> Loading dataset & getting dataloader...")
    train_dataloader, train_sampler = create_train_dataloader(hyp)
    if logger is not None:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    for epoch in range(hyp['start_epoch'], hyp['end_epoch']):
        lr_scheduler.step()
        if logger is not None:
            logger.info('>>> Epoch: [{}/{}]'.format(epoch, hyp['end_epoch']))

        # train for one epoch
        train(train_dataloader, model, optimizer, lr_scheduler, epoch, hyp, device, logger, tb_writer)
        if (epoch % hyp['ckpt_freq'] == 0):
            val_dataloader = create_val_dataloader(hyp)
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
            val_loss = validate(val_dataloader, model, device, hyp)
            print('val_loss: {:.4e}'.format(val_loss))
            if tb_writer is not None:
                tb_writer.add_scalar('Val_loss', val_loss, epoch)

        # Save checkpoint
        if ((epoch % hyp['ckpt_freq']) == 0):
            model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, hyp)
            save_checkpoint(hyp['ckpt_dir'], '3DOD', model_state_dict, utils_state_dict, epoch)

        lr_scheduler.step()
        if tb_writer is not None:
           tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)

    if tb_writer is not None:
        tb_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--config', type=str, default='./config/hyp.yaml', help='the path of the pretrained weights')
    parser.add_argument('--device', type=str, default='cpu', help='number of select device')
    opt = parser.parse_args()

    hyp = check_file(opt.config)
    assert len(hyp), '--hyp file must be specified'

    # 载入初始超参
    with open(hyp, encoding='UTF-8') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    device = select_device(opt.device, batch_size=hyp['batch_size'])

    # Logger
    logger.info(opt)
    prefix = colorstr('tensorboard: ')
    logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % hyp['logs_dir'])

    # Tensorboard
    tb_writer = None
    hyp['logs_dir'] = increment_dir(Path(hyp['logs_dir']) / 'exp')
    tb_writer = SummaryWriter(log_dir=hyp['logs_dir'])

    # 调用train()函数，开始训练
    main(hyp, device, tb_writer)