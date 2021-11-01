# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/11/1 下午1:03
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : preprocess.py
# @Software: PyCharm
# Description : None
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

def filter_dataset(data_dir):
    """
    交替遍历images和labels两个文件夹，删除未对应的数据
    param data_dir: 数据根目录，包含两个子目录images和labels
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    file_list = os.listdir(images_dir)
    for file in file_list:
        filename = os.path.splitext(file)[0]
        path = os.path.join(labels_dir, filename) + '.json'
        if not os.path.exists(path):
            os.remove(os.path.join(images_dir, file))

def write_txt(data_dir):
    src_ImageSets_dir = os.path.join(data_dir, "ImageSets")
    src_Images_dir = os.path.join(data_dir, "training/images")

    train_list = []
    filelist = os.listdir(src_Images_dir)
    for i in filelist:
        try:
            name = i.split('png')[0]
            # print(name)
            train_list.append(name[:-1])

        except:
            print(i + 'wrong')
            continue

    with open(os.path.join(src_ImageSets_dir, 'train.txt'), 'w') as json_file:
        for text in train_list:
            # print(text)
            json_file.write(text + '\n')


if __name__ == '__main__':
    # filter_dataset("/home/chenwei/HDD/Project/private/KPROI/datas/training")
    write_txt('/home/chenwei/HDD/Project/private/KPROI/datas')