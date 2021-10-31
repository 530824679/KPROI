# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/10/30 下午3:59
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : general.py
# @Software: PyCharm
# Description : None
# --------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os                   # 与操作系统进行交互的模块
import glob                 # 仅支持部分通配符的文件搜索模块
from pathlib import Path    # Path将str转换为Path对象 使字符串路径易于操作的模块

def colorstr(*input):
    """
    把输出的开头和结尾加上颜色，命令行输出显示会更加好看，如: colorstr('blue', 'hello world')
    """
    # args: 输入的颜色序列 string: 输入的字符串
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    # 定义一些基础的颜色 和 字体设置
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    # 把输出的开头和结尾加上颜色  命令行输出显示会更加好看
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def get_latest_run(search_dir='.'):
    """用于返回该项目中最近的模型 '.pth'对应的路径，用于断点续传
    :params search_dir: 要搜索的文件的根目录 默认是 '.'  表示搜索该项目中的文件
    """
    # glob.glob函数匹配所有的符合条件的文件, 并将其以list的形式返回
    last_list = glob.glob(f'{search_dir}/checkpoints/*.pth', recursive=True)
    # 返回路径列表中创建时间最晚(最近的last文件)的路径
    return max(last_list, key=os.path.getctime) if last_list else ''

def check_file(file):
    """
    检查本地有没有这个文件，相关文件路径能否找到文件 并返回文件名
    """
    # 如果传进来的是文件或者是’‘, 直接返回文件名str
    if os.path.isfile(file) or file == '':
        return file
    # 如果传进来的就是当前项目下的一个全局路径 查找匹配的文件名返回第一个
    else:
        files = glob.glob('./**/' + file, recursive=True)
        # 验证文件名是否存在
        assert len(files), 'File Not Found: %s' % file
        # 验证文件名是否唯一
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)
        # 返回第一个匹配到的文件名
        return files[0]

def increment_dir(dir, comment=''):
    """
    递增路径 如 run/train/exp --> runs/train/exp{sep}0, runs/exp{sep}1 etc.
    :params dir: run/train/exp
    :params comment:
    :params mkdir: 是否在这里创建dir  False
    """
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')