#!/usr/bin/env python
# encoding: utf-8

"""
@Version : 3.6
@Author  : Parzival
@File    : utils.py
@Software: PyCharm
"""

import numpy as np


def sigmoid(x):
    """
    sigmoid函数
    :param x: 输入变量
    """
    # 防止上溢
    if x > 20.0:
        x = 20.0
    # 防止下溢
    if x < -20.0:
        x = -20.0
    return 1 / (1 + np.exp(x))
