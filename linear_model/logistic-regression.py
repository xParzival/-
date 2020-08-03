#!/usr/bin/env python
# encoding: utf-8

"""
@Version : 3.6
@Author  : Parzival
@File    : logistic-regression.py
@Software: PyCharm
"""

import numpy as np
import utils


class LogisticRegression(object):
    """
    logistic regression实现
    """
    def __init__(self, fit_intercept=True, solver='sgd', if_standard=True, l1_ratio=None, l2_ratio=None, epochs=10,
                 eta=None, batch_size=16):
        """
        :param fit_intercept: 是否添加偏置项
        :param solver: 优化方法，默认随即梯度下降法sgd
        :param if_standard:
        :param l1_ratio: L1正则化因子，默认不正则化
        :param l2_ratio: L2正则化因子，默认不正则化
        :param epochs: 迭代次数，默认10次
        :param eta: 学习率
        :param batch_size: 批量梯度下降每一次随机抽取的样本量
        """