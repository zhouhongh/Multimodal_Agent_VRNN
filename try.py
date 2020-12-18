#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 11:04
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn

import numpy as np

x = np.array([[2,3],[3,4]])
y = x[:-1,:]
print(y.shape)