#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 10:36
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='train or test')

    # settings for the SBU datset

    parser.add_argument(
        '--dataset',
        type=str,
        default='SBU',
        help='choose the dataset, SBU or K3HI')
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='./datasets',
        help='define the datasets root path')
    parser.add_argument(
        '--dataset_split',
        default= [['01', '09', '15', '19'],
                  ['05', '07', '10', '16'],
                  ['02', '03', '20', '18'],
                  ['04', '06', '08', '11'],
                  ['12', '13', '14', '17'],
                  ['21']],
        help='split the SBU dataset to 5 part, the first 4 parts used for training, and last part(21) for testing')
    parser.add_argument(
        '--train_mean',
        default=None,
        help='data mean value, changed after loading the dataset')
    parser.add_argument(
        '--train_std',
        default=None,
        help='data std value, changed after loading the dataset')
    args = parser.parse_args()
    return args