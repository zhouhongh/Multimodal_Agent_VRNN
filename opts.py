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
        '--drop_p',
        type=float,
        default=0.2,
        help='dropout probability')
    parser.add_argument(
        '--max_iter',
        type=int,
        default=25000,
        help='training iterations')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate')
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
        '--input_ratio',
        type=float,
        default=0.5,
        help='the input frame nums / the total frame nums')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100)
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:2')

    # model settings
    parser.add_argument(
        '--x_dim',
        type=list,
        default=[45, 45, 5])
    parser.add_argument(
        '--h_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--z_dim',
        type=int,
        default=10)
    parser.add_argument(
        '--region_split',
        type=list,
        default=[[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8],
                 [9, 10, 11],
                 [12, 13, 14]]
    )

    # datset settings
    parser.add_argument('--job', type=int, default=4, help='subprocesses to use for data loading')
    parser.add_argument(
        '--dataset_split',
        default= [['01', '09', '15', '19'],
                  ['05', '07', '10', '16'],
                  ['02', '03', '20', '18'],
                  ['04', '06', '08', '11'],
                  ['12', '13', '14', '17', '21']],
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