#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 17:01
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn
import dataset
import opts
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from torch import nn
import torch
from model import VRNN


def main(opt):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    train_set = dataset.SBU_Dataset(opt)
    data_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                             collate_fn=dataset.collate_fn)
    for i, (batch_x, batch_y, batch_len) in enumerate(data_loader, 0):
        batch_x_pack = rnn_utils.pack_padded_sequence(batch_x,
                                                      batch_len, batch_first=True)
        batch_y_pack = rnn_utils.pack_padded_sequence(batch_y,
                                                      batch_len, batch_first=True)
        net = nn.LSTM(90, 256, 2, batch_first=True)
        h0 = torch.rand(2, 3, 256)
        c0 = torch.rand(2, 3, 256)

        out, (h1, c1) = net(batch_x_pack, (h0, c0))
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)

        print('END')

if __name__ == "__main__":
    opt = opts.parse_opt()
    main(opt)