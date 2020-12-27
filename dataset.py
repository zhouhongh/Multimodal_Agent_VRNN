#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 10:27
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn


from torch.utils.data import Dataset
import numpy as np
import os
import utils
from tqdm import tqdm
import opts
import torch.nn.utils.rnn as rnn_utils
import torch

def collate_fn(data):
    # data: list [[batch_i, frames 4 each batch_i, 90],]
    # data_x and data_y has the same length
    data.sort(key=lambda x: len(x), reverse=True)
    data_x = []
    data_y = []
    for i in range(len(data)):
        data_x.append(data[i][:-1, :])
        data_y.append(data[i][1:, :])
    data_length = [len(sq) for sq in data_x]
    # pad the sequence length to the last one with 0, for the using of mini-batch
    x = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0)
    batch_x_pack = rnn_utils.pack_padded_sequence(x, data_length, batch_first=True)
    y = rnn_utils.pad_sequence(data_y, batch_first=True, padding_value=0)
    batch_y_pack = rnn_utils.pack_padded_sequence(y, data_length, batch_first=True)

    return [batch_x_pack, batch_y_pack]

class SBU_Dataset(Dataset):

    def __init__(self, opt, training=True):
        self.dataset_path = os.path.join(opt.dataset_root, opt.dataset)
        train_data = []
        self.seq_num = 0
        self.set_start_nums = []
        if opt.dataset == 'SBU':
            # x and y are normalized as [0, 1] while z is normalized as[0, 7.8125]
            if training:
                sets = range(1, 19) # train
                print("************start loading SBU training set!!!*************")
            else:
                sets = range(19, 22) # test
                print("************start loading SBU testing set!!!*************")
            self.train_data = {}
            complete_train = None

            for i in tqdm(sets):# all (1,22)
                self.set_start_nums.append(self.seq_num)
                for cat in range(1, 9):
                    tmp_path = os.path.join(self.dataset_path, '%02d' % i, '%02d' % cat)
                    if not os.path.exists(tmp_path):
                        continue
                    for txt_file in os.listdir(tmp_path):
                        self.seq_num += 1
                        txt_path = os.path.join(tmp_path, txt_file)
                        one_seq_data = utils.read_SBU_txt(txt_path) # np.array float32
                        if complete_train is None:
                            complete_train = one_seq_data
                        else:
                            complete_train = np.concatenate((complete_train, one_seq_data), axis=0)
                        train_data.append(one_seq_data)
            print(complete_train.shape)
            # calculate the mean and std
            data_mean, data_std = utils.normalization_stats(complete_train)
            opt.train_mean = data_mean
            opt.train_std = data_std
            self.train_data = utils.normalize_SBU_data(train_data, data_mean, data_std)

    def __len__(self):

        return self.seq_num

    def __getitem__(self, idx):

        return torch.Tensor(self.train_data[idx])

    def get_set_split(self):

        return self.set_start_nums














if __name__ == '__main__':
    opt = opts.parse_opt()
    dataset = SBU_Dataset(opt)













