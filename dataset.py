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

class FormatData(object):
    """
    Form train/validation data.
    形成字典，分别包含编码器输入，解码器输入，解码器输出（GT）
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, sample, train):

        total_frames = self.config.input_window_size + self.config.output_window_size

        video_frames = sample.shape[0]
        idx = np.random.randint(1, video_frames - total_frames) #在可选范围中随机挑选帧起始点

        data_seq = sample[idx:idx + total_frames, :]
        encoder_inputs = data_seq[:self.config.input_window_size - 1, :]
        # 最后一个弃掉了,这里代码还可以精简
        if train:
            decoder_inputs = data_seq[self.config.input_window_size - 1:
                                      self.config.input_window_size - 1 + self.config.output_window_size, :]
        else:
            decoder_inputs = data_seq[self.config.input_window_size - 1:self.config.input_window_size, :]
        decoder_outputs = data_seq[self.config.input_window_size:, :]
        return {'encoder_inputs': encoder_inputs, 'decoder_inputs': decoder_inputs, 'decoder_outputs': decoder_outputs}


class SBU_Dataset(Dataset):

    def __init__(self, opt):
        self.dataset_path = os.path.join(opt.dataset_root, opt.dataset)
        train_data = None
        self.seq_num = 0
        if opt.dataset == 'SBU':
            # x and y are normalized as [0, 1] while z is normalized as[0, 7.8125]
            if opt.mode == 'train':
                self.train_data = {}
                complete_train = None
                # load all data first
                print("************start loading SBU dataset!!!*************")
                for i in tqdm(range(1,21)):
                    set_data = {}
                    for cat in range(1,9):
                        cat_data = []
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
                            cat_data.append(one_seq_data)
                        set_data['%02d' % cat] = cat_data
                    train_data['%02d' % i] = set_data
                print(complete_train.shape) #[6456,30,3]
                print("************complete loading SBU dataset!!!*************")
                # calculate the mean and std
                data_mean, data_std = utils.normalization_stats(complete_train)
                opt.train_mean = data_mean
                opt.train_std = data_std
                self.train_data = utils.normalize_SBU_data(train_data, data_mean, data_std)

    def __len__(self):

        return self.seq_num

    def __getitem__(self, idx):
        pass












if __name__ == '__main__':
    opt = opts.parse_opt()
    dataset = SBU_Dataset(opt)










