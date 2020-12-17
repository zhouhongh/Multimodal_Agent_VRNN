#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 15:16
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn
import numpy as np

def read_SBU_txt( txt_path):
    """
    :param txt_path:
    :return: np.array([frame_num, 30 , 3]) with dtype:float32
    """
    with open(txt_path) as f:
        data = f.readlines()
    full_data = []
    for row in data:
        posture = row
        posture_data = [np.float32(x.strip()) for x in posture.split(',')]
        joint_info = []
        for i in range(1, len(posture_data), 3):
            joint_info.append([posture_data[i], posture_data[i+1], posture_data[i+2]])
        full_data.append(np.array(joint_info))
    return np.stack(full_data, axis=0)

def normalization_stats(completeData):
    """
    Copied from https://github.com/una-dinosauria/human-motion-prediction
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    return [data_mean, data_std]

def normalize_SBU_data(data, data_mean, data_std):

    data_out = {}
    data_std[np.abs(data_std) < 1e-8] = 1.0
    for set_key in data.keys():
        one_set_dic = {}
        for cat_key in data[set_key].keys():
            seqs = []
            for seq in data[set_key][cat_key]:
                norm_seq = np.divide((seq - data_mean), data_std)
                seqs.append(norm_seq)
            one_set_dic[cat_key] = seqs
        data_out[set_key] = one_set_dic
    return data_out

if __name__ == '__main__':
    data = read_SBU_txt('./datasets/SBU/01/01/skeleton_pos_001.txt')
    print(data.shape)