#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 17:12
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn
import numpy as np
import matplotlib
import os
# matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import imageio, os
import torch
import dataset
import opts
from torch.utils.data import DataLoader
from model import VRNN
from tqdm import tqdm
import utils
def save_png(txt_path, png_floder):

    with open(txt_path) as f:
        data = f.readlines()
    frame = 0
    for row in data:
        posture = row
        frame += 1
        posture_data = [x.strip() for x in posture.split(',')]

        joint_info = {}
        for i, n in enumerate(range(0, len(posture_data), 3)):
            joint_info[i + 1] = [1280 - float(posture_data[n]) * 2560,
                                 960 - (float(posture_data[n + 1]) / 1.2) * 1920, posture_data[n + 2]]
            # joint_info[i+1] = [-float(posture_data[n]), -float(posture_data[n+1]), posture_data[n+2]]
        # print("Number of people in scene:\t", len(joint_info)/15, end='\n\n')

        person_1 = {k: joint_info[k] for k in range(1, 16, 1)}
        # print(person_1)
        person_2 = {k - 15: joint_info[k] for k in range(16, 31, 1)}
        # print(person_1)
        joint_details = {1: 'HEAD',
                         2: 'NECK',
                         3: 'TORSO',
                         4: 'LEFT_SHOULDER',
                         5: 'LEFT_ELBOW',
                         6: 'LEFT_HAND',
                         7: 'RIGHT_SHOULDER',
                         8: 'RIGHT_ELBOW',
                         9: 'RIGHT_HAND',
                         10: 'LEFT_HIP',
                         11: 'LEFT_KNEE',
                         12: 'LEFT_FOOT',
                         13: 'RIGHT_HIP',
                         14: 'RIGHT_KNEE',
                         15: 'RIGHT_FOOT'}
        connect_map = [[1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 7, 8, 10, 11, 13, 14],
                       [2, 3, 4, 7, 4, 7, 10, 13, 5, 6, 8, 9, 11, 12, 14, 15]]

        for key, value in person_1.items():
            plt.plot(value[0], value[1], 'bo')
            # plt.annotate(joint_details[key], (value[0], value[1]))
        for m, n in zip(connect_map[0], connect_map[1]):
            plt.plot((person_1[m][0], person_1[n][0]), (person_1[m][1], person_1[n][1]), 'b--')

        for key, value in person_2.items():
            plt.plot(value[0], value[1], 'go')
            # plt.annotate(joint_details[key], (value[0], value[1]))
        for m, n in zip(connect_map[0], connect_map[1]):
            plt.plot((person_2[m][0], person_2[n][0]), (person_2[m][1], person_2[n][1]), 'g--')

        plt.title(frame)
        plt.xlim(-1280, 1280)
        plt.ylim(-960, 960)
        # plt.axis('off')
        # plt.pause(0.1)
        plt.savefig(os.path.join(png_floder, 'skeleton%03d.png' % frame))
        plt.clf()

        # plt.show()


def gen_gif(png_floder, gif_floder):
    images = []
    name = png_floder.split('/')[-1]
    filenames=sorted((fn for fn in os.listdir(png_floder) if fn.endswith('.png')))
    for filename in filenames:
        images.append(imageio.imread(os.path.join(png_floder, filename)))
    imageio.mimsave(os.path.join(gif_floder, name + '.gif'), images, duration=0.1)

def generate(opt, txt_path):

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    test_set = dataset.SBU_Dataset(opt, training=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=opt.job, pin_memory=True,
                             collate_fn=dataset.collate_fn, drop_last=False)
    data_mean = opt.train_mean
    data_std = opt.train_std
    net = VRNN(opt)
    net.to(device)
    net.load_state_dict(torch.load('./models/bestmodel', map_location=device)["state_dict"])
    net.eval()

    with torch.no_grad():
        for n_iter, [batch_x_pack, batch_y_pack] in enumerate(test_loader, 0):
            batch_x_pack = batch_x_pack.float().to(device)
            batch_y_pack = batch_y_pack.float().to(device)
            output = net(batch_x_pack, batch_y_pack)
            _, _, _, _, decoder_all_1, decoder_all_2, _, _, _, _ = output
            # decoder_all_1: list,len = max_step, element = [1, 45]
            seq = []

            for t in range(len(decoder_all_1)):
                joints = np.concatenate((decoder_all_1[t].squeeze(dim=0).cpu().numpy(),
                                         decoder_all_2[t].squeeze(dim=0).cpu().numpy()), axis=0)
                joints = utils.unNormalizeData(joints, data_mean, data_std)
                seq.append(joints)
            np.savetxt(os.path.join(txt_path, '%03d.txt' % (n_iter+1)), np.array(seq), fmt="%.4f", delimiter=',')

def eval(opt):
    txt_floder = './output/txt'
    png_root = './output/png'
    gif_floder = './output/gif'
    if not os.path.exists(txt_floder):
        os.makedirs(txt_floder)
    if not os.path.exists(gif_floder):
        os.makedirs(gif_floder)
    generate(opt, txt_floder)
    for txt in tqdm(os.listdir(txt_floder)):
        png_floder = os.path.join(png_root, txt.split(".")[0])
        if not os.path.exists(png_floder):
            os.makedirs(png_floder)
        save_png(os.path.join(txt_floder, txt), png_floder)
        gen_gif(png_floder, gif_floder)


if __name__ == "__main__":

    opt = opts.parse_opt()
    eval(opt)