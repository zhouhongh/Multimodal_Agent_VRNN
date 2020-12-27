#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 17:01
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn
import dataset

import opts
from torch.utils.data import DataLoader
import torch
from model import VRNN
from loss_function import elbo_loss as Loss
import math
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import numpy as np

def main(opt):

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    train_set = dataset.SBU_Dataset(opt, training=True)
    test_set = dataset.SBU_Dataset(opt, training=False)
    lens = train_set.__len__()
    iters_per_epoch = math.ceil(lens / opt.batch_size)
    max_epoch = math.ceil(opt.max_iter / iters_per_epoch)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                             num_workers=opt.job, pin_memory=True,
                             collate_fn=dataset.collate_fn, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.job, pin_memory=True,
                              collate_fn=dataset.collate_fn, drop_last=False)

    writer = SummaryWriter()
    print("loading the model.......")
    net = VRNN(opt)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    best_loss = 10000
    bar = tqdm(range(max_epoch))
    for epoch in bar:
        bar.set_description('train epoch %06d' % epoch)
        train(train_loader, net, device, optimizer, writer, epoch)
        test_loss = test(test_loader, net, device, writer, epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            save_model(net, optimizer,epoch)

    writer.close()

def train(data_loader, net, device, optimizer, writer, epoch):
    net.train()
    epoch_loss = 0
    for n_iter, [batch_x_pack, batch_y_pack] in enumerate(data_loader, 0):
        # batch_x: [batch_size, longest seq len, data_dim] with 0 padding
        # batch_x_pack(PackedSequence): batch_sizes(len = the longest seq), data(shape = [all steps, 90])
        batch_x_pack = batch_x_pack.float().to(device)
        batch_y_pack = batch_y_pack.float().to(device)
        output = net(batch_x_pack, batch_y_pack)
        loss = Loss(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.cpu().detach().numpy()
    # show_grad(net)
    writer.add_scalars('scalar/train', {'loss': epoch_loss / (n_iter + 1)}, epoch)
    tqdm.write("VRNN training loss(epoch %d): %.03f" % (epoch, epoch_loss / (n_iter + 1)))

def test(data_loader, net, device, writer, epoch):
    net.eval()
    with torch.no_grad():
        epoch_loss = 0
        for n_iter, [batch_x_pack, batch_y_pack] in enumerate(data_loader, 0):
            batch_x_pack = batch_x_pack.float().to(device)
            batch_y_pack = batch_y_pack.float().to(device)
            output = net(batch_x_pack, batch_y_pack)
            loss = Loss(output)
            epoch_loss += loss.cpu().detach().numpy()

        writer.add_scalars('scalar/test', {'loss': epoch_loss / (n_iter + 1)}, epoch)
        tqdm.write("VRNN testing loss(epoch %d): %.03f" % (epoch, epoch_loss / (n_iter + 1)))
    return epoch_loss / (n_iter + 1)

def save_model(net, optimizer, epoch):
    folder = './models'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    if epoch > 50:
        state = {
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(folder, './bestmodel'))
        tqdm.write('Model saved at epoch %d !!' % epoch)

def show_grad(net):
    for name, parms in net.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
if __name__ == "__main__":
    opt = opts.parse_opt()
    main(opt)