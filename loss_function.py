#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 11:46
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn

import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch

def kl_loss(mean1,var1,mean2,var2):
    norm_dis1 = Norm.Normal(mean1, var1)
    norm_dis2 = Norm.Normal(mean2, var2)
    kl_ls = torch.mean(KL.kl_divergence(norm_dis1, norm_dis2))
    return kl_ls

def recon_loss(x,y):
    return torch.mean(torch.norm(x-y, p=2, dim=1))

def elbo_loss(package):
    prior_means_all, prior_var_all, poster_means_all, poster_var_all,\
    decoder_all_1, decoder_all_2, decoder_all_3, \
    target_x1, target_x2, target_x3 = package
    max_step = len(target_x1)
    loss = 0.
    for i in range(max_step):
        # Kl loss
        kl_ls = kl_loss(prior_means_all[i], prior_var_all[i], poster_means_all[i], poster_var_all[i])
        # reconstruction loss
        x1_loss = recon_loss(decoder_all_1[i], target_x1[i])
        x2_loss = recon_loss(decoder_all_2[i], target_x2[i])
        x3_loss = recon_loss(decoder_all_3[i], target_x3[i])
        loss += x1_loss + x2_loss + x3_loss + kl_ls
        # loss += kl_ls
    loss = loss / max_step
    return loss