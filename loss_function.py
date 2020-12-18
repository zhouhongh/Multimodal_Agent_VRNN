#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 11:46
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn

import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch


def loss(package, y, steps):

    prior_means, prior_var, decoder_means, decoder_var, x_decoded = package
    loss = 0.
    for i in range(steps):
        # Kl loss
        norm_dis1 = Norm.Normal(prior_means[i], prior_var[i])
        norm_dis2 = Norm.Normal(decoder_means[i], decoder_var[i])
        kl_loss = torch.mean(KL.kl_divergence(norm_dis1, norm_dis2))

        # reconstruction loss
        x_loss = torch.mean(torch.norm(x_decoded-y, p=1))
        loss += x_loss + kl_loss

    return loss