#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 10:12
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn

import torch
import torch.nn as nn



class VRNN(nn.Module):

    def __init__(self, opt):
        super(VRNN, self).__init__()
        self.region = opt.region_split
        self.dropout_1 = nn.Dropout(opt.drop_p)
        self.dropout_2 = nn.Dropout(opt.drop_p)
        self.dropout_3 = nn.Dropout(opt.drop_p)
        # modal 1
        self.h1_dim, self.x_fea_1,\
        self.prior_fea_1, self.prior_mean_1, self.prior_var_1,\
        self.decoder_fea_1, self.decoder_mean_1,\
        self.encoder_fea_1, self.encoder_mean_1, self.encoder_var_1,\
        self.rnn_1 = self.define_paras(opt, modal=1)

        # modal 2
        self.h2_dim, self.x_fea_2, \
        self.prior_fea_2, self.prior_mean_2, self.prior_var_2, \
        self.decoder_fea_2, self.decoder_mean_2, \
        self.encoder_fea_2, self.encoder_mean_2, self.encoder_var_2,\
        self.rnn_2 = self.define_paras(opt, modal=2)

        # modal 3
        self.h3_dim, self.x_fea_3,\
        self.prior_fea_3, self.prior_mean_3, self.prior_var_3, \
        self.decoder_fea_3, _,\
        self.encoder_fea_3, self.encoder_mean_3, self.encoder_var_3,\
        self.rnn_3 = self.define_paras(opt, modal=3)

        # z
        z_dim = opt.z_dim
        h_dim = opt.h_dim
        self.z_fea = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
    def define_paras(self, opt, modal=1):
        x_dim = opt.x_dim[modal-1]
        h_dim = opt.h_dim
        z_dim = opt.z_dim
        # feature extractors of x and z
        # paper: We found that these feature extractors are crucial for learnting complex sequences
        # paper: 'all of phi_t have four hidden layers using rectificed linear units ReLu'
        x_fea = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # prior: input h output mu, sigma
        prior_fea = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        prior_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        prior_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        # decoder: input phi(z), h
        decoder_fea = None
        decoder_mean = None
        if modal == 3:
            decoder_fea = nn.Sequential(
                nn.Linear(h_dim, x_dim),
                nn.ReLU()
            )
        else:
            decoder_fea = nn.Sequential(
                nn.Linear(h_dim*2, h_dim),
                nn.ReLU()
            )
            decoder_mean = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, x_dim),
                nn.Sigmoid()
            )

        # encoder: input: phi(x), h
        encoder_fea = nn.Sequential(
            nn.Linear(h_dim*2, h_dim),
            nn.ReLU()
        )
        # VRE regard mean values sampled from z as the output
        encoder_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        encoder_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        # using the recurrence equation to update its hidden state
        rnn = nn.GRUCell(h_dim*2, h_dim)
        return (h_dim, x_fea,
                prior_fea, prior_mean, prior_var,
                decoder_fea, decoder_mean,
                encoder_fea, encoder_mean, encoder_var,
                rnn)

    def product_of_experts(self, means, logvar):
        P = 1.0 / torch.exp(logvar)
        Psum = P.sum(dim=0)
        prod_means = torch.sum(means * P, dim=0) / Psum
        prod_logvar = torch.log(1.0 / Psum)
        return prod_means, prod_logvar

    def modal_encoder(self,x,h,modal):
        if modal == 1:
            x_fea = self.x_fea_1
            prior_fea = self.prior_fea_1
            prior_mean = self.prior_mean_1
            prior_var = self.prior_var_1
            encoder_fea = self.encoder_fea_1
            encoder_mean = self.encoder_mean_1
            encoder_var = self.encoder_var_1
        elif modal == 2:
            x_fea = self.x_fea_2
            prior_fea = self.prior_fea_2
            prior_mean = self.prior_mean_2
            prior_var = self.prior_var_2
            encoder_fea = self.encoder_fea_2
            encoder_mean = self.encoder_mean_2
            encoder_var = self.encoder_var_2
        else:
            x_fea = self.x_fea_3
            prior_fea = self.prior_fea_3
            prior_mean = self.prior_mean_3
            prior_var = self.prior_var_3
            encoder_fea = self.encoder_fea_3
            encoder_mean = self.encoder_mean_3
            encoder_var = self.encoder_var_3

        # feature extractor:
        phi_x_ = x_fea(x)
        # prior
        prior_fea_ = prior_fea(h)
        prior_means_ = prior_mean(prior_fea_)
        prior_var_ = prior_var(prior_fea_)
        # encoder
        encoder_fea_ = encoder_fea(torch.cat([phi_x_, h], dim=1))
        encoder_means_ = encoder_mean(encoder_fea_)
        encoder_var_ = encoder_var(encoder_fea_)
        return phi_x_, prior_means_, prior_var_, encoder_means_, encoder_var_

    def cal_x3(self, x1_hat, x1):
        st = torch.abs(x1_hat - x1)
        mean_st = torch.mean(st, dim=1)  # [batch_size(t)]
        st_1 = torch.mean(st[:, self.region[0]]) > mean_st
        st_2 = torch.mean(st[:, self.region[1]]) > mean_st
        st_3 = torch.mean(st[:, self.region[2]]) > mean_st
        st_4 = torch.mean(st[:, self.region[3]]) > mean_st
        st_5 = torch.mean(st[:, self.region[4]]) > mean_st
        xt_3 = torch.stack((st_1, st_2, st_3, st_4, st_5), dim=1).float()
        return xt_3

    def forward(self, source, target):
        """

        :param
        :return:
        """
        x, batch_sizes, _, _ = source
        y, _, _, _ = target
        max_step = len(batch_sizes)
        batch_id = torch.cumsum(batch_sizes, dim=0)
        prefix = torch.tensor([0])
        batch_id = torch.cat((prefix, batch_id), dim=0)
        # init h for modal 1 2 3
        ht_1 = torch.randn([batch_sizes[0], self.h1_dim], device=x.device)
        ht_2 = torch.randn([batch_sizes[0], self.h2_dim], device=x.device)
        ht_3 = torch.randn([batch_sizes[0], self.h3_dim], device=x.device)
        # global record paras for loss
        prior_means_all = []
        prior_var_all = []
        poster_means_all = []
        poster_var_all = []
        decoder_all_1 = []
        decoder_all_2 = []
        decoder_all_3 = []
        target_x3 = []
        target_x2 = []
        target_x1 = []
        xt_3 = torch.zeros((batch_sizes[0], 5), device=x.device)
        # step forward
        for t in range(max_step):
            xt_1 = x[batch_id[t]:batch_id[t+1], :45]
            xt_2 = x[batch_id[t]:batch_id[t+1], 45:]
            xt_3 = xt_3[:batch_sizes[t], :]
            ht_1 = ht_1[:batch_sizes[t]]
            ht_2 = ht_2[:batch_sizes[t]]
            ht_3 = ht_3[:batch_sizes[t]]
            yt_1 = y[batch_id[t]:batch_id[t + 1], :45]
            yt_2 = y[batch_id[t]:batch_id[t + 1], 45:]
            """
            modal 1 : prior and poster distribution
            """
            phi_x_1, prior_means_1, prior_var_1, encoder_means_1, encoder_var_1 = self.modal_encoder(xt_1, ht_1, modal=1)
            """
            modal 2 : prior and poster distribution
            """
            phi_x_2, prior_means_2, prior_var_2, encoder_means_2, encoder_var_2 = self.modal_encoder(xt_2, ht_2, modal=2)
            """
            modal 3 : prior and poster distribution
            """
            phi_x_3, prior_means_3, prior_var_3, encoder_means_3, encoder_var_3 = self.modal_encoder(xt_3, ht_3, modal=3)
            """
            POE
            """
            # prior
            prior_means = torch.cat(
                (prior_means_1.unsqueeze(0), prior_means_2.unsqueeze(0), prior_means_3.unsqueeze(0)))
            prior_logvar = torch.cat(
                (prior_var_1.unsqueeze(0), prior_var_2.unsqueeze(0), prior_var_3.unsqueeze(0)))
            prior_means, prior_logvar = self.product_of_experts(prior_means, prior_logvar)
            # poster(encoder)
            poster_means = torch.cat((encoder_means_1.unsqueeze(0), encoder_means_2.unsqueeze(0), encoder_means_3.unsqueeze(0)))
            poster_logvar = torch.cat((encoder_var_1.unsqueeze(0), encoder_var_2.unsqueeze(0), encoder_var_3.unsqueeze(0)))
            poster_means, poster_logvar = self.product_of_experts(poster_means, poster_logvar)

            # decoder
            z_sampled = self.reparametrizing(poster_means, poster_logvar)
            phi_z = self.z_fea(z_sampled)

            """
            modal 1 generation and recurrence
            """

            decoder_fea_1 = self.decoder_fea_1(torch.cat([phi_z, ht_1], dim=1))
            decoder_means_1 = self.decoder_mean_1(decoder_fea_1)
            decoder_means_1 = self.dropout_1(decoder_means_1)
            # rnn
            ht_1 = self.rnn_1(torch.cat([phi_x_1, phi_z], dim=1), ht_1)


            """
            modal 2 generation and recurrence
            """
            decoder_fea_2 = self.decoder_fea_2(torch.cat([phi_z, ht_2], dim=1))
            decoder_means_2 = self.decoder_mean_2(decoder_fea_2)
            decoder_means_2 = self.dropout_2(decoder_means_2)
            # rnn
            ht_2 = self.rnn_2(torch.cat([phi_x_2, phi_z], dim=1), ht_2)

            """
            modal 3 generation and recurrence
            """
            decoder_fea_3 = self.decoder_fea_3(ht_3)
            decoder_3 = torch.sigmoid(decoder_fea_3)
            decoder_3 = self.dropout_3(decoder_3)
            # rnn
            ht_3 = self.rnn_3(torch.cat([phi_x_3, phi_z], dim=1), ht_3)

            """
            calculate the ground truth (saliency map) of modal 3
            """
            xt_3 = self.cal_x3(decoder_means_1, yt_1)
            """
            record the distributions at each time
            """
            prior_means_all.append(prior_means)
            prior_var_all.append(prior_logvar)
            poster_means_all.append(poster_means)
            poster_var_all.append(poster_logvar)
            """
            record the output x123 at each time
            """
            decoder_all_1.append(decoder_means_1)
            decoder_all_2.append(decoder_means_2)
            decoder_all_3.append(decoder_3)
            """
            record the ground truth
            """
            target_x3.append(xt_3)
            target_x2.append(yt_2)
            target_x1.append(yt_1)
        return [
            prior_means_all, prior_var_all, poster_means_all, poster_var_all,
            decoder_all_1, decoder_all_2, decoder_all_3,
            target_x1, target_x2, target_x3
        ]

    def reparametrizing(self, *args):
        z_mean, z_log_var = args
        epsilon = torch.rand_like(z_mean, device=z_mean.device)
        return z_mean + z_log_var * epsilon

