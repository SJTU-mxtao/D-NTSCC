from torch import nn
import torch
from torch.nn.functional import gumbel_softmax
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import torchvision
import random
from modules import awgn, ChannelNorm, ConvBlock, ConvBlockUpSample, fading
from modules import normalize as normalize
# from network_bl import STN
import math
import entropy_model
from conditional_entropy_model_jscc import ConditionalEntropyBottleneck, lower_bound
from WITT_modules import *
from Swin_Encoder import SwinEncoder
from Swin_Decoder import SwinDecoder
from Swin_Encoder import BasicLayer
from stn_net import SpatialTransform



class HyperAnalysisTransform(nn.Module):
    def __init__(self, num_filters=192):
        super(HyperAnalysisTransform, self).__init__()
        self.conv_h1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.relu_h1 = nn.ReLU()
        self.conv_h2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.relu_h2 = nn.ReLU()
        self.conv_h3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2, bias=False)

    def forward(self, x):
        x = self.conv_h1(x)
        x = self.relu_h1(x)
        x = self.conv_h2(x)
        x = self.relu_h2(x)
        x = self.conv_h3(x)

        return x


class HyperSynthesisTransform(nn.Module):
    def __init__(self, num_filters=192, num_filters_out=192):
        super(HyperSynthesisTransform, self).__init__()
        self.conv_h4 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.relu_h4 = nn.ReLU()
        self.conv_h5 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.relu_h5 = nn.ReLU()
        self.conv_h6 = nn.ConvTranspose2d(num_filters, num_filters_out * 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_h4(x)
        x = self.relu_h4(x)
        x = self.conv_h5(x)
        x = self.relu_h5(x)
        x = self.conv_h6(x)

        miu = x[:, :int(x.shape[1]/2), :, :]
        sigma = x[:, int(x.shape[1] / 2):, :, :]

        return miu, sigma


class STN(nn.Module):
    def __init__(self, c_in):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=[1, 2]),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 1 * 2, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, f_map, x):
        xs = self.localization(f_map)
        xs = xs.view(-1, 64 * 1 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, align_corners=True)
        return x, theta


class MultiLayerSelector(nn.Module):
    def __init__(self, device):
        super(MultiLayerSelector, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(264, 8),
            nn.Linear(264, 16),
            nn.Linear(264, 24),
            nn.Linear(264, 32),
            nn.Linear(264, 40),
            nn.Linear(264, 48),
            nn.Linear(264, 56),
            nn.Linear(264, 64),
            nn.Linear(264, 72),
            nn.Linear(264, 80),
            nn.Linear(264, 88),
            nn.Linear(264, 96),
            nn.Linear(264, 104),
            nn.Linear(264, 112),
            nn.Linear(264, 120),
            nn.Linear(264, 128),
            nn.Linear(264, 136),
            nn.Linear(264, 144),
            nn.Linear(264, 152),
            nn.Linear(264, 160),
        ])
        self.output_dims = torch.tensor([8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]).to(device)

    def forward(self, x, selected_indices):
        max_output_dim = self.output_dims.max().item()
        output_shape = (x.shape[0], x.shape[1], max_output_dim)
        output = torch.zeros(output_shape, device=x.device)  # 将所有输出初始化为 0

        for i, fc_layer in enumerate(self.fc_layers):
            mask = (selected_indices == i)  # (2, 128)

            selected_vectors = x[mask].view(-1, x.size(-1))  # (N, 40)

            if selected_vectors.size(0) > 0:  # 如果有选择的向量
                layer_output = fc_layer(selected_vectors)  # (N, output_dim)

                output_dim = layer_output.size(-1)
                padded_output = torch.zeros((layer_output.size(0), max_output_dim), device=x.device)
                padded_output[:, :output_dim] = layer_output  # 补齐为最大输出维度

                output[mask, :] = padded_output

        return output


class MultiLayerDe(nn.Module):
    def __init__(self):
        super(MultiLayerDe, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(8, 256),
            nn.Linear(16, 256),
            nn.Linear(24, 256),
            nn.Linear(32, 256),
            nn.Linear(40, 256),
            nn.Linear(48, 256),
            nn.Linear(56, 256),
            nn.Linear(64, 256),
            nn.Linear(72, 256),
            nn.Linear(80, 256),
            nn.Linear(88, 256),
            nn.Linear(96, 256),
            nn.Linear(104, 256),
            nn.Linear(112, 256),
            nn.Linear(120, 256),
            nn.Linear(128, 256),
            nn.Linear(136, 256),
            nn.Linear(144, 256),
            nn.Linear(152, 256),
            nn.Linear(160, 256),
        ])
        self.input_dims = torch.tensor([8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160])

    def forward(self, input_matrix, indices):

        max_output_dim = input_matrix.size(-1)
        output_matrix = torch.zeros((input_matrix.shape[0], input_matrix.shape[1], 256), device=input_matrix.device)

        for i, fc_layer in enumerate(self.fc_layers):
            mask = (indices == i)

            if mask.sum() == 0:
                continue

            selected_vectors = input_matrix[mask][:, :self.input_dims[i]]

            layer_output = fc_layer(selected_vectors)

            output_matrix[mask, :] = layer_output

        return output_matrix


class Trans_De(nn.Module):
    def __init__(self):
        super(Trans_De, self).__init__()

        self.multi_rate_trans = BasicLayer(dim=264,
                                           out_dim=264,
                                           input_resolution=(16, 32),  # (16, 32)
                                           depth=4,
                                           num_heads=8,
                                           window_size=2,
                                           mlp_ratio=4,
                                           qkv_bias=True, qk_scale=None,
                                           norm_layer=nn.LayerNorm,
                                           downsample=None)

    def forward(self, x1, x2):

        x1_output = self.multi_rate_trans(x1)
        x2_output = self.multi_rate_trans(x2)

        return x1_output, x2_output


class FEncoder(nn.Module):
    def __init__(self, config):
        super(FEncoder, self).__init__()

        self.config = config

        self.encoder_kwargs = dict(
                img_size=(128, 256), patch_size=2, in_chans=3,  # (128, 256)
                embed_dims=[128, 160, 192, 256], depths=[2, 2, 6, 2], num_heads=[4, 8, 8, 8], C=256,  # the parameter C does not effect the architecture
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        self.encoder = SwinEncoder(**self.encoder_kwargs)

        self.ha1 = HyperAnalysisTransform(256)
        self.hs1 = HyperSynthesisTransform(256, 256)

        self.ha2 = HyperAnalysisTransform(256)
        self.hs2 = HyperSynthesisTransform(256, 256)

        self.entropy_bottleneck = entropy_model.EntropyBottleneck(256)

        self.conditional_entropy_bottleneck = ConditionalEntropyBottleneck()
        self.bound = 0.05

    def forward(self, x1, x2):
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)

        y1_re = y1.reshape(-1, 8, 16, 256).permute(0, 3, 1, 2)
        y2_re = y2.reshape(-1, 8, 16, 256).permute(0, 3, 1, 2)

        z1 = self.ha1(y1_re)
        z2 = self.ha2(y2_re)

        z1_tilde, z2_tilde, z_likelihoods, z1_predicted, z2_predicted = self.entropy_bottleneck(z1, z2)

        miu_1, sigma_1 = self.hs1(z1_tilde)
        miu_2, sigma_2 = self.hs1(z2_tilde)

        miu_1_predicted, sigma_1_predicted = self.hs1(z1_predicted)
        miu_2_predicted, sigma_2_predicted = self.hs1(z2_predicted)

        sigma_1 = sigma_1.reshape(-1, 256, 128).permute(0, 2, 1)
        sigma_2 = sigma_2.reshape(-1, 256, 128).permute(0, 2, 1)

        miu_1 = miu_1.reshape(-1, 256, 128).permute(0, 2, 1)
        miu_2 = miu_2.reshape(-1, 256, 128).permute(0, 2, 1)

        sigma_1_predicted = sigma_1_predicted.reshape(-1, 256, 128).permute(0, 2, 1)
        sigma_2_predicted = sigma_2_predicted.reshape(-1, 256, 128).permute(0, 2, 1)

        miu_1_predicted = miu_1_predicted.reshape(-1, 256, 128).permute(0, 2, 1)
        miu_2_predicted = miu_2_predicted.reshape(-1, 256, 128).permute(0, 2, 1)

        sigma_1 = lower_bound(sigma_1, self.bound)
        y1_tilde, y1_likelihoods = self.conditional_entropy_bottleneck(y1, sigma_1, miu_1)
        sigma_2 = lower_bound(sigma_2, self.bound)
        y2_tilde, y2_likelihoods = self.conditional_entropy_bottleneck(y2, sigma_2, miu_2)

        sigma_1_predicted = lower_bound(sigma_1_predicted, self.bound)
        _, y1_likelihoods_predicted = self.conditional_entropy_bottleneck(miu_1_predicted, sigma_1_predicted, miu_1_predicted)
        sigma_2_predicted = lower_bound(sigma_2_predicted, self.bound)
        _, y2_likelihoods_predicted = self.conditional_entropy_bottleneck(miu_2_predicted, sigma_2_predicted, miu_2_predicted)

        return y1, y2, y1_likelihoods, y2_likelihoods, y1_likelihoods_predicted, y2_likelihoods_predicted, z_likelihoods, \
               z1_tilde, z2_tilde, z1_predicted, z2_predicted


class FDecoder(nn.Module):
    def __init__(self, config):
        super(FDecoder, self).__init__()
        self.decoder_kwargs = dict(
                img_size=(128, 256),
                embed_dims=[528, 384, 320, 256], depths=[2, 6, 2, 2], num_heads=[8, 8, 8, 4], C=528,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        self.decoder = SwinDecoder(**self.decoder_kwargs)

    def forward(self, x1, x2):
        x1, x2 = self.decoder(x1, x2)
        return x1, x2



class D_NTSCC(nn.Module):
    def __init__(self, device, config):
        super(D_NTSCC, self).__init__()

        self.device = device
        self.config = config

        self.encoder = FEncoder(config)
        self.decoder = FDecoder(config)

        self.multi_rate_trans = BasicLayer(dim=264,
                                           out_dim=264,
                                           input_resolution=(16, 32),
                                           depth=4,
                                           num_heads=8,
                                           window_size=2,
                                           mlp_ratio=4,
                                           qkv_bias=True, qk_scale=None,
                                           norm_layer=nn.LayerNorm,
                                           downsample=None)
        self.multi_rate = MultiLayerSelector(device)

        self.output_dims = torch.tensor([8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]).to(
            device)

        tokens = nn.Parameter(torch.randn(20, 4))

        self.register_parameter('tokens', tokens)

        self.multi_rate_de = MultiLayerDe()

        self.input_dims = torch.tensor([8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160])

        self.trans_de = Trans_De()

        self.spatial_transform = SpatialTransform(264)

    def compute_rate(self, likelihood):
        rate = -torch.log2(likelihood)
        rate_sum = rate.sum(dim=-1)

        rate_diffs = torch.abs(rate_sum.unsqueeze(-1) - self.output_dims)
        selected_indices = rate_diffs.argmin(dim=-1)
        return selected_indices

    def ave_rate(self, indices):
        length_sum = 0
        for i in range(len(self.input_dims)):
            mask = (indices == i)
            length_sum += mask.sum().item() * self.input_dims[i]
        return length_sum / (indices.shape[1])

    def forward(self, x1, x2):
        snr_1 = torch.tensor([self.config.snr_train_1], dtype=torch.float).to(self.device)
        snr_2 = torch.tensor([self.config.snr_train_2], dtype=torch.float).to(self.device)

        y1, y2, y1_likelihoods, y2_likelihoods, y1_likelihoods_predicted, y2_likelihoods_predicted, \
        z_likelihoods, z1_tilde, z2_tilde, z1_predicted, z2_predicted = self.encoder(x1, x2)

        selected_indices_1 = self.compute_rate(y1_likelihoods.detach())
        selected_indices_2 = self.compute_rate(y2_likelihoods.detach())

        selected_indices_1_at_2 = self.compute_rate(y1_likelihoods_predicted.detach())
        selected_indices_2_at_1 = self.compute_rate(y2_likelihoods_predicted.detach())

        token_1 = self.tokens[selected_indices_1]
        token_2 = self.tokens[selected_indices_2]

        token_2_at_1 = self.tokens[selected_indices_2_at_1]
        token_1_at_2 = self.tokens[selected_indices_1_at_2]

        y1_token = torch.concat([y1, token_1, token_2_at_1], dim=-1)
        y2_token = torch.concat([y2, token_2, token_1_at_2], dim=-1)

        y1_t = self.multi_rate_trans(y1_token)
        y2_t = self.multi_rate_trans(y2_token)

        y1_rate = self.multi_rate(y1_t, selected_indices_1)
        y2_rate = self.multi_rate(y2_t, selected_indices_2)

        rate_1 = self.ave_rate(selected_indices_1)
        rate_2 = self.ave_rate(selected_indices_2)

        _, s1 = normalize(y1_rate)
        _, s2 = normalize(y2_rate)

        if self.config.channel_mode == 'awgn':
            s1_hat = awgn(snr_1, s1, self.device)
            s2_hat = awgn(snr_2, s2, self.device)
        elif self.config.channel_mode == 'fading':
            s1_hat = fading(snr_1, s1, self.device)
            s2_hat = fading(snr_2, s2, self.device)

        s1_de = self.multi_rate_de(s1_hat, selected_indices_1)
        s2_de = self.multi_rate_de(s2_hat, selected_indices_2)

        token_1_rx = self.tokens[selected_indices_1]
        token_2_rx = self.tokens[selected_indices_1]

        s1_token = torch.concat([s1_de, token_1_rx, token_2_rx], dim=-1)
        s2_token = torch.concat([s2_de, token_2_rx, token_1_rx], dim=-1)

        s1_output, s2_output = self.trans_de(s1_token, s2_token)

        y2_si, y1_si, theta_1, theta_2 = self.spatial_transform(s1_output.reshape(-1, 8, 16, 264).permute(0, 3, 1, 2), s2_output.reshape(-1, 8, 16, 264).permute(0, 3, 1, 2))

        y2_si = y2_si.reshape(y2_si.shape[0], y2_si.shape[1], -1).permute(0, 2, 1)
        y1_si = y1_si.reshape(y1_si.shape[0], y1_si.shape[1], -1).permute(0, 2, 1)

        s1_output_ = torch.concat([s1_output, y1_si], dim=-1)
        s2_output_ = torch.concat([s2_output, y2_si], dim=-1)

        x1_hat, x2_hat = self.decoder(s1_output_, s2_output_)

        return x1_hat, x2_hat, y1_likelihoods, y2_likelihoods, z_likelihoods, rate_1, rate_2, z1_tilde, z2_tilde, z1_predicted, z2_predicted

