from torch import nn
import torch
import math


def awgn(snr, x, device):
    # snr(db)
    n = 1 / (10 ** (snr / 10))
    sqrt_n = n ** 0.5
    noise = torch.randn_like(x) * sqrt_n.unsqueeze(-1).unsqueeze(-1)
    noise = noise.to(device)
    x_hat = x + noise
    return x_hat


def fading(snr_db, x, device, eps=1e-8):

    x = x.to(device)
    assert x.shape[-1] % 2 == 0, "Last dimension must be even for IQ split"
    C = x.shape[-1] // 2

    x_I = x[..., :C]
    x_Q = x[..., C:]
    x_complex = x_I + 1j * x_Q

    h_real = torch.randn_like(x_I)
    h_imag = torch.randn_like(x_I)
    h = (h_real + 1j * h_imag) / math.sqrt(2)

    snr_linear = 10 ** (snr_db / 10)
    noise_var = 2 / snr_linear
    noise_std = math.sqrt(noise_var / 2)

    n_real = torch.randn_like(x_I) * noise_std
    n_imag = torch.randn_like(x_I) * noise_std
    noise = n_real + 1j * n_imag

    y = h * x_complex + noise

    y_hat = y / (h + eps)

    x_hat = torch.cat([y_hat.real, y_hat.imag], dim=-1)
    return x_hat


def normalize(x, power=1):
    mask = (x != 0)
    non_zero_count = mask.sum()
    power_emp = torch.sum(x[mask] ** 2) / non_zero_count
    factor = (power / power_emp) ** 0.5
    x = x * factor
    return power_emp, x


class ChannelNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(ChannelNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, 1, 1))
        self.bias = nn.Parameter(torch.zeros(hidden_size, 1, 1))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.PReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.prelu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel, stride=1):
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel, stride, int((kernel - 1) / 2)),
            ChannelNorm(channel_out),
            nn.ReLU())

        self.context_extractor = nn.AdaptiveAvgPool2d(1)
        self.factor_predictor_snr = nn.Sequential(
            nn.Linear(channel_out, channel_out),
            nn.ReLU(),
            nn.Linear(channel_out, channel_out),
            nn.Sigmoid()
        )

        self.factor_predictor_l = nn.Sequential(
            nn.Linear(channel_out, channel_out),
            nn.ReLU(),
            nn.Linear(channel_out, channel_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer(x)

        p = self.context_extractor(x).squeeze(-1).squeeze(-1)
        # p_snr = torch.concat([p, snr], dim=1)
        factor_snr = self.factor_predictor_snr(p).unsqueeze(-1).unsqueeze(-1)
        return x * factor_snr


class ConvBlockUpSample(nn.Module):
    def __init__(self, channel_in, channel_out, kernel, stride=1):
        super(ConvBlockUpSample, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(channel_in, channel_out * stride * stride, kernel, 1, int((kernel - 1) / 2)),
            ChannelNorm(channel_out * stride * stride),
            nn.PixelShuffle(stride),
            nn.ReLU())

        self.context_extractor = nn.AdaptiveAvgPool2d(1)
        self.factor_predictor_snr = nn.Sequential(
            nn.Linear(channel_out, channel_out),
            nn.ReLU(),
            nn.Linear(channel_out, channel_out),
            nn.Sigmoid()
        )
        self.factor_predictor_l = nn.Sequential(
            nn.Linear(channel_out, channel_out),
            nn.ReLU(),
            nn.Linear(channel_out, channel_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer(x)
        p = self.context_extractor(x).squeeze(-1).squeeze(-1)
        # p_snr = torch.concat([p, snr], dim=1)
        factor_snr = self.factor_predictor_snr(p).unsqueeze(-1).unsqueeze(-1)
        return x * factor_snr