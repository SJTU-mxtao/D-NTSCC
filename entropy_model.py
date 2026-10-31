from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
import torch.optim
import matplotlib.pyplot as plt
import math

# The `lower_bound_fn' is 
# from the repo: https://github.com/jorge-pessoa/pytorch-gdn/blob/master/pytorch_gdn/__init__.py,
# which is available under the MIT license:

'''MIT License

Copyright (c) 2019 Jorge Pessoa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

class lower_bound_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


lower_bound = lower_bound_fn.apply

# This "entropy bottleneck" implementation clearly corresponds to the one mentioned in the paper below. 
# `Appendix' describes each step of the entropy estimation part.

"""
> "Variational image compression with a scale hyperprior"<br />
  > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
  > https://arxiv.org/abs/1802.01436
"""

# Their TensorFlow-based implementation can be found at:
# https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/entropy_models.py

class ChannelwiseTwoLayerLinear(nn.Module):
    def __init__(self, channel, input_dim):
        super(ChannelwiseTwoLayerLinear, self).__init__()
        reduced_dim = input_dim // 2  # l/2

        self.weight1 = nn.Parameter(torch.randn(channel, input_dim, reduced_dim))  # (channel, l, l/2)
        self.bias1 = nn.Parameter(torch.randn(channel, reduced_dim))  # (channel, l/2)

        self.weight2 = nn.Parameter(torch.randn(channel, reduced_dim, 1))  # (channel, l/2, 1)
        self.bias2 = nn.Parameter(torch.randn(channel, 1))  # (channel, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(0, 1)
        x = torch.bmm(x, self.weight1) + self.bias1.unsqueeze(1)  # (batchsize, channel, l/2)
        x = self.relu(x)

        x = torch.bmm(x, self.weight2) + self.bias2.unsqueeze(1)  # (batchsize, channel, 1)

        x = x.transpose(0, 1)
        return x


class EntropyBottleneck(nn.Module):
    def __init__(self, channels, number_of_mixture=2, initial_scale=2, tail_mass=1e-9,
                 optimize_integer_offset=True, likelihood_bound=1e-9, quantize=True,
                 range_coder_precision=16):

        super(EntropyBottleneck, self).__init__()
        self.tail_mass = float(tail_mass)
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1, got {}.".format(self.tail_mass))
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.likelihood_bound = float(likelihood_bound)
        self.range_coder_precision = int(range_coder_precision)

        rho_data = torch.randn((channels, 1)) * initial_scale
        rho = nn.Parameter(data=rho_data, requires_grad=True)
        self.register_parameter('rho', rho)

        w_data = torch.randn((channels, number_of_mixture))
        self.weight = nn.Parameter(data=w_data, requires_grad=True)

        self.mean = ChannelwiseTwoLayerLinear(channels, 8)
        self.var = ChannelwiseTwoLayerLinear(channels, 8)

        target = np.log(2 / self.tail_mass - 1)
        self.target = torch.tensor([-target, 0, target])
        self.quantize = quantize

    def multivariate_gaussian_pdf(self, x, mean, cov):
        dim = mean.size(-1)
        cov_det = torch.det(cov)
        cov_inv = torch.inverse(cov)

        norm_const = 1.0 / (torch.sqrt((2 * torch.pi) ** dim * cov_det) + 1e-8)

        x_mu = x - mean.unsqueeze(1)
        exponent = -0.5 * torch.sum((x_mu @ cov_inv) * x_mu, dim=-1)

        pdf_vals = norm_const.unsqueeze(-1) * torch.exp(exponent)
        return pdf_vals

    def integrate_region(self, upper_right, upper_left, lower_right, lower_left, mean, cov, num_points=50):
        M, N, _ = upper_right.shape

        x_min = torch.min(torch.stack([upper_right[:, :, 0], upper_left[:, :, 0], lower_right[:, :, 0], lower_left[:, :, 0]], dim=2), dim=2)[0]
        x_max = torch.max(torch.stack([upper_right[:, :, 0], upper_left[:, :, 0], lower_right[:, :, 0], lower_left[:, :, 0]], dim=2), dim=2)[0]
        y_min = torch.min(torch.stack([upper_right[:, :, 1], upper_left[:, :, 1], lower_right[:, :, 1], lower_left[:, :, 1]], dim=2), dim=2)[0]
        y_max = torch.max(torch.stack([upper_right[:, :, 1], upper_left[:, :, 1], lower_right[:, :, 1], lower_left[:, :, 1]], dim=2), dim=2)[0]

        x_vals = torch.linspace(0, 1, num_points, device=upper_right.device)
        y_vals = torch.linspace(0, 1, num_points, device=upper_right.device)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # (num_points**2, 2)

        grid_points = grid_points[None, None, :, :] * (torch.stack([x_max - x_min, y_max - y_min], dim=-1)[:, :, None, :]) + \
                      torch.stack([x_min, y_min], dim=-1)[:, :, None, :]  # (M, N, num_points**2, 2)

        grid_points = grid_points.view(M, N * num_points ** 2, 2)

        pdf_vals = self.multivariate_gaussian_pdf(grid_points, mean, cov)  # (M, N * num_points**2)

        area = (x_max - x_min) * (y_max - y_min)  # 形状为 (M, N)

        pdf_vals = pdf_vals.view(M, N, -1)  # (M, N, num_points**2)
        integral_values = area * pdf_vals.mean(dim=2)  # (M, N)
        return integral_values

    def logits_cumulative(self, upper_right, upper_left, lower_right, lower_left, z1, z2):
        mean_1 = self.mean(z1)
        sigma_1 = self.var(z1)
        var_1 = sigma_1 ** 2

        mean_2 = self.mean(z2)
        sigma_2 = self.var(z2)
        var_2 = sigma_2 ** 2

        mean = torch.concat([mean_1, mean_2], dim=-1)
        cov_matrices = torch.zeros(z1.shape[0], z1.shape[1], 2, 2, device=sigma_2.device)

        cov_matrices[:, :, 0, 0] = var_1.squeeze(-1)
        cov_matrices[:, :, 1, 1] = var_2.squeeze(-1)
        cov_matrices[:, :, 0, 1] = self.__getattr__('rho').squeeze(-1)
        cov_matrices[:, :, 1, 0] = self.__getattr__('rho{}').squeeze(-1)

        integrals = self.integrate_region(upper_right, upper_left, lower_right, lower_left, mean, cov_matrices, num_points=50)

        return integrals

    def standardized_cumulative(self, inputs):
        half = torch.tensor(0.5, dtype=inputs.dtype, device=inputs.device)
        const = torch.tensor(-(2 ** -0.5), dtype=inputs.dtype, device=inputs.device)
        return half * torch.erfc(const * inputs)

    def conditional_gaussian_predict(self, known_values, mean, cov, known_index=0):
        known_values = known_values.squeeze(1)

        if known_index == 0:
            # know X, predict Y
            mu_X, mu_Y = mean[:, 0:1], mean[:, 1:2]  # (128, 1)
            sigma_X2, sigma_Y2 = cov[:, 0, 0:1], cov[:, 1, 1:2]  # (128, 1)
            sigma_XY = cov[:, 0, 1:2]  # (128, 1)

            conditional_mean = mu_Y + sigma_XY / sigma_X2 * (known_values - mu_X)
        else:
            # know Y, predict X
            mu_Y, mu_X = mean[:, 1:2], mean[:, 0:1]
            sigma_Y2, sigma_X2 = cov[:, 1, 1:2], cov[:, 0, 0:1]
            sigma_YX = cov[:, 1, 0:1]

            conditional_mean = mu_X + sigma_YX / sigma_Y2 * (known_values - mu_Y)

        return conditional_mean

    def forward(self, x1, x2):
        mean_1 = self.mean(x1.reshape(x1.shape[0], x1.shape[1], -1))
        sigma_1 = self.var(x1.reshape(x1.shape[0], x1.shape[1], -1))
        var_1 = sigma_1 ** 2
        var_1 = var_1.clamp(0.1, 20)

        mean_2 = self.mean(x2.reshape(x1.shape[0], x1.shape[1], -1))
        sigma_2 = self.var(x2.reshape(x1.shape[0], x1.shape[1], -1))
        var_2 = sigma_2 ** 2
        var_2 = var_2.clamp(0.1, 20)

        mean = torch.concat([mean_1, mean_2], dim=-1)
        cov_matrices = torch.zeros(x1.shape[0], x1.shape[1], 2, 2, device=sigma_2.device)

        cov_matrices[:, :, 0, 0] = var_1.squeeze(-1)
        cov_matrices[:, :, 1, 1] = var_2.squeeze(-1)

        rho = torch.tanh(self.__getattr__('rho').squeeze(-1)).repeat(x1.shape[0], 1)
        cov = rho * var_1.squeeze(-1) ** 0.5 * var_2.squeeze(-1) ** 0.5
        cov_matrices[:, :, 0, 1] = cov
        cov_matrices[:, :, 1, 0] = cov

        shape = x1.shape
        x1 = torch.reshape(x1, (shape[0] * shape[1], 1, -1)).float()

        shape = x2.shape
        x2 = torch.reshape(x2, (shape[0] * shape[1], 1, -1)).float()

        x = torch.concat([x1, x2], dim=1)
        x = x.permute(0, 2, 1)

        if self.quantize:
            if self.training:
                x = x + torch.rand_like(x) - 0.5  # during training, adding noise for differentiability.
            else:
                x = torch.round(x)

        offset_upper_right = torch.tensor([0.5, 0.5], device=x.device).view(1, 1, 2)
        offset_upper_left = torch.tensor([-0.5, 0.5], device=x.device).view(1, 1, 2)
        offset_lower_right = torch.tensor([0.5, -0.5], device=x.device).view(1, 1, 2)
        offset_lower_left = torch.tensor([-0.5, -0.5], device=x.device).view(1, 1, 2)

        likelihood = self.integrate_region(x + offset_upper_right, x + offset_upper_left, x + offset_lower_right, x + offset_lower_left, mean.reshape(-1, 2), cov_matrices.reshape(-1, 2, 2),
                                          num_points=50)

        if self.likelihood_bound > 0:
            likelihood = lower_bound(likelihood, self.likelihood_bound)

        x = x.permute(0, 2, 1)
        x1 = x[:, :1, :]
        x2 = x[:, 1:, :]

        x2_predicted = self.conditional_gaussian_predict(x1, mean.reshape(-1, 2), cov_matrices.reshape(-1, 2, 2), 0)
        x1_predicted = self.conditional_gaussian_predict(x2, mean.reshape(-1, 2), cov_matrices.reshape(-1, 2, 2), 1)

        x1 = torch.reshape(x1, shape)
        x2 = torch.reshape(x2, shape)

        x1_predicted = torch.reshape(x1_predicted, shape)
        x2_predicted = torch.reshape(x2_predicted, shape)

        likelihood = torch.reshape(likelihood, shape)

        return x1, x2, likelihood, x1_predicted, x2_predicted

