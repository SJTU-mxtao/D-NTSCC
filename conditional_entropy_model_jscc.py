from torch import nn
import torch
import torch.optim
from entropy_model import lower_bound_fn

lower_bound = lower_bound_fn.apply


class ConditionalEntropyBottleneck(nn.Module):

    def __init__(self, likelihood_bound=1e-9):
        super(ConditionalEntropyBottleneck, self).__init__()
        self.likelihood_bound = float(likelihood_bound)

    def standardized_cumulative(self, inputs):
        half = torch.tensor(.5, dtype=inputs.dtype)
        const = torch.tensor(-(2 ** -0.5), dtype=inputs.dtype)
        return half * torch.erfc(const * inputs)

    def forward(self, x, sigma, miu):
        x = x + torch.rand_like(x) - 0.5

        upper = self.standardized_cumulative((x + 0.5 - miu) / sigma)
        lower = self.standardized_cumulative((x - 0.5 - miu) / sigma)

        likelihood = upper - lower

        if self.likelihood_bound > 0:
            likelihood = lower_bound(likelihood, self.likelihood_bound)

        return x, likelihood
