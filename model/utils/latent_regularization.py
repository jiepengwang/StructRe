import numpy as np
import torch
import torch.nn as nn
from termcolor import cprint


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False, mask_valid=None):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.mask_valid = mask_valid
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        if self.deterministic:
            return self.parameters.new_zeros(1)
        if other is None:
            kl_value = torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar
            if self.mask_valid is not None:
                kl_value = kl_value[self.mask_valid]
            return 0.5 * torch.sum(kl_value)
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return self.parameters.new_zeros(1)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class KLReg(nn.Module):
    def __init__(self):
        super().__init__()
        cprint('Use KL regularizer', 'yellow')

    def forward(
        self,
        inp,
        sample_posterior=True,
        mask_valid=None,
    ):
        posterior = DiagonalGaussianDistribution(inp, mask_valid=mask_valid)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return z, posterior.kl(), {}, {}
