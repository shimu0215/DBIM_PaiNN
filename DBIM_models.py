import torch
from egnn_pytorch import EGNN_Network
import torch.nn as nn
import math
from torch.nn import functional as F
import numpy as np

class DBIMGenerativeModel(nn.Module):
    def __init__(self, number_tokens = None, dim = 17, num_layers=11, in_features=16, m_dim=256, T=1000):
        super(DBIMGenerativeModel, self).__init__()
        self.egnn = EGNN_Network(
                                    # num_tokens = number_tokens,
                                    num_positions=29,
                                    # unless what you are passing in is an unordered set, set this to the maximum sequence length
                                    dim=dim,
                                    m_dim=m_dim,
                                    depth=num_layers,
                                    num_nearest_neighbors=8,
                                    update_feats=True,
                                    # norm_coors = True,
                                    # norm_feats=True,
                                    # coor_weights_clamp_value = 2.
        )

        self.T = T

    def forward(self, xt, h, mask=None):

        predicted_pos = self.egnn(h, xt, mask=mask)
        return predicted_pos

class DiffusionEGNN(nn.Module):
    def __init__(self, number_tokens = None, dim = 11, num_layers=9, in_features=16, m_dim=256, T=1000):
        super(DiffusionEGNN, self).__init__()
        self.egnn = EGNN_Network(
                                # num_tokens = number_tokens,
                                 num_positions = 29,           # unless what you are passing in is an unordered set, set this to the maximum sequence length
                                 dim = dim,
                                 m_dim = m_dim,
                                 depth = num_layers,
                                 num_nearest_neighbors = 8,
                                 update_feats = False,
                                 # norm_coors = True,
                                 # coor_weights_clamp_value = 2.
                                 )
        # self.T = T
        # self.number_tokens = number_tokens

    def forward(self, h, x, mask=None):

        # print(self.number_tokens)
        predicted_noise = self.egnn(h, x, mask=mask)
        return predicted_noise

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)

class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma

def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2

def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        # if noise_schedule == 'cosine':
        #     alphas2 = cosine_beta_schedule(timesteps)
        # elif 'polynomial' in noise_schedule:
        # splits = noise_schedule.split('_')
        splits = 2
        # assert len(splits) == 2
        power = float(2)
        alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        # else:
        #     raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            # torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            torch.tensor(-log_alphas2_to_sigmas2, dtype=torch.float32),
        requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]