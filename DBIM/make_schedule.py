import torch

from DBIM_models import polynomial_schedule

def get_snr(alpha, sigma):
    return alpha ** 2 / sigma ** 2

def get_noise_schedule(T, device=None):
    # betas = torch.linspace(1e-4, 0.02, T, device=device)
    # alphas = torch.cumprod(1 - betas, dim=0).sqrt()
    # sigmas = (1 - alphas ** 2).sqrt()

    alphas2 = torch.tensor(polynomial_schedule(T, power=float(2)), dtype=torch.float32, device=device)
    sigmas2 = 1 - alphas2
    alphas = alphas2.sqrt()
    sigmas = sigmas2.sqrt()

    return alphas, sigmas


def make_noise_schedule(T=1000, eta=0.0, device=None):

    alphas, sigmas = get_noise_schedule(T, device)

    snrs = get_snr(alphas, sigmas)
    snrT_to_t = snrs[-1] / snrs

    ats = alphas / alphas[-1] * snrT_to_t
    bts = alphas * (1 - snrT_to_t)
    cts = sigmas * torch.sqrt(1 - snrT_to_t)

    rhos = eta * sigmas[:-1] * torch.sqrt(1 - snrs[1:] / snrs[:-1])
    rhos = torch.cat([rhos[:-1], cts[-1:]], dim=0)  # enforce rho_{N-1} = c_{t_{N-1}}

    return ats, bts, cts, rhos, sigmas