import torch
from torch.nn import functional as F
from torch_geometric.data import Data
import numpy as np

from DBIM_PaiNN.utils import sample_time_step, perturb_coordinates, sub_center
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
    rhos = torch.cat([rhos[:], cts[-1:]], dim=0) # enforce rho_{N-1} = c_{t_{N-1}}

    return ats, bts, cts, rhos, sigmas


def data_prepare(data, args):
    device = args.device
    dtype = torch.float32

    atom_type_scaling = args.atom_type_scaling
    max_atom_number = args.max_atom_number

    x = data['pos'].to(device, dtype)
    node_mask = data['atom_mask'].to(device).bool()
    h = data['atomic_numbers_one_hot'].to(device, dtype) * atom_type_scaling

    # reformat
    curr_batch_size = x.size()[0] // max_atom_number

    x = x.view(curr_batch_size, max_atom_number, 3)
    node_mask = node_mask.view(curr_batch_size, max_atom_number)
    h = h.view(curr_batch_size, max_atom_number, -1)

    x = sub_center(x)

    x_noisy = perturb_coordinates(x0=x, noise_std=args.noise_level)
    match_fail = torch.tensor([smiles == '' for smiles in data['smiles']]).to(device)
    rpos = data['rdkit_pos'].to(device, dtype)

    rpos = rpos.view(curr_batch_size, max_atom_number, 3)
    match_fail = match_fail.unsqueeze(1).unsqueeze(2).repeat(1, max_atom_number, 3)

    xT = rpos + match_fail * x_noisy
    xT = sub_center(xT)

    return x, h, node_mask, xT


def forward_transition(data, ats, bts, cts, args, use_pure_noise=False):
    x, h, node_mask, xT = data_prepare(data=data, args=args)
    device = args.device
    T = args.T

    t = sample_time_step(x=x, T=T)
    t_norm = t / T

    x0 = x.clone()
    if use_pure_noise:
        xT = perturb_coordinates(x0=x0, noise_std=args.noise_level)
    xT = sub_center(xT)

    noise = torch.randn_like(xT, device=device)
    noise = sub_center(noise)

    xt = ats[t] * xT + bts[t] * x0 + cts[t] * noise
    xt = sub_center(xt)

    return h, xt, xT, t_norm, node_mask, noise, x0, t


def predict_DBIM(h, xt, xT, t_norm, node_mask, generative_model, args):

    max_atom_number = args.max_atom_number
    curr_batch_size = xt.size()[0]

    h = torch.cat([h, xt], dim=-1)
    h = torch.cat([h, xT], dim=-1)
    h = torch.cat([h, t_norm], dim=-1)

    if len(node_mask.shape) == 3:
        node_mask = node_mask.squeeze(-1)

    _, pos_predict = generative_model(xt=xt, h=h, mask=node_mask)

    pos_predict = sub_center(pos_predict)

    node_mask = node_mask.view(curr_batch_size, max_atom_number, -1)

    return pos_predict, node_mask

def predict_PaiNN(data, node_mask, pretrained_PaiNN, args, pos=None):
    device = args.device

    if pos is None:
        pos_clean = data.pos.to(device)[node_mask.view(-1)].clone()
    else:
        pos_clean = pos[node_mask.squeeze(-1)].clone()

    data_PaiNN = Data(
        pos=pos_clean,
        batch=data.batch.to(device)[node_mask.view(-1)].clone(),
        atomic_numbers=data.atomic_numbers.to(device)[node_mask.view(-1)].clone(),
        energy=data.energy.to(device).clone(),
        energy_grad=data.energy_grad.to(device)[node_mask.view(-1)].clone(),
        npa_charges=data.npa_charges.to(device)[node_mask.view(-1)].clone(),
        natoms=data.natoms.to(device).clone()
    )

    pred = pretrained_PaiNN(data_PaiNN)

    return pred, data_PaiNN

def sampling(data, ats, bts, cts, rhos, generative_model, args):
    sample_steps = args.sample_steps
    t_norm = torch.tensor(np.linspace(0.00001, 1, sample_steps)).to(args.device)
    ts = torch.round(t_norm*args.T).int()

    x, h, node_mask, xT = data_prepare(data=data, args=args)

    x0 = x.clone()
    xT = perturb_coordinates(x0=x0, noise_std=args.noise_level)
    xT = sub_center(xT)
    xt = xT.clone()

    ato, bto, cto = -1, -1, -1

    for t in reversed(ts):
        at = ats[t]
        bt = bts[t]
        ct = cts[t]
        rt = rhos[t]

        noise = torch.randn_like(xT)
        noise = sub_center(noise)

        ht = h.clone()
        tt = t.expand(h.size(0), x.size(1), 1) / args.T

        x0_hat, node_mask = predict_DBIM(ht, xt, xT, tt, node_mask, generative_model, args)

        # sampling method 2 - containing xt-1
        if t == args.T:
            xt = at * xT + bt * x0_hat + rt * noise
        else:
            xt = at * xT + bt * x0_hat + rt * noise + torch.sqrt(ct ** 2 - rt ** 2) * (xt - ato * xT - bto * x0_hat) / cto

        # sampling method 1 - simple
        # xt = at * xT + bt * x0_hat + rt * noise

        xt = sub_center(xt)

        ato = at
        bto = bt
        cto = ct

        # check sampling process
        # loss = F.mse_loss(x0_hat * node_mask, x0 * node_mask)
        # print('t:', t, 'dis:', loss)

    return xt, x, node_mask