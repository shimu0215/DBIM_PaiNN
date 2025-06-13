import torch
import torch.nn.functional as F
import torch.nn as nn

from loss_functions import DBIMLoss, PaiNNLoss
from airp import make_noise_schedule, forward_transition, predict_DBIM, predict_PaiNN, sampling
from DBIM_argument import parse_opt
from utils import read_list, load_model
from PaiNN.painn_model import PaiNN
from DBIM_models import DBIMGenerativeModel


def evaluate(val_loader, generative_model, args):
    with torch.no_grad():
        generative_model.eval()

        DBIM_criterion = DBIMLoss()

        ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)

        eval_loss = 0.0
        for batch_idx, data in enumerate(val_loader):
            h, xt, xT, t_norm, node_mask, noise, x0, t = forward_transition(data=data, ats=ats, bts=bts, cts=cts,
                                                                            args=args)
            pos_predict, node_mask = predict_DBIM(h, xt, xT, t_norm, node_mask, generative_model, args)
            loss = DBIM_criterion(model_predict=pos_predict,  x0=x0, node_mask=node_mask)

            eval_loss += loss

        eval_loss = eval_loss / len(val_loader)

        return eval_loss

def eval_property_prediction(val_loader, generative_model, pretrained_PaiNN, args, metric):

    ats, bts, cts, rhos, sigmas = make_noise_schedule(T=args.T, eta=args.eta, device=args.device)
    criterion_PaiNN = PaiNNLoss()

    diffusion_bias = torch.zeros(3).to(args.device)
    error = torch.zeros(4).to(args.device)

    for batch_idx, data in enumerate(val_loader):
        x_predict, x0_val, node_mask = sampling(data=data, ats=ats, bts=bts, cts=cts, rhos=rhos,
                                                generative_model=generative_model, args=args)
        pos_dif = F.mse_loss(x_predict * node_mask, x0_val * node_mask)

        if pos_dif.item() <= 1:
            pred_PaiNN, data_PaiNN = predict_PaiNN(data, node_mask, pretrained_PaiNN, args, pos=x_predict)
            pred_PaiNN_ref, data_PaiNN_ref = predict_PaiNN(data, node_mask, pretrained_PaiNN, args)

            if metric == 'MAE':
                loss = nn.L1Loss()
            else:
                loss = nn.MSELoss()

            energy_dis = loss(pred_PaiNN['energy'], pred_PaiNN_ref['energy'])
            force_dis = loss(pred_PaiNN['forces'], pred_PaiNN_ref['forces'])
            npa_dis = loss(pred_PaiNN['npa_charges'], pred_PaiNN_ref['npa_charges'])
            dis = torch.stack([energy_dis, force_dis, npa_dis], dim=0)
            diffusion_bias += dis

            error += criterion_PaiNN(pred=pred_PaiNN, data=data_PaiNN)
        else:
            print('Fail when generate graph.')

        if batch_idx % 50 == 0:
            print(f"Batch [{batch_idx}/{len(val_loader)}] complete")

    diffusion_bias = diffusion_bias / len(val_loader)
    error = error / len(val_loader)

    print('Metric:', metric)
    print('Error:', 'Energy-', error[1], 'force-', error[2], 'npa-', error[3])
    print('Error by Diffusion:', 'Energy-', diffusion_bias[0], 'force-', diffusion_bias[1], 'npa-', diffusion_bias[2])

    return diffusion_bias, error


if __name__ == "__main__":
    args = parse_opt()
    train_loader, val_loader, test_loader = read_list(args.data_path, args)

    device = args.device
    dtype = torch.float32

    generative_model = DBIMGenerativeModel(num_layers=args.num_layers, m_dim=args.m_dim).to(device)
    pretrained_PaiNN = PaiNN(use_pbc=False).to(device)

    if args.load_DBIM:
        generative_model = load_model(model_path=args.DBIM_path, device=device, dtype=dtype)
    if args.load_PaiNN:
        pretrained_PaiNN.load_state_dict(torch.load(args.PaiNN_path)['model_state_dict'])

    eval_property_prediction(val_loader, generative_model, pretrained_PaiNN, args, metric='MAE')