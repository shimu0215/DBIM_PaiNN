import torch
import torch.nn as nn
import torch.nn.functional as F


class DBIMLoss(nn.Module):
    def __init__(self, sigma_data=0.5):
        super(DBIMLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.sigma_data = sigma_data

    def forward(self, model_predict, xt, x0, node_mask, noise, sigma_t=None, weighted=False):

        # F_theta = model_predict - xt
        # x_theta = xt - F_theta
        
        # x_theta = x_theta / sigma ** 2
        # x0 = x0 / sigma ** 2
        # loss = self.mse_loss(x_theta * node_mask, x0 * node_mask)

        # model_predict = model_predict / sigma ** 2
        # x0 = x0 / sigma ** 2

        # loss = self.mse_loss((model_predict-xt) * node_mask, noise * node_mask)

        if weighted:
            weight = (sigma_t ** 2 + self.sigma_data ** 2) / (sigma_t * self.sigma_data) ** 2
            wse = weight * (model_predict * node_mask - x0 * node_mask) ** 2
            loss = wse.mean()
        else:
            loss = self.mse_loss(model_predict * node_mask, x0 * node_mask)

        return loss


class PaiNNLoss(nn.Module):
    def __init__(self):
        super(PaiNNLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, data, weight=True):
        energy = pred['energy']
        energy_grad = pred['forces']
        npa_charges = pred['npa_charges']

        # loss_energy = F.mse_loss(energy, data.energy)
        # loss_force = F.mse_loss(energy_grad, data.energy_grad)
        # loss_npa = F.mse_loss(npa_charges, data.npa_charges)

        loss_energy = F.l1_loss(energy, data.energy)
        loss_force = F.l1_loss(energy_grad, data.energy_grad)
        loss_npa = F.l1_loss(npa_charges, data.npa_charges)

        # lambda_1 = 1.0 / (loss_energy.item() + 1e-6)
        # lambda_2 = 1.0 / (loss_force.item() + 1e-6)
        # lambda_3 = 1.0 / (loss_npa.item() + 1e-6)

        lambda_1 = 0.05
        lambda_2 = 0.75
        lambda_3 = 0.2

        total_loss = lambda_1 * loss_energy + lambda_2 * loss_force + lambda_3 * loss_npa

        # total_loss = torch.log(1 + loss_energy) + torch.log(1 + loss_force) + torch.log(1 + loss_npa)

        return total_loss, loss_energy, loss_force, loss_npa
        