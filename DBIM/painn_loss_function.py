import torch
import torch.nn as nn
import torch.nn.functional as F


class energy_force_npa_Loss(nn.Module):
    def __init__(self):
        super(energy_force_npa_Loss, self).__init__()
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

        total_loss = 0.0

        # if weight:
        lambda_1 = 0.05
        lambda_2 = 0.75
        lambda_3 = 0.2

        total_loss = lambda_1 * loss_energy + lambda_2 * loss_force + lambda_3 * loss_npa

        # total_loss = torch.log(1 + loss_energy) + torch.log(1 + loss_force) + torch.log(1 + loss_npa)

        return total_loss, loss_energy, loss_force, loss_npa