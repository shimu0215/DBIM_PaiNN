# import net
import torch
import torch_cluster
# import torch_scatter
# import torch_sparse
# import torch_spline_conv

# from fairchem.core.models.painn import PaiNN
from painn import PaiNN
import torch.nn.functional as F
from torch_geometric.data import Data

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

from preprocess import read_dataset
from loss_function import energy_force_npa_Loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_list, val_list, test_list = read_dataset("data/processed")

train_loader = DataLoader(train_list, batch_size=128, shuffle=True)
val_loader = DataLoader(val_list, batch_size=128)
test_loader = DataLoader(test_list)

model = PaiNN(use_pbc = False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_npa_energy_force = energy_force_npa_Loss()

def train(data):
    model.train()
    optimizer.zero_grad()
    pred = model(data)

    loss, loss_energy, loss_force, loss_npa = loss_npa_energy_force(pred=pred, data=data)

    loss /= len(data)
    loss_energy /= len(data)
    loss_force /= len(data)
    loss_npa /= len(data)

    loss.backward()
    optimizer.step()

    return loss, loss_energy, loss_force, loss_npa

best_val_loss = np.inf
start_patience = patience = 5

for epoch in range(100):

    # for batch in train_loader:
    num_batches = len(train_loader)
    epoch_loss = 0.0
    epoch_loss_energy = 0.0
    epoch_loss_force = 0.0
    epoch_loss_npa = 0.0

    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        loss, loss_energy, loss_force, loss_npa = train(batch)

        epoch_loss += loss.item()
        epoch_loss_energy += loss_energy.item()
        epoch_loss_force += loss_force.item()
        epoch_loss_npa += loss_npa.item()

        if i % 10 == 0:
            print(f"Epoch {epoch} batch {i}/{num_batches} : "
                  f"total Loss = {loss:.4f} "
                  f"energy Loss = {loss_energy:.4f} "
                  f"forces Loss = {loss_force:.3e} "
                  f"npa_charges Loss = {loss_npa:.4f}")

        # if i == 0:
        #     break
    # break

    print(f"Epoch {epoch} "
          f"total_loss = {epoch_loss/num_batches:.4f}, "
          f"energy_loss = {epoch_loss_energy/num_batches:.4f}, "
          f"forces_loss = {epoch_loss_force/num_batches:.3e}, "
          f"npa_charges_loss = {epoch_loss_npa/num_batches:.4f} ")

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:  # Validation dataset
            batch = batch.to(device)
            pred = model(batch)
            loss, loss_energy, loss_force, loss_npa = loss_npa_energy_force(pred=pred, data=batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch}: val_Loss = {val_loss:.4f}")

    # break

    if best_val_loss >= val_loss:
        patience = start_patience
        best_val_loss = val_loss
    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break

with torch.no_grad():
    model.eval()
    for batch in test_loader:
        test_data = batch.to(device)
        pred = model(test_data)
        loss, loss_energy, loss_force, loss_npa = loss_npa_energy_force(pred=pred, data=test_data)

    print(f"total Loss = {loss:.4f} energy Loss = {loss_energy:.4f} forces Loss = {loss_force:.3e} npa_charges Loss = {loss_npa:.4f}")


path = 'saved_model/'
torch.save({'model_state_dict': model.state_dict()}, path + 'PaiNN_0525')
