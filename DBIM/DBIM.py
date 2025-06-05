import math
from datetime import datetime
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from DBIM_argument import parse_opt_DBIM
from DBIM_read_data import read_dataset
from DBIM_utils import sample_time_step, perturb_coordinates, sub_center, plot_result, read_list, load_model
from DBIM_models import DBIMGenerativeModel, DBIMLoss
from make_schedule import make_noise_schedule

from painn_model import PaiNN
from painn_loss_function import energy_force_npa_Loss


def sampling(data, ats, bts, rhos, generative_model, args):
    sample_steps = args.sample_steps
    t_norm = torch.tensor(np.linspace(0.00001, 1, sample_steps)).to(args.device)
    ts = torch.round(t_norm*args.T).int()

    x, h, node_mask, xT = data_prepare(data=data, args=args)

    x0 = x.clone()
    xT = perturb_coordinates(x0=x0)
    xT = sub_center(xT)
    xt = xT.clone()

    for t in reversed(ts):  
        at = ats[t]
        bt = bts[t]
        # ct = cts[t]
        rt = rhos[t]

        noise = torch.randn_like(xT)
        noise = sub_center(noise)

        ht = h.clone()
        tt = t.expand(h.size(0), x.size(1), 1) / args.T

        x0_hat, node_mask = predict(ht, xt, xT, tt, node_mask, generative_model, args)
        xt = at * xT + bt * x0_hat + rt * noise
        # xt = at * xT + bt * x0_hat
        # xt = at * xT + bt * x0_hat + rt * noise + torch.sqrt(ct ** 2 - rt ** 2) * 
        xt = sub_center(xt)

        loss = F.mse_loss(x0_hat * node_mask, x0 * node_mask)
        print('t:', t, 'dis:', loss)
        

    return xt, x, node_mask

def predict(h, xt, xT, t_norm, node_mask, generative_model, args):

    atom_type_scaling = args.atom_type_scaling
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

def evaluate(val_loader, generative_model, ats, bts, cts, criterion, pretrained_PaiNN, criterion_PaiNN, args):
    with torch.no_grad():
        generative_model.eval()
        evaluating_loss = 0.0
        for batch_idx, data in enumerate(val_loader):

            h, xt, xT, t_norm, node_mask, noise, x0, t = train_data_prepare(data=data, ats=ats, bts=bts, cts=cts, args=args)
            pos_predict, node_mask = predict(h, xt, xT, t_norm, node_mask, generative_model, args)

            # pos_predict, node_mask, noise, t, x0, xt, xT = (
            #     predict(data=data, generative_model=generative_model, ats=ats, bts=bts, cts=cts, args=args))
            # PaiNN_evaluating_loss = predict_paiNN(data, node_mask, pos_predict, pretrained_PaiNN, criterion_PaiNN, args)[0]
            DBIM_evaluating_loss = criterion(model_predict=pos_predict, xt=xt, x0=x0, node_mask=node_mask, noise=noise)
            # loss = DBIM_evaluating_loss + PaiNN_evaluating_loss
            loss = DBIM_evaluating_loss
            evaluating_loss += loss

        evaluating_loss = evaluating_loss / len(val_loader)
        return evaluating_loss

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

    x_noise = perturb_coordinates(x0=x)
    match_fail = torch.tensor([smiles == '' for smiles in data['smiles']]).to(device)
    rpos = data['rdkit_pos'].to(device, dtype)

    rpos = rpos.view(curr_batch_size, max_atom_number, 3)
    match_fail = match_fail.unsqueeze(1).unsqueeze(2).repeat(1, max_atom_number, 3)

    xT = rpos + match_fail * x_noise
    xT = sub_center(xT)

    return x, h, node_mask, xT

def train_data_prepare(data, ats, bts, cts, args):

    x, h, node_mask, xT = data_prepare(data=data, args=args)
    device = args.device
    dtype = torch.float32

    atom_type_scaling = args.atom_type_scaling
    max_atom_number = args.max_atom_number
    curr_batch_size = x.size()[0] // max_atom_number

    T = args.T
    t = sample_time_step(x=x, T=T)
    t_norm = t / T

    x0 = x.clone()
    xT = perturb_coordinates(x0=x0)
    xT = sub_center(xT)

    noise = torch.randn_like(xT, device=device)
    noise = sub_center(noise)

    # xta = ats[t] * xT + bts[t] * x0
    # xta = sub_center(xta)
    xt = ats[t] * xT + bts[t] * x0 + cts[t] * noise
    xt = sub_center(xt)

    return h, xt, xT, t_norm, node_mask, noise, x0, t

def predict_paiNN(data, node_mask, pos_predict, pretrained_PaiNN, criterion_PaiNN, args):
    device = args.device
    dtype = torch.float32

    pos_clean = pos_predict[node_mask.squeeze(-1)].clone()
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
    loss, loss_energy, loss_force, loss_npa = criterion_PaiNN(pred=pred, data=data_PaiNN)

    return loss, loss_energy, loss_force, loss_npa

def train(args):

    device = args.device
    dtype = torch.float32

    # train_loader, val_loader, test_loader = read_dataloader(args)
    train_loader, val_loader, test_loader = read_list(args.data_path, args)

    epochs = args.epochs
    generative_model = DBIMGenerativeModel().to(device)
    # generative_model = load_model(model_path='saved_model/DBIM.pth', device=device, dtype=dtype)

    pretrained_PaiNN = PaiNN(use_pbc=False).to(device)
    # pretrained_PaiNN.load_state_dict(torch.load('saved_model/PaiNN-0525-3')['model_state_dict'])

    optimizer = torch.optim.AdamW(generative_model.parameters(), lr=args.lr, amsgrad=False, weight_decay=1e-12)
    criterion = DBIMLoss()
    criterion_PaiNN = energy_force_npa_Loss()

    T = args.T
    ats, bts, cts, rhos, sigmas = make_noise_schedule(T=T, eta=0.0, device=device)

    atom_type_scaling = args.atom_type_scaling
    max_atom_number = args.max_atom_number

    batch_result_list = []
    epoch_result_list = []

    best_val_loss = float('inf')
    patience = args.patience
    counter = 0

    writer = SummaryWriter(log_dir=f"../DBIM_log/exp_on_whole_1_x0_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    batch_count = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            # continue
            generative_model.train()
            optimizer.zero_grad()

            h, xt, xT, t_norm, node_mask, noise, x0, t = train_data_prepare(data=data, ats=ats, bts=bts, cts=cts, args=args)
            pos_predict, node_mask = predict(h, xt, xT, t_norm, node_mask, generative_model, args)

            sigma_t = sigmas[t]
            loss = criterion(model_predict=pos_predict, xt=xt, x0=x0, node_mask=node_mask, noise=noise, sigma_t=sigma_t, weighted=True)

            # loss, loss_energy, loss_force, loss_npa
            # loss_PaiNN = predict_paiNN(data, node_mask, pos_predict, pretrained_PaiNN, criterion_PaiNN, args)

            # loss = loss_DBIM + loss_PaiNN[0]
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            # torch.nn.utils.clip_grad_norm_(generative_model.parameters(), max_norm=1.0)
            optimizer.step()

            if loss < 2 or len(batch_result_list) == 0:
                epoch_loss += loss

                writer.add_scalar("Loss/batch_loss", loss, batch_count+1)
                # writer.add_scalar("Loss/batch_loss_DBIM", loss_DBIM, batch_count+1)
                # writer.add_scalar("Loss/batch_loss_PaiNN", loss_PaiNN[0], batch_count+1)
                # writer.add_scalar("Loss/batch_loss_energy", loss_PaiNN[1], batch_count+1)
                # writer.add_scalar("Loss/batch_loss_force", loss_PaiNN[2], batch_count+1)
                # writer.add_scalar("Loss/batch_loss_npa", loss_PaiNN[3], batch_count+1)
            else:
                epoch_loss += batch_result_list[-1]

                writer.add_scalar("Loss/batch_loss", loss, batch_count+1)
                # writer.add_scalar("Loss/batch_loss_DBIM", loss_DBIM, batch_count+1)
                # writer.add_scalar("Loss/batch_loss_PaiNN", loss_PaiNN[0], batch_count+1)
                # writer.add_scalar("Loss/batch_loss_energy", loss_PaiNN[1], batch_count+1)
                # writer.add_scalar("Loss/batch_loss_force", loss_PaiNN[2], batch_count+1)
                # writer.add_scalar("Loss/batch_loss_npa", loss_PaiNN[3], batch_count+1)

            # epoch_loss += loss
            # batch_result_list.append(loss)
            # writer.add_scalar("Loss/batch_loss", loss, batch_count + batch_idx)

            batch_count += 1

            if batch_idx % 100 == 0:
                # print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] DBIM Train Loss: {loss_DBIM.item():.10f} PaiNN Eval Loss: {loss_PaiNN[0].item():.10f} Total Loss: {loss.item():.10f}")
                print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] DBIM Train Loss: {loss.item():.10f}")

        with torch.no_grad():
            epoch_loss = epoch_loss / len(train_loader)
            if math.isnan(epoch_loss):
                break

            writer.add_scalar("Loss/epoch_train_loss", epoch_loss, epoch)

            val_loss = evaluate(val_loader=val_loader, generative_model=generative_model,
                         ats=ats, bts=bts, cts=cts, criterion=criterion, pretrained_PaiNN=pretrained_PaiNN, criterion_PaiNN=criterion_PaiNN, args=args)

            print(f"Epoch [{epoch + 1}/{epochs}] Val Loss: {val_loss:.10f}")
            writer.add_scalar("Loss/epoch_val_loss", val_loss, epoch)

            if epoch % 1 == 0:
                val_sample_loss = 0.0
                val_predict_loss = torch.zeros(4)
                for batch_idx, data in enumerate(val_loader):
                    x_predict, x0_val, node_mask = sampling(data=data, ats=ats, bts=bts, rhos=rhos, generative_model=generative_model, args=args)
                    loss = F.mse_loss(x_predict * node_mask, x0_val * node_mask)
                    # print(loss)

                    # if loss.item() <= 1:
                    #     PaiNN_loss = predict_paiNN(data, node_mask, x_predict, pretrained_PaiNN, criterion_PaiNN, args)
                        # val_predict_loss += PaiNN_loss

                        # val_sample_loss += loss
                        # print(PaiNN_loss)

                print(f"Epoch [{epoch + 1}/{epochs}] Val Sample Loss: {val_sample_loss/len(val_loader):.10f}")
                # print(f"Epoch [{epoch + 1}/{epochs}] Val Sample Loss: {val_sample_loss/len(val_loader):.10f}")
                # writer.add_scalar("Loss/val_sample_loss_pos", val_sample_loss, epoch)
                # writer.add_scalar("Loss/val_sample_loss_energy", val_predict_loss[1], batch_count+1)
                # writer.add_scalar("Loss/val_sample_loss_force", val_predict_loss[2], batch_count+1)
                # writer.add_scalar("Loss/val_sample_loss_npa", val_predict_loss[3], batch_count+1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(generative_model.state_dict(), 'saved_model/DBIM_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

    writer.close()


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    args = parse_opt_DBIM()
    train(args)
    # sampling(data=None, ats=None, bts=None, rhos=None, generative_model=None, args=args)
