import os
import re
import glob
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

from generate_rdkit import *


def nbatch_transform(indices):
    nbatch = torch.tensor([(indices[i + 1] - indices[i]).item() for i in range(len(indices) - 1)])

    return nbatch

def batch_transform(indices):
    batch = torch.tensor([i for i in range(len(indices) - 1) for _ in range(indices[i + 1] - indices[i])])

    return batch

def generate_edge_mask(atom_number, max_atom_number):
    A = torch.ones(atom_number, atom_number)
    B = torch.zeros(max_atom_number, max_atom_number)

    # 将 A 复制到 B 的左上角区域
    B[:atom_number, :atom_number] = A

    return B

def generate_atom_mask(atom_number, max_atom_number):
    A = torch.ones(atom_number, 1)
    B = torch.zeros(max_atom_number, 1)
    B[:atom_number] = A
    return B

def extend_atom_3_feature(pos, max_atom_number):
    n_atoms = pos.shape[0]
    pos_ext = torch.zeros(max_atom_number, 3)
    pos_ext[:n_atoms] = pos

    return pos_ext

def extend_atom_feature(feature, max_atom_number):
    n_atoms = feature.shape[0]
    new_feature = torch.zeros(max_atom_number)
    new_feature[:n_atoms] = feature

    return new_feature

def extend_atomic_numbers_one_hot(atomic_numbers, max_atom_number, max_atom_id):
    n_atoms = atomic_numbers.shape[0]
    atomic_numbers_ext = torch.zeros(max_atom_number, max_atom_id)
    atomic_numbers_ext[:n_atoms] = atomic_numbers

    return atomic_numbers_ext

import numpy as np
from scipy.optimize import linear_sum_assignment

import numpy as np
from scipy.optimize import linear_sum_assignment

def type_constrained_matching(A: np.ndarray, B: np.ndarray,
                              tA: np.ndarray, tB: np.ndarray,
                              p=2):
    if not A.shape == B.shape: return None
    N = len(tA)
    perm_B = np.empty(N, dtype=int)

    for typ in np.unique(tA):
        idxA = np.where(tA == typ)[0]
        idxB = np.where(tB == typ)[0]
        if not len(idxA) == len(idxB): return None

        subA = A[idxA]                  # (k,d)
        subB = B[idxB]                  # (k,d)

        # k×k 距离矩阵
        diff = subA[:, None, :] - subB[None, :, :]
        if p == 1:
            D = np.abs(diff).sum(axis=2)
        elif p == 2:
            D = np.linalg.norm(diff, axis=2)
        else:
            D = (np.abs(diff)**p).sum(axis=2)**(1/p)

        row, col = linear_sum_assignment(D)
        # 写入映射 & 累计成本
        perm_B[idxA[row]] = idxB[col]

    return perm_B

def generate_a_sample_dataset(loaded_tensor, species_name, max_atom_number, max_atom_id, file_name):

    database = loaded_tensor[0]

    n_atoms = len(database.z)
    n_molecules = len(database.y)

    species = -1
    if species_name == "anion":
        species = 0
    elif species_name == "cation":
        species = 1
    elif species_name == "neutral":
        species = 2
    elif species_name == "radical":
        species = 3

    pos = database.pos
    energy_grad = database.energy_grad
    z = database.z
    original_batch = batch_transform(loaded_tensor[1]['z'])
    natoms = nbatch_transform(loaded_tensor[1]['z'])
    energy = database.energy
    npa_charges = database.npa_charges

    data_list = []
    indices = loaded_tensor[1]['z']

    for i in range(n_molecules):
        start, end = indices[i], indices[i + 1]

        # The version that makes all molecules have the same number of atoms
        data = Data(
        pos=extend_atom_3_feature(pos[start:end], max_atom_number), # atom
        atomic_numbers=extend_atom_feature(z[start:end], max_atom_number), # atom
        atomic_numbers_one_hot = extend_atomic_numbers_one_hot(F.one_hot(z[start:end], num_classes=max_atom_id), max_atom_number, max_atom_id), # atom
        atom_mask = generate_atom_mask(z[start:end].shape[0], max_atom_number), # atom
        edge_mask = generate_edge_mask(z[start:end].shape[0], max_atom_number),
        # original_batch=original_batch[start:end], # atom
        batch=extend_atom_feature(original_batch[start:end], max_atom_number),  # atom
        natoms = natoms[i], # molecule
        energy = energy[i], # molecule
        energy_grad = extend_atom_3_feature(energy_grad[start:end], max_atom_number), # atom
        npa_charges = extend_atom_feature(npa_charges[start:end], max_atom_number), # atom
        species = extend_atom_feature(torch.full((end-start,), species), max_atom_number), # atom
        smiles = '',
        rdkit_pos = []
        )

        # smiles, mol = get_smiles(atomic_numbers=atomic_numbers, coords=coords)
        # generate_pos(smiles, mol)

        smiles,mol = get_smiles(data.atomic_numbers[:(end-start)].int().cpu().numpy(), data.pos[:(end-start)].cpu().numpy())
        posr, idr, map = generate_pos(smiles, mol)
        if posr is None:
            data.rdkit_pos = data.pos
            continue
        # posr_new = reorder_by_group(pos[start:end], posr, id)
        perm = type_constrained_matching(pos[start:end],posr, z[start:end], idr)
        if perm is None:
            data.rdkit_pos = data.pos
            continue

        posr_align = torch.tensor(posr[perm])  # ⟵ 复制式重排
        idr_align = np.asarray(idr)[np.asarray(perm, dtype=int)]  # ⟵ 类型索引同步重排
        data.smiles = smiles
        data.rdkit_pos = extend_atom_3_feature(posr_align, max_atom_number)
        # data.

        # print('su')

        # The version that keeps the original number of atoms for each molecule
        # data = Data(
        #     pos=pos[start:end],  # atom
        #     atomic_numbers=z[start:end],  # atom
        #     atomic_numbers_one_hot=F.one_hot(z[start:end], num_classes=max_atom_id),  # atom
        #     atom_mask=z[start:end].shape[0],  # atom
        #     # edge_mask=z[start:end].shape[0],
        #     # original_batch=original_batch[start:end], # atom
        #     batch=original_batch[start:end],  # atom
        #     natoms=natoms[i],  # molecule
        #     energy=energy[i],  # molecule
        #     energy_grad=energy_grad[start:end],  # atom
        #     npa_charges=npa_charges[start:end],  # atom
        #     species=torch.full((end - start,), species)  # atom
        # )

        data_list.append(data)

    # ② 保存到本地 .pt 文件
    torch.save(data_list, "../data_rd/"+file_name)

    return data_list

def split_dataset(data, train_ratio=0.6, val_ratio=0.3):

    train_set, temp_set = train_test_split(data, train_size=train_ratio, random_state=42)
    val_set, test_set = train_test_split(temp_set, train_size=val_ratio / (1 - train_ratio), random_state = 42)

    return train_set, val_set, test_set

def calculate_max_node_number(loaded_tensor):
    diff = torch.abs(loaded_tensor[1:] - loaded_tensor[:-1])
    max_diff = torch.max(diff)

    return max_diff.item()


def read_dataset(path, max_atom_number=29, max_atom_id=10, train=0.8, val=0.1):

    file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))
    pattern = re.compile(r"qm9star_(.+?)_chunk\d+_processed\.pt")

    data_list = []
    for file in file_list:
        match = pattern.search(file)
        name = match.group(1)

        loaded_tensor = torch.load(file, weights_only=False)
        dataset = generate_a_sample_dataset(loaded_tensor, name, max_atom_number, max_atom_id, match.group(0))

        data_list += dataset

    # train_ratio and val_ratio are the ratio of the entire dataset
    train_list, val_list, test_list = split_dataset(data_list, train_ratio=train, val_ratio=val)

    return train_list, val_list, test_list

def calculate_n_nodes(loaded_tensor_z):

    values, counts = torch.unique(loaded_tensor_z, return_counts=True)

    freq_dict = {int(v): int(c) for v, c in zip(values, counts)}
    freq_dict = dict(sorted(freq_dict.items()))

    pretty = ", ".join(f"{k}: {v}" for k, v in freq_dict.items())
    print(pretty)

    return freq_dict


def calculate_datasets_config(path):
    file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))
    pattern = re.compile(r"qm9star_(.+?)_chunk\d+_processed\.pt")

    data_list = []
    n_nodes = {}
    for file in file_list:
        match = pattern.search(file)
        name = match.group(1)

        loaded_tensor = torch.load(file, weights_only=False)

        curr_n_nodes = calculate_n_nodes(loaded_tensor[0].z)
        n_nodes.update(curr_n_nodes)
        print(curr_n_nodes)


def estimate_dataset(path):
    """
    Calculate config of dataset, currently calculate the max number atoms in one molecule
    and the largest atom id used.
    :param path: path to dataset
    Result; max_atom_number:  29 max_atom_id:  10
    """
    max_atom_number = -1
    max_atom_id = -1

    file_list = glob.glob(os.path.join(path, "qm9star_*_chunk*_processed.pt"))

    for file in file_list:

        loaded_tensor = torch.load(file)

        curr_max_atom_number = torch.max(nbatch_transform(loaded_tensor[1]['z'])).item()
        curr_max_atom_id = torch.max(loaded_tensor[0].z).item()

        if curr_max_atom_number > max_atom_number:
            max_atom_number = curr_max_atom_number
        if curr_max_atom_id > max_atom_id:
            max_atom_id = curr_max_atom_id

        print('current file: ', file, 'max_atom_number: ', curr_max_atom_number, 'max_atom_id: ', curr_max_atom_id, '\n')

    print("max_atom_number: ", max_atom_number, "max_atom_id: ", max_atom_id)


if __name__ == "__main__":

    # calculate_datasets_config("data/processed")

    max_atom_number = 29
    max_atom_id = 10

    # load data set
    train_list, val_list, test_list = read_dataset("../data/processed",
                                                   max_atom_number = max_atom_number, max_atom_id = max_atom_id)


    # loaded_list = torch.load("../data_rd/qm9star_anion_chunk00_processed.pt", map_location="cpu")
    # loader = DataLoader(loaded_list)
    # print(loader)
    train_loader = DataLoader(train_list, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_list, batch_size=32)
    test_loader = DataLoader(test_list)