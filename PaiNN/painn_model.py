"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

MIT License

Copyright (c) 2021 www.compscience.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
import typing

import torch
from torch import nn

if typing.TYPE_CHECKING:
    from torch_geometric.data.batch import Batch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, segment_coo

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BackboneInterface, GraphModelMixin, HeadInterface
from fairchem.core.models.gemnet.layers.base_layers import ScaledSiLU
from fairchem.core.models.gemnet.layers.embedding_block import AtomEmbedding
from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis
from fairchem.core.modules.scaling import ScaleFactor
from fairchem.core.modules.scaling.compat import load_scales_compat

from fairchem.core.models.painn.utils import get_edge_id, repeat_blocks


@registry.register_model("painn")
class PaiNN(nn.Module, GraphModelMixin):
    r"""PaiNN model based on the description in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties
    and molecular spectra, https://arxiv.org/abs/2102.03150.
    """

    def __init__(
            self,
            hidden_channels: int = 512,
            num_layers: int = 6,
            num_rbf: int = 128,
            cutoff: float = 12.0,
            max_neighbors: int = 50,
            rbf: dict[str, str] | None = None,
            envelope: dict[str, str | int] | None = None,
            regress_forces: bool = True,
            direct_forces: bool = True,
            use_pbc: bool = True,
            use_pbc_single: bool = False,
            otf_graph: bool = True,
            num_elements: int = 83,
            scale_file: str | None = None,
    ) -> None:
        if envelope is None:
            envelope = {"name": "polynomial", "exponent": 5}
        if rbf is None:
            rbf = {"name": "gaussian"}
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single

        # Borrowed from GemNet.
        self.symmetric_edge_symmetrization = False

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_rbf).jittable()
            )
            self.update_layers.append(PaiNNUpdate(hidden_channels))
            setattr(self, "upd_out_scalar_scale_%d" % i, ScaleFactor())

        self.out_energy = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.out_npa = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        if self.regress_forces is True and self.direct_forces is True:
            self.out_forces = PaiNNOutput(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()

        load_scales_compat(self, scale_file)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_energy[0].weight)
        self.out_energy[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_energy[2].weight)
        self.out_energy[2].bias.data.fill_(0)

    # Borrowed from GemNet.
    def select_symmetric_edges(
            self, tensor, mask, reorder_idx, inverse_neg
    ) -> torch.Tensor:
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        return tensor_cat[reorder_idx]

    # Borrowed from GemNet.
    def symmetrize_edges(
            self,
            edge_index,
            cell_offsets,
            neighbors,
            batch_idx,
            reorder_tensors,
            reorder_tensors_invneg,
    ):
        """
        Symmetrize edges to ensure existence of counter-directional edges.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors.
        If `symmetric_edge_symmetrization` is False,
        we only use i->j edges here. So we lose some j->i edges
        and add others by making it symmetric.
        If `symmetric_edge_symmetrization` is True,
        we always use both directions.
        """
        num_atoms = batch_idx.shape[0]

        if self.symmetric_edge_symmetrization:
            edge_index_bothdir = torch.cat(
                [edge_index, edge_index.flip(0)],
                dim=1,
            )
            cell_offsets_bothdir = torch.cat(
                [cell_offsets, -cell_offsets],
                dim=0,
            )

            # Filter for unique edges
            edge_ids = get_edge_id(edge_index_bothdir, cell_offsets_bothdir, num_atoms)
            unique_ids, unique_inv = torch.unique(edge_ids, return_inverse=True)
            perm = torch.arange(
                unique_inv.size(0),
                dtype=unique_inv.dtype,
                device=unique_inv.device,
            )
            unique_idx = scatter(
                perm,
                unique_inv,
                dim=0,
                dim_size=unique_ids.shape[0],
                reduce="min",
            )
            edge_index_new = edge_index_bothdir[:, unique_idx]

            # Order by target index
            edge_index_order = torch.argsort(edge_index_new[1])
            edge_index_new = edge_index_new[:, edge_index_order]
            unique_idx = unique_idx[edge_index_order]

            # Subindex remaining tensors
            cell_offsets_new = cell_offsets_bothdir[unique_idx]
            reorder_tensors = [
                self.symmetrize_tensor(tensor, unique_idx, False)
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.symmetrize_tensor(tensor, unique_idx, True)
                for tensor in reorder_tensors_invneg
            ]

            # Count edges per image
            # segment_coo assumes sorted edge_index_new[1] and batch_idx
            ones = edge_index_new.new_ones(1).expand_as(edge_index_new[1])
            neighbors_per_atom = segment_coo(
                ones, edge_index_new[1], dim_size=num_atoms
            )
            neighbors_per_image = segment_coo(
                neighbors_per_atom, batch_idx, dim_size=neighbors.shape[0]
            )
        else:
            # Generate mask
            mask_sep_atoms = edge_index[0] < edge_index[1]
            # Distinguish edges between the same (periodic) atom by ordering the cells
            cell_earlier = (
                    (cell_offsets[:, 0] < 0)
                    | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
                    | (
                            (cell_offsets[:, 0] == 0)
                            & (cell_offsets[:, 1] == 0)
                            & (cell_offsets[:, 2] < 0)
                    )
            )
            mask_same_atoms = edge_index[0] == edge_index[1]
            mask_same_atoms &= cell_earlier
            mask = mask_sep_atoms | mask_same_atoms

            # Mask out counter-edges
            edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

            # Concatenate counter-edges after normal edges
            edge_index_cat = torch.cat(
                [edge_index_new, edge_index_new.flip(0)],
                dim=1,
            )

            # Count remaining edges per image
            batch_edge = torch.repeat_interleave(
                torch.arange(neighbors.size(0), device=edge_index.device),
                neighbors,
            )
            batch_edge = batch_edge[mask]
            # segment_coo assumes sorted batch_edge
            # Factor 2 since this is only one half of the edges
            ones = batch_edge.new_ones(1).expand_as(batch_edge)
            neighbors_per_image = 2 * segment_coo(
                ones, batch_edge, dim_size=neighbors.size(0)
            )

            # Create indexing array
            edge_reorder_idx = repeat_blocks(
                torch.div(neighbors_per_image, 2, rounding_mode="floor"),
                repeats=2,
                continuous_indexing=True,
                repeat_inc=edge_index_new.size(1),
            )

            # Reorder everything so the edges of every image are consecutive
            edge_index_new = edge_index_cat[:, edge_reorder_idx]
            cell_offsets_new = self.select_symmetric_edges(
                cell_offsets, mask, edge_reorder_idx, True
            )
            reorder_tensors = [
                self.select_symmetric_edges(tensor, mask, edge_reorder_idx, False)
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.select_symmetric_edges(tensor, mask, edge_reorder_idx, True)
                for tensor in reorder_tensors_invneg
            ]

        # Indices for swapping c->a and a->c (for symmetric MP)
        # To obtain these efficiently and without any index assumptions,
        # we get order the counter-edge IDs and then
        # map this order back to the edge IDs.
        # Double argsort gives the desired mapping
        # from the ordered tensor to the original tensor.
        edge_ids = get_edge_id(edge_index_new, cell_offsets_new, num_atoms)
        order_edge_ids = torch.argsort(edge_ids)
        inv_order_edge_ids = torch.argsort(order_edge_ids)
        edge_ids_counter = get_edge_id(
            edge_index_new.flip(0), -cell_offsets_new, num_atoms
        )
        order_edge_ids_counter = torch.argsort(edge_ids_counter)
        id_swap = order_edge_ids_counter[inv_order_edge_ids]

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_per_image,
            reorder_tensors,
            reorder_tensors_invneg,
            id_swap,
        )

    def generate_graph_values(self, data):
        graph = self.generate_graph(data)

        # Unit vectors pointing from edge_index[1] to edge_index[0],
        # i.e., edge_index[0] - edge_index[1] divided by the norm.
        # make sure that the distances are not close to zero before dividing
        mask_zero = torch.isclose(graph.edge_distance, torch.tensor(0.0), atol=1e-6)
        # graph.edge_distance[mask_zero] = 1.0e-6
        graph.edge_distance = torch.where(mask_zero, torch.full_like(graph.edge_distance, 1.0e-6), graph.edge_distance)
        edge_vector = graph.edge_distance_vec / graph.edge_distance[:, None]

        empty_image = graph.neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )

        # Symmetrize edges for swapping in symmetric message passing
        (
            edge_index,
            cell_offsets,
            neighbors,
            [edge_dist],
            [edge_vector],
            id_swap,
        ) = self.symmetrize_edges(
            graph.edge_index,
            graph.cell_offsets,
            graph.neighbors,
            data.batch,
            [graph.edge_distance],
            [edge_vector],
        )

        return (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        )

    def generate_toy_dataset(self, num_samples: int = 100, num_atoms_range: tuple = (3, 10)):
        """
        Generates a toy dataset compatible with the PaiNN model.

        Args:
            num_samples (int): Number of samples in the dataset.
            num_atoms_range (tuple): Range for the number of atoms per molecule.

        Returns:
            list[torch_geometric.data.Data]: A list of molecular data objects.
        """
        import numpy as np
        from torch_geometric.data import Data

        def random_molecule(num_atoms):
            """
            Generate a random molecule with the specified number of atoms.
            - x: Random node features.
            - pos: Random positions.
            - z: Random atomic numbers.
            - y: Random target property (e.g., energy).
            """
            x = torch.randn(num_atoms, 5)  # Random 5D node features
            pos = torch.randn(num_atoms, 3)  # Random 3D atom positions
            z = torch.randint(1, 10, (num_atoms,))  # Atomic numbers (H=1, C=6, O=8, etc.)
            y = torch.randn(1)  # Random scalar property (e.g., energy)
            return Data(x=x, pos=pos, atomic_numbers=z, y=y)

        dataset = [random_molecule(np.random.randint(*num_atoms_range)) for _ in range(num_samples)]
        return dataset

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        pos = data.pos
        batch = data.batch
        z = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos = pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        ) = self.generate_graph_values(data)

        assert z.dim() == 1
        assert z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec
            x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)

        #### Output block #####################################################

        per_atom_energy = self.out_energy(x).squeeze(1)
        energy = scatter(per_atom_energy, batch, dim=0)
        outputs = {"energy": energy}

        if self.regress_forces:
            if self.direct_forces:
                forces = self.out_forces(x, vec)
            else:
                forces = (
                        -1
                        * torch.autograd.grad(
                    x,
                    pos,
                    grad_outputs=torch.ones_like(x),
                    create_graph=True,
                )[0]
                )
            outputs["forces"] = forces

        npa_charges = self.out_npa(x).squeeze(1)
        # npa_charges = scatter(per_atom_npa, batch, dim=0)
        outputs["npa_charges"] = npa_charges

        return outputs

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"max_neighbors={self.max_neighbors}, "
            f"cutoff={self.cutoff})"
        )


@registry.register_model("painn_backbone")
class PaiNNBackbone(PaiNN, BackboneInterface):
    @conditional_grad(torch.enable_grad())
    def forward(self, data) -> dict[str, torch.Tensor]:
        pos = data.pos
        z = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos = pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        ) = self.generate_graph_values(data)

        assert z.dim() == 1
        assert z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec
            x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)

        return {"node_embedding": x, "node_vec": vec}


class PaiNNMessage(MessagePassing):
    def __init__(
            self,
            hidden_channels,
            num_rbf,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(self, x, vec, edge_index, edge_rbf, edge_vector):
        xh = self.x_proj(self.x_layernorm(x))

        # TODO(@abhshkdz): Nans out with AMP here during backprop. Debug / fix.
        rbfh = self.rbf_proj(edge_rbf)

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
            self,
            features: tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            dim_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
            self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return inputs


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        # NOTE: Can't use torch.norm because the gradient is NaN for input = 0.
        # Add an epsilon offset to make sure sqrt is always positive.
        x_vec_h = self.xvec_proj(
            torch.cat([x, torch.sqrt(torch.sum(vec2 ** 2, dim=-2) + 1e-8)], dim=-1)
        )
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec1

        return dx, dvec


class PaiNNOutput(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
            self,
            hidden_channels,
            out_channels,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = ScaledSiLU()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v


@registry.register_model("painn_energy_head")
class PaiNNEnergyHead(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        backbone.out_energy = None
        self.out_energy = nn.Sequential(
            nn.Linear(backbone.hidden_channels, backbone.hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(backbone.hidden_channels // 2, 1),
        )

        nn.init.xavier_uniform_(self.out_energy[0].weight)
        self.out_energy[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_energy[2].weight)
        self.out_energy[2].bias.data.fill_(0)

    def forward(
            self, data: Batch, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        per_atom_energy = self.out_energy(emb["node_embedding"]).squeeze(1)
        return {"energy": scatter(per_atom_energy, data.batch, dim=0)}


@registry.register_model("painn_force_head")
class PaiNNForceHead(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        self.direct_forces = backbone.direct_forces

        if self.direct_forces:
            backbone.out_forces = None
            self.out_forces = PaiNNOutput(backbone.hidden_channels)

    def forward(
            self, data: Batch, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.direct_forces:
            forces = self.out_forces(emb["node_embedding"], emb["node_vec"])
        else:
            forces = (
                    -1
                    * torch.autograd.grad(
                emb["node_embedding"],
                data.pos,
                grad_outputs=torch.ones_like(emb["node_embedding"]),
                create_graph=True,
            )[0]
            )
        return {"forces": forces}
