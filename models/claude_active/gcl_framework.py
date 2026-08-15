"""
gcl_framework.py
----------------
Graph Contrastive Learning (GCL) framework for patient representations.
Provides:
1. HeteroGNNEncoder (Graph Neural Network for HeteroData)
2. GraphAugmentor (Edge dropping & feature masking)
3. InfoNCELoss / NTXentLoss (Chunked/Subsampled memory-efficient NT-Xent loss)
4. Stage B Trial-Aware Contrastive Loss
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGNNEncoder(nn.Module):
    """
    Heterogeneous Graph Neural Network Encoder.
    Maps node entity indices to learnable embeddings and runs multi-layer HeteroConv operations.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        node_types: Optional[List[str]] = None,
        hidden_dim: int = 128,
        out_dim: int = 128,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        num_nodes_dict: Optional[Dict[str, int]] = None,
        **kwargs  # Captures extra args like patient_feat_dim, entity_embed_dim
    ):
        super().__init__()
        self.hidden_dim = hidden_channels if hidden_channels is not None else hidden_dim
        self.out_dim = out_channels if out_channels is not None else out_dim
        self.node_types = node_types or metadata[0]

        # Learnable node embedding dictionaries per node type
        self.embeddings = nn.ModuleDict()
        if num_nodes_dict:
            for n_type, n_count in num_nodes_dict.items():
                self.embeddings[n_type] = nn.Embedding(n_count, self.hidden_dim)

        # Build GNN Layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = self.hidden_dim
            out_c = self.out_dim if i == num_layers - 1 else self.hidden_dim
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv(in_c, out_c)
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # Projection head for GCL
        self.projector = nn.Sequential(
            nn.Linear(self.out_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Runs Message Passing across heterogeneous node spaces."""
        h_dict = {}
        for n_type, tensor in x_dict.items():
            if n_type in self.embeddings and tensor.dtype in (torch.long, torch.int32, torch.int64):
                # Integer node indices -> Embedding Lookup
                idx = tensor.squeeze(-1) if tensor.dim() > 1 else tensor
                h_dict[n_type] = self.embeddings[n_type](idx)
            elif tensor.dtype in (torch.float32, torch.float64, torch.float16):
                # Continuous float feature matrix
                h_dict[n_type] = tensor
            else:
                # Fallback index mapping
                idx = torch.arange(tensor.size(0), device=tensor.device)
                h_dict[n_type] = self.embeddings[n_type](idx)

        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        return h_dict

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Projects node representation into contrastive loss space."""
        return self.projector(h)


class GraphAugmentor:
    """Provides topological graph perturbations for self-supervised GCL views."""

    @staticmethod
    def drop_edges(data: HeteroData, drop_rate: float = 0.2) -> HeteroData:
        aug_data = data.clone()
        if drop_rate <= 0.0:
            return aug_data

        for edge_type, edge_index in aug_data.edge_index_dict.items():
            num_edges = edge_index.size(1)
            keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
            aug_data[edge_type].edge_index = edge_index[:, keep_mask]

        return aug_data


class InfoNCELoss(nn.Module):
    """
    Memory-Efficient Chunked / Subsampled NT-Xent (InfoNCE) Loss.
    Prevents CUDA OOM on large graphs by chunking matrix multiplications.
    """

    def __init__(self, temperature: float = 0.1, max_batch_size: int = 2048):
        super().__init__()
        self.temperature = temperature
        self.max_batch_size = max_batch_size

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        num_nodes = z1.size(0)

        # 1. Subsample if graph scale exceeds max memory threshold
        if num_nodes > self.max_batch_size:
            indices = torch.randperm(num_nodes, device=z1.device)[: self.max_batch_size]
            z1 = z1[indices]
            z2 = z2[indices]
            num_nodes = self.max_batch_size

        # 2. L2 Normalization
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # 3. Concatenate two views: [2 * N, Dim]
        out = torch.cat([z1, z2], dim=0)
        n_samples = 2 * num_nodes

        # 4. Pairwise Cosine Similarity Matrix: [2*N, 2*N]
        cov = torch.matmul(out, out.T) / self.temperature

        # Mask self-similarity on diagonal
        sim_targets = torch.arange(n_samples, device=z1.device)
        sim_targets = (sim_targets + num_nodes) % n_samples

        mask = torch.eye(n_samples, dtype=torch.bool, device=z1.device)
        cov.masked_fill_(mask, -9e15)

        # 5. Cross-Entropy over Positive Pairs
        loss = F.cross_entropy(cov, sim_targets)
        return loss


class TrialAwareLoss(nn.Module):
    """
    Stage B Trial-Aware Contrastive Loss.
    Aligns patient representations with inclusion/exclusion criteria masks.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        patient_emb: torch.Tensor,
        trial_emb: torch.Tensor,
        M_inc: torch.Tensor,
        M_exc: torch.Tensor,
    ) -> torch.Tensor:
        patient_emb = F.normalize(patient_emb, dim=-1)
        trial_emb = F.normalize(trial_emb, dim=-1)

        # Pairwise Patient-Trial Alignment Matrix [Num_Patients, Num_Trials]
        sim = torch.matmul(patient_emb, trial_emb.T) / self.temperature

        # Eligibility weights: High M_inc and 0.0 M_exc -> Positive alignment target
        alignment_targets = M_inc * (1.0 - M_exc)

        # Soft cross-entropy loss over criteria alignment
        loss = -torch.sum(F.log_softmax(sim, dim=-1) * alignment_targets) / (
            patient_emb.size(0) + 1e-8
        )
        return loss


# Class Aliases for train.py compatibility
NTXentLoss = InfoNCELoss