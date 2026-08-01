"""
trial_embedding.py
Vectorized Trial and Criterion Encoders for Fast Batch GPU Computation
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from trial_graph import Operator, Trial


class CriterionEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        meta_dim = len(Operator) + 2  # 8 + 2 = 10
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + meta_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward_batch(self, concept_embeds: torch.Tensor, meta_features: torch.Tensor) -> torch.Tensor:
        """Batch input: concept_embeds [B, embed_dim], meta_features [B, meta_dim]"""
        return self.mlp(torch.cat([concept_embeds, meta_features], dim=-1))


class TrialEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_pool_inc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_pool_inc = nn.Linear(embed_dim, 1, bias=False)
        self.W_pool_exc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_pool_exc = nn.Linear(embed_dim, 1, bias=False)

    def pool_group(self, Z: torch.Tensor, pool_type: str = 'inc') -> torch.Tensor:
        """Batch input Z: [Num_Criteria, d]"""
        if Z.size(0) == 0:
            return None
        if pool_type == 'inc':
            scores = self.w_pool_inc(torch.tanh(self.W_pool_inc(Z))).squeeze(-1)
        else:
            scores = self.w_pool_exc(torch.tanh(self.W_pool_exc(Z))).squeeze(-1)
            
        attn = F.softmax(scores, dim=0)
        return (attn.unsqueeze(-1) * Z).sum(dim=0)


def extract_criterion_meta(criterion, device) -> torch.Tensor:
    weight_val = float(getattr(criterion, 'severity_weight', 1.0))
    value_val = float(getattr(criterion, 'value', 0.0) or 0.0)
    
    weight = torch.tensor([weight_val], dtype=torch.float32, device=device)
    value_norm = torch.tensor([value_val / 100.0], dtype=torch.float32, device=device)
    
    op_val = getattr(criterion, 'operator', None)
    op_idx = op_val.to_numeric() if hasattr(op_val, 'to_numeric') else (int(op_val) if isinstance(op_val, int) else 0)
    
    op_one_hot = F.one_hot(torch.tensor(op_idx, device=device), num_classes=len(Operator)).float()
    return torch.cat([weight, value_norm, op_one_hot], dim=-1)


def get_concept_embedding(entity_type: str, entity_code: str, entity_maps: Dict[str, Dict[str, int]], post_gnn_embeddings: Dict[str, torch.Tensor], embed_dim: int, device) -> torch.Tensor:
    node_map = entity_maps.get(entity_type)
    if node_map is None or entity_code not in node_map:
        return torch.zeros(embed_dim, device=device)
    node_idx = node_map[entity_code]
    return post_gnn_embeddings[entity_type][node_idx]


def encode_all_trials(trials: List[Trial], entity_maps, post_gnn_embeddings, criterion_encoder, trial_encoder, embed_dim, device):
    trial_embeds = {}
    for t in trials:
        # Process Inclusion Criteria
        z_inc_list = []
        for c in t.inclusion_criteria:
            c_emb = get_concept_embedding(c.entity_type, c.entity_code, entity_maps, post_gnn_embeddings, embed_dim, device)
            meta = extract_criterion_meta(c, device)
            z_inc_list.append(criterion_encoder.forward_batch(c_emb.unsqueeze(0), meta.unsqueeze(0)).squeeze(0))
        
        if z_inc_list:
            z_inc = trial_encoder.pool_group(torch.stack(z_inc_list, dim=0), pool_type='inc')
        else:
            z_inc = torch.zeros(embed_dim, device=device)

        # Process Exclusion Criteria
        z_exc_list = []
        for c in t.exclusion_criteria:
            c_emb = get_concept_embedding(c.entity_type, c.entity_code, entity_maps, post_gnn_embeddings, embed_dim, device)
            meta = extract_criterion_meta(c, device)
            z_exc_list.append(criterion_encoder.forward_batch(c_emb.unsqueeze(0), meta.unsqueeze(0)).squeeze(0))

        if z_exc_list:
            z_exc = trial_encoder.pool_group(torch.stack(z_exc_list, dim=0), pool_type='exc')
        else:
            z_exc = torch.zeros(embed_dim, device=device)

        trial_embeds[t.trial_id] = (z_inc, z_exc)

    return trial_embeds