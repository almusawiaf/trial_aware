"""
trial_embedding.py

Builds trial embeddings z_T^inc and z_T^exc in the SAME space R^d as the
patient embeddings produced by HeteroGNNEncoder.encode(), by:

  1. encoding each criterion c into z_c (CriterionEncoder), combining the
     POST-GNN concept-node embedding with the criterion's operator/value/
     severity-weight metadata, and
  2. attention-pooling the inclusion-criteria z_c's into z_inc and the
     exclusion-criteria z_c's into z_exc (TrialEncoder), using SEPARATE
     pooling parameters for each group to allow distinct representations.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trial_graph import Criterion, Operator, Trial

_OPERATORS = list(Operator)


class CriterionEncoder(nn.Module):
    """Produces z_c for a single criterion given its concept embedding + metadata."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Operator one-hot + normalized value + severity weight
        meta_dim = len(Operator) + 2  # 8 + 2 = 10
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + meta_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _meta_features(self, criterion, device):
        # 1. Severity weight
        weight_val = getattr(criterion, 'severity_weight', 1.0)
        if not isinstance(weight_val, (int, float)):
            weight_val = 1.0
        weight = torch.tensor([weight_val], dtype=torch.float32, device=device)
        
        # 2. Normalized value
        value_val = getattr(criterion, 'value', 0.0)
        if not isinstance(value_val, (int, float)):
            try:
                value_val = float(value_val)
            except (ValueError, TypeError):
                value_val = 0.0
        # Normalize (adjust max_value based on your data range)
        max_value = 100.0  # Adjust based on your lab value ranges
        value_norm = torch.tensor([value_val / max_value], dtype=torch.float32, device=device)
        
        # 3. Operator one-hot
        op_val = getattr(criterion, 'operator', None)
        if op_val is None:
            op_idx = 0
        elif hasattr(op_val, 'to_numeric'):
            op_idx = op_val.to_numeric()
        elif isinstance(op_val, int):
            op_idx = op_val
        else:
            op_idx = 0
        
        num_operators = len(Operator)
        op_one_hot = F.one_hot(torch.tensor(op_idx, device=device), 
                               num_classes=num_operators).float()
        
        # Concatenate: [weight, value_norm, op_one_hot]
        return torch.cat([weight, value_norm, op_one_hot], dim=-1)

    def forward(self, concept_embedding: torch.Tensor, criterion: Criterion) -> torch.Tensor:
        meta = self._meta_features(criterion, concept_embedding.device)
        return self.mlp(torch.cat([concept_embedding, meta]))


class TrialEncoder(nn.Module):
    """
    Attention-pools a list of z_c vectors into a single z_inc (or z_exc)
    vector. Uses SEPARATE pooling parameters for inclusion and exclusion
    to allow distinct representations for each group.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # Separate pooling parameters for inclusion and exclusion
        self.W_pool_inc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_pool_inc = nn.Linear(embed_dim, 1, bias=False)
        self.W_pool_exc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_pool_exc = nn.Linear(embed_dim, 1, bias=False)

    def pool_inc(self, z_criteria: List[torch.Tensor]) -> torch.Tensor:
        """Pool inclusion criteria with inclusion-specific parameters."""
        if len(z_criteria) == 0:
            return None
        Z = torch.stack(z_criteria, dim=0)                      # [C, d]
        scores = self.w_pool_inc(torch.tanh(self.W_pool_inc(Z))).squeeze(-1)  # [C]
        attn = F.softmax(scores, dim=0)                         # beta_c / alpha_c
        return (attn.unsqueeze(-1) * Z).sum(dim=0)               # [d]
    
    def pool_exc(self, z_criteria: List[torch.Tensor]) -> torch.Tensor:
        """Pool exclusion criteria with exclusion-specific parameters."""
        if len(z_criteria) == 0:
            return None
        Z = torch.stack(z_criteria, dim=0)                      # [C, d]
        scores = self.w_pool_exc(torch.tanh(self.W_pool_exc(Z))).squeeze(-1)  # [C]
        attn = F.softmax(scores, dim=0)                         # beta_c / alpha_c
        return (attn.unsqueeze(-1) * Z).sum(dim=0)               # [d]


def get_concept_embedding(entity_type: str, 
                           entity_code: str,
                           entity_maps: Dict[str, Dict[str, int]],
                           post_gnn_embeddings: Dict[str, torch.Tensor],
                           embed_dim: int, 
                           device) -> torch.Tensor:
    """Looks up the post-GNN embedding for a criterion's target concept node."""
    # Safely fetch the entity map for diagnosis, medication, lab, procedure, etc.
    node_map = entity_maps.get(entity_type)

    # If entity_type is missing from entity_maps (e.g., 'procedure') or code is missing/unknown:
    if node_map is None or entity_code not in node_map:
        return torch.zeros(embed_dim, device=device)

    node_idx = node_map[entity_code]
    return post_gnn_embeddings[entity_type][node_idx]


def encode_trial(trial: Trial,
                  entity_maps: Dict[str, Dict[str, int]],
                  post_gnn_embeddings: Dict[str, torch.Tensor],
                  criterion_encoder: CriterionEncoder,
                  trial_encoder: TrialEncoder,
                  embed_dim: int,
                  device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (z_inc, z_exc) for a single trial, or zero vectors if a group is empty."""

    def encode_group(criteria: List[Criterion], pool_type: str = 'inc') -> torch.Tensor:
        z_list = []
        for c in criteria:
            concept_emb = get_concept_embedding(
                c.entity_type, c.entity_code, entity_maps, post_gnn_embeddings, embed_dim, device
            )
            z_list.append(criterion_encoder(concept_emb, c))
        
        if len(z_list) == 0:
            return torch.zeros(embed_dim, device=device)
        
        # Use the appropriate pooling method
        if pool_type == 'inc':
            pooled = trial_encoder.pool_inc(z_list)
        else:
            pooled = trial_encoder.pool_exc(z_list)
        
        return pooled if pooled is not None else torch.zeros(embed_dim, device=device)

    z_inc = encode_group(trial.inclusion_criteria, 'inc')
    z_exc = encode_group(trial.exclusion_criteria, 'exc')
    return z_inc, z_exc


def encode_all_trials(trials: List[Trial], entity_maps, post_gnn_embeddings,
                       criterion_encoder, trial_encoder, embed_dim, device):
    """Returns dict trial_id -> (z_inc, z_exc)."""
    return {
        t.trial_id: encode_trial(t, entity_maps, post_gnn_embeddings,
                                  criterion_encoder, trial_encoder, embed_dim, device)
        for t in trials
    }