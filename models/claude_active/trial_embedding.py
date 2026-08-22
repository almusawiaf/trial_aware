"""
trial_embedding.py
Vectorized Trial and Criterion Encoders for Fast Batch GPU Computation
"""

import logging
from typing import Dict, List, Tuple, Optional
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from trial_graph import Operator, Trial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class CriterionEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        meta_dim = len(Operator) + 3  # 8 + 2 original + 1 new is_inclusion flag
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

    # NEW: explicit flag so the encoder always knows "this is an
    # inclusion criterion" vs "this is an exclusion criterion", even if
    # the concept embedding itself is weak or missing (OOV -> zero vector).
    is_inc_val = 1.0 if getattr(criterion, 'is_inclusion', True) else 0.0
    is_inclusion_flag = torch.tensor([is_inc_val], dtype=torch.float32, device=device)

    return torch.cat([weight, value_norm, op_one_hot, is_inclusion_flag], dim=-1)


def normalize_entity_code(entity_type: str, entity_code: str) -> str:
    """
    Makes a trial criterion's code comparable to the codes in entity_maps.

    Diagnoses: your ICD9->ICD10 crosswalk strips dots when building the
    vocabulary (K76.9 -> K769), but trial criteria still have dots.

    Medications: NDC codes in the trial JSON were apparently stored/read as
    floats at some point upstream (e.g. '10019095601.0' instead of
    '10019095601') -- strip that artifact. Some also lost leading zeros in
    the process (floats don't preserve them), so if the digit count is short
    of the standard 11-digit NDC length, left-pad with zeros. This is a
    heuristic -- NDC leading-zero placement technically depends on which of
    3 formats (4-4-2 / 5-3-2 / 5-4-1) the original code used, so it won't be
    100% correct, but it recovers a real chunk of otherwise-lost matches.
    """
    code = str(entity_code).strip()
    if entity_type == 'diagnosis':
        code = code.replace('.', '').upper()
    elif entity_type == 'medication':
        code = re.sub(r'\.0+$', '', code)  # drop float-artifact suffix
        code = re.sub(r'[^0-9]', '', code)  # drop stray dashes/spaces if any
        if code.isdigit() and len(code) < 11:
            code = code.zfill(11)
    return code


# ---------------------------------------------------------------------------
# Optional hierarchical diagnosis alignment (entity_alignment.py). When
# initialized via init_diagnosis_aligner(entity_maps), a trial diagnosis code
# that isn't an exact match to any patient code will resolve to a
# clinically-equivalent patient code in the same ICD-10 category, recovering
# a large fraction of criteria that were previously discarded as OOV (see
# diagnose_entity_resolution.py). Left None by default so behavior is
# unchanged unless a caller explicitly opts in.
# ---------------------------------------------------------------------------
_DX_ALIGNER = None


def init_diagnosis_aligner(entity_maps: Dict[str, Dict[str, int]]):
    """Build the module-level diagnosis aligner from the patient vocabulary.
    Call once (in train.py / evaluate.py) before encoding trials."""
    global _DX_ALIGNER
    try:
        from entity_alignment import DiagnosisAligner
        _DX_ALIGNER = DiagnosisAligner(set(entity_maps.get('diagnosis', {}).keys()))
        logging.info(f"[alignment] Hierarchical diagnosis aligner initialized "
                     f"({len(_DX_ALIGNER.exact)} patient diagnosis codes, "
                     f"{len(_DX_ALIGNER.by_category)} ICD-10 categories).")
    except Exception as e:
        logging.warning(f"[alignment] Could not initialize diagnosis aligner: {e}. "
                         f"Falling back to exact-match only.")
        _DX_ALIGNER = None


def _aligned_diagnosis_code(entity_code: str, entity_maps: Dict[str, Dict[str, int]]) -> Optional[str]:
    """Return the patient-vocabulary diagnosis code this trial code aligns to
    (exact or category-level), or None. Only active if the aligner is set."""
    if _DX_ALIGNER is None:
        return None
    return _DX_ALIGNER.resolve(entity_code)


def is_resolvable_code(entity_type: str, entity_code: str, entity_maps: Dict[str, Dict[str, int]]) -> bool:
    """
    True only if this criterion has a real code we can actually look up.
    Filters out two different kinds of "nothing to look up here":
      - placeholder codes like 'UNMATCHED_47327' (upstream extraction never
        found a real code -- these are not failed matches, they're not codes)
      - genuinely OOV codes not present in entity_maps for this entity_type

    When the hierarchical diagnosis aligner is active, a diagnosis code also
    counts as resolvable if a clinically-equivalent patient code exists in
    the same ICD-10 category (recovers crosswalk-granularity mismatches).
    """
    code = str(entity_code)
    if code.startswith('UNMATCHED_') or code.strip() == '' or code.lower() == 'none':
        return False
    node_map = entity_maps.get(entity_type)
    if node_map is None:
        return False
    if normalize_entity_code(entity_type, code) in node_map:
        return True
    if entity_type == 'diagnosis' and _aligned_diagnosis_code(code, entity_maps) is not None:
        return True
    return False


def get_concept_embedding(entity_type: str, entity_code: str, entity_maps: Dict[str, Dict[str, int]], post_gnn_embeddings: Dict[str, torch.Tensor], embed_dim: int, device) -> torch.Tensor:
    node_map = entity_maps.get(entity_type)
    if node_map is None:
        return torch.zeros(embed_dim, device=device)
    norm_code = normalize_entity_code(entity_type, entity_code)
    if norm_code not in node_map and entity_type == 'diagnosis':
        # Try hierarchical alignment before giving up on this diagnosis code.
        aligned = _aligned_diagnosis_code(entity_code, entity_maps)
        if aligned is not None:
            norm_code = aligned
    if norm_code not in node_map:
        return torch.zeros(embed_dim, device=device)
    node_idx = node_map[norm_code]
    return post_gnn_embeddings[entity_type][node_idx]

def encode_all_trials(trials: List[Trial], entity_maps, post_gnn_embeddings, criterion_encoder, trial_encoder, embed_dim, device):
    """Vectorized trial encoding: Batches ALL 7,591 criteria into 1 GPU matrix pass."""
    
    # Step 1: Gather all criteria embeddings & metadata into single lists
    all_c_embs = []
    all_metas = []
    
    # Store index ranges to map back to individual trials
    # trial_map -> {trial_id: {'inc_slice': (start, end), 'exc_slice': (start, end)}}
    trial_slices = {}
    current_idx = 0

    for t in trials:
        t_info = {}

        # NEW: keep only criteria we can actually resolve to a real code.
        # Feeding in zero vectors for unresolvable criteria (old behavior)
        # made most criteria across most trials look nearly identical --
        # that's what was flattening z_inc/z_exc into near-duplicates.
        # Dropping them means only real signal reaches the pooling step.
        inc_criteria = [c for c in t.inclusion_criteria if is_resolvable_code(c.entity_type, c.entity_code, entity_maps)]
        exc_criteria = [c for c in t.exclusion_criteria if is_resolvable_code(c.entity_type, c.entity_code, entity_maps)]

        # Inclusion criteria tracking
        inc_len = len(inc_criteria)
        if inc_len > 0:
            for c in inc_criteria:
                c_emb = get_concept_embedding(c.entity_type, c.entity_code, entity_maps, post_gnn_embeddings, embed_dim, device)
                meta = extract_criterion_meta(c, device)
                all_c_embs.append(c_emb)
                all_metas.append(meta)
            t_info['inc_slice'] = (current_idx, current_idx + inc_len)
            current_idx += inc_len
        else:
            t_info['inc_slice'] = None

        # Exclusion criteria tracking
        exc_len = len(exc_criteria)
        if exc_len > 0:
            for c in exc_criteria:
                c_emb = get_concept_embedding(c.entity_type, c.entity_code, entity_maps, post_gnn_embeddings, embed_dim, device)
                meta = extract_criterion_meta(c, device)
                all_c_embs.append(c_emb)
                all_metas.append(meta)
            t_info['exc_slice'] = (current_idx, current_idx + exc_len)
            current_idx += exc_len
        else:
            t_info['exc_slice'] = None

        trial_slices[t.trial_id] = t_info

    # Step 2: SINGLE BATCH GPU PASS for all 7,591 criteria
    if all_c_embs:
        stacked_c_embs = torch.stack(all_c_embs, dim=0) # [7591, embed_dim]
        stacked_metas = torch.stack(all_metas, dim=0)   # [7591, meta_dim]
        
        # 1 GPU Kernel Launch instead of 7,591!
        all_criterion_embeds = criterion_encoder.forward_batch(stacked_c_embs, stacked_metas)
    else:
        all_criterion_embeds = torch.empty((0, embed_dim), device=device)

    # Step 3: Reconstruct Trial Embeddings using Slice Pooling
    trial_embeds = {}
    for t in trials:
        t_info = trial_slices[t.trial_id]
        
        # Pool Inclusion
        if t_info['inc_slice'] is not None:
            s_inc, e_inc = t_info['inc_slice']
            z_inc = trial_encoder.pool_group(all_criterion_embeds[s_inc:e_inc], pool_type='inc')
        else:
            z_inc = torch.zeros(embed_dim, device=device)

        # Pool Exclusion
        if t_info['exc_slice'] is not None:
            s_exc, e_exc = t_info['exc_slice']
            z_exc = trial_encoder.pool_group(all_criterion_embeds[s_exc:e_exc], pool_type='exc')
        else:
            z_exc = torch.zeros(embed_dim, device=device)

        trial_embeds[t.trial_id] = (z_inc, z_exc)

    return trial_embeds