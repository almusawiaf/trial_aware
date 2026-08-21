"""
dataset.py -- turns (subject_id, trial_id, criterion, label) triples into
padded tensors matching the shapes COMPOSE's model.py expects:

  ehr_ids     : (B, T, 3, K)   code-vocab ids, 3 categories (diag/med/lab),
                                K codes per category (padded with 0)
  ehr_mask    : (B, T, 3, K)   1 = real code, 0 = padding
  visit_mask  : (B, T)         1 = real visit, 0 = padded visit
  demo        : (B, demo_dim)  all-zero (see data_utils.py docstring)
  crit_ids    : (B, L)         criterion token ids (entity_type, code, operator, value-slot)
  crit_mask   : (B, L)         1 = real token, 0 = padding
  crit_value  : (B, 1)         raw scalar value for GT/LT/EQ criteria (0 if none)
  label       : (B,)           0=match, 1=mismatch, 2=unknown
"""
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from config import config
from vocab import code_vocab_for


CATEGORY_ORDER = ["diagnosis", "medication", "lab"]


class ComposeDataset(Dataset):
    def __init__(self, triples, visit_sequences: dict, vocabs: dict):
        """
        triples: list of (subject_id, trial_id, criterion, label)
        visit_sequences: dict[sid] -> list of visit dicts (see data_utils.py)
        vocabs: dict from vocab.build_vocabs
        """
        self.triples = triples
        self.visit_sequences = visit_sequences
        self.vocabs = vocabs

    def __len__(self):
        return len(self.triples)

    def _encode_visits(self, sid):
        visits = self.visit_sequences.get(sid, [{"diagnosis": [], "medication": [], "lab": []}])
        visits = visits[: config.MAX_VISITS]
        T, K = len(visits), config.MAX_CODES_PER_CATEGORY

        ehr_ids = torch.zeros(config.MAX_VISITS, 3, K, dtype=torch.long)
        ehr_mask = torch.zeros(config.MAX_VISITS, 3, K, dtype=torch.float32)
        visit_mask = torch.zeros(config.MAX_VISITS, dtype=torch.float32)

        for t, visit in enumerate(visits):
            visit_mask[t] = 1.0
            for cat_idx, cat in enumerate(CATEGORY_ORDER):
                codes = visit.get(cat, [])[:K]
                vocab = code_vocab_for(self.vocabs, cat)
                for k, code in enumerate(codes):
                    ehr_ids[t, cat_idx, k] = vocab(code)
                    ehr_mask[t, cat_idx, k] = 1.0

        return ehr_ids, ehr_mask, visit_mask

    def _encode_criterion(self, criterion) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = config.MAX_CRITERION_TOKENS  # entity_type, code, operator, value
        crit_ids = torch.zeros(L, dtype=torch.long)
        crit_mask = torch.zeros(L, dtype=torch.float32)

        entity_type_vocab = self.vocabs["entity_type"]
        operator_vocab = self.vocabs["operator"]
        code_vocab = code_vocab_for(self.vocabs, criterion.entity_type)

        crit_ids[0] = entity_type_vocab(criterion.entity_type)
        crit_mask[0] = 1.0

        crit_ids[1] = code_vocab(criterion.entity_code)
        crit_mask[1] = 1.0

        op_name = criterion.operator.value if hasattr(criterion.operator, "value") else str(criterion.operator)
        crit_ids[2] = operator_vocab(op_name)
        crit_mask[2] = 1.0

        value = 0.0
        if criterion.value is not None:
            # Slot 3 reuses the operator id as its discrete token (there is no
            # separate vocabulary for "value tokens"); the actual scalar goes
            # in through value_proj on top of this embedding -- see
            # ComposeModel.encode_criterion in model.py.
            crit_ids[3] = crit_ids[2]
            crit_mask[3] = 1.0
            value = float(criterion.value)

        crit_value = torch.tensor([value], dtype=torch.float32)
        return crit_ids, crit_mask, crit_value

    def __getitem__(self, idx):
        sid, tid, criterion, label = self.triples[idx]
        ehr_ids, ehr_mask, visit_mask = self._encode_visits(sid)
        crit_ids, crit_mask, crit_value = self._encode_criterion(criterion)
        return {
            "ehr_ids": ehr_ids,
            "ehr_mask": ehr_mask,
            "visit_mask": visit_mask,
            "demo": torch.zeros(config.DEMO_DIM, dtype=torch.float32),
            "crit_ids": crit_ids,
            "crit_mask": crit_mask,
            "crit_value": crit_value,
            "label": torch.tensor(label, dtype=torch.long),
            "entity_type": criterion.entity_type,  # str, routes the code embedding table
            "subject_id": sid,
            "trial_id": tid,
        }


def collate_fn(batch: List[dict]) -> dict:
    out = {}
    tensor_keys = ["ehr_ids", "ehr_mask", "visit_mask", "demo", "crit_ids", "crit_mask", "crit_value", "label"]
    for key in tensor_keys:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["entity_type"] = [b["entity_type"] for b in batch]
    out["subject_id"] = [b["subject_id"] for b in batch]
    out["trial_id"] = [b["trial_id"] for b in batch]
    return out
