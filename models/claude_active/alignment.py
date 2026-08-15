"""
alignment.py

The trial-aware contribution that was MISSING from the original code: the
dual-force alignment loss that pulls a patient toward the trial's inclusion
coordinate z_T^inc while pushing it away from hard-negative and random-
negative trials, plus the inference-time composite similarity score used
for ranking (P@k, ETE@k) once bifurcated trial coordinates are in play.

L_Align(P_i) =        M_inc(P_i, T_j)      * D(z_Pi, z_Tj^inc)^2                       [attraction]
       + lambda_1 * sum_{T_h in hard negs} M_exc(P_i, T_h) * max(0, m_hard - D(z_Pi, z_Th^exc))^2   [hard-neg repulsion]
       + lambda_2 * sum_{T_n in rand negs}                  max(0, m_rand - D(z_Pi, z_Tn^inc))^2    [random-neg repulsion]

D(.,.) is Euclidean distance. m_hard > m_rand by design: hard negatives are
"almost positives" that got disqualified on a single exclusion criterion, so
they need a larger safety margin than an arbitrary unrelated trial.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.norm(u - v, p=2)


class AlignmentLoss(nn.Module):
    def __init__(self, lambda_1: float = 1.0, lambda_2: float = 0.5,
                 margin_hard: float = 1.0, margin_rand: float = 0.5):
        super().__init__()
        assert margin_hard > margin_rand, "m_hard must exceed m_rand by design (see module docstring)."
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.margin_hard = margin_hard
        self.margin_rand = margin_rand

    def forward(self,
                z_patient: torch.Tensor,
                m_inc_target: float,
                z_target_inc: torch.Tensor,
                hard_negatives: List[Tuple[torch.Tensor, float]],   # [(z_exc, M_exc), ...]
                random_negatives: List[torch.Tensor]                # [z_inc, ...]
                ) -> torch.Tensor:
        # Attraction toward the positive trial's inclusion coordinate.
        attraction = m_inc_target * euclidean(z_patient, z_target_inc) ** 2

        # Repulsion from hard negatives' EXCLUSION coordinate, weighted by
        # how strongly the patient actually violates that trial's exclusion
        # criteria (M_exc) -- a trial the patient barely fails to be
        # excluded from contributes less penalty than a clear violation.
        hard_term = torch.zeros((), device=z_patient.device)
        for z_exc, m_exc in hard_negatives:
            margin_violation = F.relu(self.margin_hard - euclidean(z_patient, z_exc))
            hard_term = hard_term + m_exc * margin_violation ** 2

        # Repulsion from unrelated random negatives' INCLUSION coordinate --
        # generic representation-quality term, smaller margin than hard negs.
        rand_term = torch.zeros((), device=z_patient.device)
        for z_inc in random_negatives:
            margin_violation = F.relu(self.margin_rand - euclidean(z_patient, z_inc))
            rand_term = rand_term + margin_violation ** 2

        return attraction + self.lambda_1 * hard_term + self.lambda_2 * rand_term


def similarity(z_patient: torch.Tensor, z_trial_inc: torch.Tensor,
               z_trial_exc: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    """
    Similarity(P_i, T_j) = cos(z_Pi, z_Tj^inc) - eta * cos(z_Pi, z_Tj^exc)
    """
    # Ensure all tensors are on the same device and properly shaped
    z_patient = z_patient.squeeze()
    z_trial_inc = z_trial_inc.squeeze()
    z_trial_exc = z_trial_exc.squeeze()
    
    inc_sim = F.cosine_similarity(z_patient.unsqueeze(0), z_trial_inc.unsqueeze(0)).item()
    exc_sim = F.cosine_similarity(z_patient.unsqueeze(0), z_trial_exc.unsqueeze(0)).item()
    return inc_sim - eta * exc_sim