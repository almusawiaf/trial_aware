"""
labeling.py
===========
Reproduces the upstream ground-truth rule, vectorised, and exposes it as a
first-class object so its properties can be audited rather than assumed.

The upstream label is

    M_inc = mean_c ( match(patient, c) * w_c )      over inclusion criteria
    M_exc = max_c  ( match(patient, c) * w_c )      over exclusion criteria
    y     = 1  iff  M_inc >= tau_inc  and  M_exc < tau_exc

Two things follow, and both drive the design of this benchmark:

1. `y` is a deterministic function of the model inputs. Any feature that
   exposes per-criterion match statistics reconstructs `y` exactly. See
   `audit_leakage` -- it fits a one-feature logistic model on M_inc and M_exc
   and reports the AUC, which should come out at ~1.0. That number is the
   ceiling any "leaky" model will hit, and it is a property of the label, not
   evidence of clinical skill.

2. `y` is *weak supervision*, not observed enrolment. Every result in this
   benchmark measures agreement with a rule, so headline claims should be
   phrased as "recovers the eligibility rule", never "predicts eligibility".

Two upstream variants exist and disagree. `trial_graph.compute_matching_indices`
(used by evaluate.py) scores an unresolvable criterion as a *failed match*;
`matching_engine.compute_matching_indices` (the COMPOSE-style path) drops it.
Both are implemented; `drop_unresolved` selects between them and the resulting
prevalence shift is logged, because it is large.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .data import CriterionSpec, Dataset, PatientRecord, TrialSpec

log = logging.getLogger(__name__)

_NUMERIC_OPS = {"GT", "GTE", "LT", "LTE", "EQ", "BETWEEN"}


@dataclass
class LabelMatrices:
    """Dense (n_patients, n_trials) label and score matrices."""

    y: np.ndarray            # binary eligibility
    m_inc: np.ndarray        # inclusion index
    m_exc: np.ndarray        # exclusion index
    patient_ids: List[str]
    trial_ids: List[str]
    n_dropped_trials: int = 0

    @property
    def prevalence(self) -> float:
        return float(self.y.mean())

    def stats(self) -> Dict[str, float]:
        per_trial = self.y.sum(axis=0)
        per_patient = self.y.sum(axis=1)
        return {
            "n_patients": float(self.y.shape[0]),
            "n_trials": float(self.y.shape[1]),
            "n_positive_pairs": float(self.y.sum()),
            "prevalence": self.prevalence,
            "trials_with_any_positive": float((per_trial > 0).sum()),
            "trials_all_positive": float((per_trial == self.y.shape[0]).sum()),
            "patients_with_any_positive": float((per_patient > 0).sum()),
            "median_positives_per_trial": float(np.median(per_trial)),
            "max_positives_per_trial": float(per_trial.max()) if per_trial.size else 0.0,
        }


class RuleLabeler:
    """Vectorised implementation of the upstream matching rule."""

    def __init__(
        self,
        inc_threshold: float = 0.15,
        exc_threshold: float = 0.80,
        sigmoid_delta: float = 1.0,
        drop_unresolved: bool = True,
        inclusion_aggregation: str = "mean_weighted",  # upstream trial_graph
        exclusion_aggregation: str = "max",
    ):
        self.inc_threshold = inc_threshold
        self.exc_threshold = exc_threshold
        self.sigmoid_delta = sigmoid_delta
        self.drop_unresolved = drop_unresolved
        self.inclusion_aggregation = inclusion_aggregation
        self.exclusion_aggregation = exclusion_aggregation

        # Populated by `fit`
        self._code_index: Dict[str, int] = {}
        self._lab_index: Dict[str, int] = {}
        self._presence: Optional[sparse.csc_matrix] = None
        self._lab_values: Optional[np.ndarray] = None
        self._lab_mask: Optional[np.ndarray] = None
        self.patient_ids: List[str] = []

    # -- patient side --------------------------------------------------
    def fit(self, patients: Sequence[PatientRecord]) -> "RuleLabeler":
        """Index the patient side once; trials are then scored against it."""
        self.patient_ids = [p.patient_id for p in patients]

        codes: Dict[str, int] = {}
        labs: Dict[str, int] = {}
        for p in patients:
            for c in p.diagnosis_codes:
                codes.setdefault(f"dx:{c}", len(codes))
            for c in p.medication_codes:
                codes.setdefault(f"rx:{c}", len(codes))
            for c in p.lab_values:
                codes.setdefault(f"lab:{c}", len(codes))
                labs.setdefault(str(c), len(labs))

        rows, cols = [], []
        for i, p in enumerate(patients):
            for c in p.diagnosis_codes:
                rows.append(i); cols.append(codes[f"dx:{c}"])
            for c in p.medication_codes:
                rows.append(i); cols.append(codes[f"rx:{c}"])
            for c in p.lab_values:
                rows.append(i); cols.append(codes[f"lab:{c}"])

        n_p, n_c = len(patients), max(len(codes), 1)
        presence = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_p, n_c)
        )
        self._presence = presence.tocsc()
        self._code_index = codes
        self._lab_index = labs

        lab_vals = np.zeros((n_p, max(len(labs), 1)), dtype=np.float32)
        lab_mask = np.zeros_like(lab_vals, dtype=bool)
        for i, p in enumerate(patients):
            for item, v in p.lab_values.items():
                j = labs[str(item)]
                lab_vals[i, j] = float(v)
                lab_mask[i, j] = True
        self._lab_values = lab_vals
        self._lab_mask = lab_mask

        log.info(
            "RuleLabeler indexed %d patients, %d distinct codes, %d lab items",
            n_p, len(codes), len(labs),
        )
        return self

    # -- criterion scoring ---------------------------------------------
    def _criterion_scores(self, c: CriterionSpec) -> Optional[np.ndarray]:
        """Per-patient match score in [0, 1] for one criterion.

        Returns None when the criterion is unresolvable AND `drop_unresolved`
        is set, signalling the caller to exclude it from aggregation.
        """
        if self.drop_unresolved and not c.is_resolved:
            return None

        n_p = len(self.patient_ids)
        prefix = {"diagnosis": "dx", "medication": "rx", "lab": "lab"}.get(
            c.entity_type, "dx"
        )
        key = f"{prefix}:{c.entity_code}"
        col = self._code_index.get(key)
        present = (
            np.asarray(self._presence[:, col].todense()).ravel()
            if col is not None
            else np.zeros(n_p, dtype=np.float32)
        )

        op = c.operator
        if op == "EXISTS":
            return present.astype(np.float32)
        if op == "NOT_EXISTS":
            return (1.0 - present).astype(np.float32)

        # Numeric operators are only meaningful for labs that were measured.
        if op in _NUMERIC_OPS:
            if c.entity_type != "lab" or c.value is None:
                return np.zeros(n_p, dtype=np.float32)
            j = self._lab_index.get(str(c.entity_code))
            if j is None:
                return np.zeros(n_p, dtype=np.float32)
            x = self._lab_values[:, j]
            measured = self._lab_mask[:, j]
            d = self.sigmoid_delta
            if op in ("GT", "GTE"):
                s = _sigmoid(d * (x - c.value))
            elif op in ("LT", "LTE"):
                s = _sigmoid(d * (c.value - x))
            elif op == "EQ":
                s = np.exp(-d * np.abs(x - c.value))
            else:  # BETWEEN
                hi = c.max_value if c.max_value is not None else c.value
                s = _sigmoid(d * (x - c.value)) * _sigmoid(d * (hi - x))
            return (s * measured).astype(np.float32)

        return np.zeros(n_p, dtype=np.float32)

    # -- trial scoring -------------------------------------------------
    def score_trial(self, trial: TrialSpec) -> Tuple[np.ndarray, np.ndarray]:
        n_p = len(self.patient_ids)

        inc_stack, inc_weights = [], []
        for c in trial.inclusion:
            s = self._criterion_scores(c)
            if s is None:
                continue
            inc_stack.append(s * c.severity_weight)
            inc_weights.append(c.severity_weight)

        if inc_stack:
            arr = np.stack(inc_stack, axis=1)
            if self.inclusion_aggregation == "mean_weighted":
                # upstream trial_graph.py: np.mean of weight-multiplied scores
                m_inc = arr.mean(axis=1)
            elif self.inclusion_aggregation == "weighted_average":
                # upstream matching_engine.py: sum(w*s) / sum(w)
                m_inc = arr.sum(axis=1) / max(sum(inc_weights), 1e-8)
            else:
                raise ValueError(self.inclusion_aggregation)
        else:
            # Upstream convention: a trial with no usable inclusion criterion
            # is vacuously satisfied by everybody.
            m_inc = np.ones(n_p, dtype=np.float32)

        exc_stack = []
        for c in trial.exclusion:
            s = self._criterion_scores(c)
            if s is None:
                continue
            exc_stack.append(s * c.severity_weight)

        if exc_stack:
            arr = np.stack(exc_stack, axis=1)
            m_exc = arr.max(axis=1) if self.exclusion_aggregation == "max" else arr.mean(axis=1)
        else:
            m_exc = np.zeros(n_p, dtype=np.float32)

        return m_inc.astype(np.float32), m_exc.astype(np.float32)

    # -- full matrix ---------------------------------------------------
    def build(
        self,
        dataset: Dataset,
        drop_trials_without_inclusion: bool = True,
        drop_degenerate_trials: bool = True,
    ) -> LabelMatrices:
        """Score every (patient, trial) pair.

        `drop_degenerate_trials` removes trials whose label column is constant
        (all-eligible or all-ineligible). Those columns carry zero ranking
        information, break per-trial AUC, and -- because upstream treats an
        empty inclusion list as M_inc = 1.0 -- would otherwise inflate
        prevalence with pairs no model could ever get wrong.
        """
        if self._presence is None:
            self.fit(dataset.patients)

        kept: List[TrialSpec] = []
        inc_cols, exc_cols = [], []
        n_dropped = 0

        for t in dataset.trials:
            usable_inc = [
                c for c in t.inclusion if (not self.drop_unresolved) or c.is_resolved
            ]
            if drop_trials_without_inclusion and not usable_inc:
                n_dropped += 1
                continue
            m_inc, m_exc = self.score_trial(t)
            y_col = ((m_inc >= self.inc_threshold) & (m_exc < self.exc_threshold))
            if drop_degenerate_trials and (y_col.all() or (~y_col).all()):
                n_dropped += 1
                continue
            kept.append(t)
            inc_cols.append(m_inc)
            exc_cols.append(m_exc)

        if not kept:
            raise RuntimeError(
                "Every trial was dropped. Loosen inc_threshold, disable "
                "drop_degenerate_trials, or check criterion-code resolution."
            )

        m_inc = np.stack(inc_cols, axis=1)
        m_exc = np.stack(exc_cols, axis=1)
        y = ((m_inc >= self.inc_threshold) & (m_exc < self.exc_threshold)).astype(np.int8)

        dataset.trials = kept  # keep the dataset and matrices aligned
        out = LabelMatrices(
            y=y,
            m_inc=m_inc,
            m_exc=m_exc,
            patient_ids=list(self.patient_ids),
            trial_ids=[t.trial_id for t in kept],
            n_dropped_trials=n_dropped,
        )
        log.info(
            "Labels built: %d x %d, prevalence=%.4f, dropped %d trials",
            y.shape[0], y.shape[1], out.prevalence, n_dropped,
        )
        return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------
def audit_leakage(labels: LabelMatrices) -> Dict[str, float]:
    """Quantify how completely the rule inputs determine the rule output.

    Fits nothing -- just scores `M_inc - M_exc` directly against `y`. If the
    AUC is ~1.0 (it will be, by construction), then any model handed those two
    statistics is measuring the labeller, not the biology. Printed at the top
    of every run so the ORACLE regime is never mistaken for a result.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = labels.y.ravel()
    score = (labels.m_inc - labels.m_exc).ravel()
    out = {
        "oracle_score_roc_auc": float(roc_auc_score(y, score)),
        "oracle_score_pr_auc": float(average_precision_score(y, score)),
        "prevalence": float(y.mean()),
    }
    # M_inc alone, since exclusion fires rarely.
    out["m_inc_only_roc_auc"] = float(roc_auc_score(y, labels.m_inc.ravel()))
    return out
