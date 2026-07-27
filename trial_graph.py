"""
trial_graph.py

Structured representation of clinical-trial eligibility criteria, and the
patient <-> trial matching-index functions (M_inc, M_exc) from the paper's
"Trial Embedding Space Construction" section.

Scope note: extracting (Entity, Operator, Value) triplets from raw free-text
eligibility criteria is a clinical NER-RE task (e.g. via MedCAT / scispaCy /
a fine-tuned LLM extractor) and is intentionally OUT OF SCOPE of this file.
This module consumes the ALREADY-STRUCTURED output of that step (see
`mock_data.generate_mock_trials` for the expected schema) and focuses on
what the paper's methodology actually specifies mathematically: how those
triplets become matching indices and training pairs.
"""
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np


class Operator(Enum):
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"
    GT = "GT"
    GTE = "GTE"
    LT = "LT"
    LTE = "LTE"
    EQ = "EQ"


@dataclass
class Criterion:
    entity_type: str          # 'diagnosis' | 'medication' | 'lab'
    entity_code: str          # must key into the same node maps used for the patient graph
    operator: Operator
    value: Optional[float]
    is_inclusion: bool
    severity_weight: float = 1.0


@dataclass
class Trial:
    trial_id: str
    criteria: List[Criterion]

    @property
    def inclusion_criteria(self) -> List[Criterion]:
        return [c for c in self.criteria if c.is_inclusion]

    @property
    def exclusion_criteria(self) -> List[Criterion]:
        return [c for c in self.criteria if not c.is_inclusion]


class TrialStore:
    """Loads structured trial-criteria records (JSON or in-memory list of dicts)."""

    def __init__(self, trials: List[Trial]):
        self.trials: Dict[str, Trial] = {t.trial_id: t for t in trials}

    @classmethod
    @classmethod
    def from_records(cls, records: List[dict]) -> "TrialStore":
        trials = []
        for rec in records:
            # Check both trial_id and nct_id
            t_id = rec.get("trial_id") or rec.get("nct_id", f"TRIAL_{len(trials)}")
            
            criteria = [
                Criterion(
                    entity_type=c["entity_type"],
                    entity_code=str(c["entity_code"]),
                    operator=Operator(c["operator"]),
                    value=c.get("value"),
                    is_inclusion=c["is_inclusion"],
                    severity_weight=c.get("severity_weight", 1.0),
                )
                for c in rec["criteria"]
            ]
            trials.append(Trial(trial_id=t_id, criteria=criteria))
        return cls(trials)

    @classmethod
    def from_json(cls, path: str) -> "TrialStore":
        with open(path, "r") as f:
            records = json.load(f)
        return cls.from_records(records)

    def __iter__(self):
        return iter(self.trials.values())

    def __getitem__(self, trial_id: str) -> Trial:
        return self.trials[trial_id]

    def __len__(self):
        return len(self.trials)


class PatientClinicalState:
    """
    A lightweight per-patient snapshot of known diagnoses / medications / last
    normalized lab values, used only to EVALUATE matching indices (M_inc,
    M_exc) against trial criteria -- not part of the learned graph itself.
    """

    def __init__(self, diagnosis_codes: Set[str], medication_codes: Set[str],
                 lab_last_values: Dict[str, float]):
        self.diagnosis_codes = diagnosis_codes
        self.medication_codes = medication_codes
        self.lab_last_values = lab_last_values

    @classmethod
    def build_from_tables(cls, subject_id: int, diag_df, rx_df, labs_df) -> "PatientClinicalState":
        dx = set(diag_df.loc[diag_df.SUBJECT_ID == subject_id, "ICD10_CODE"].astype(str))
        rx = set(rx_df.loc[rx_df.SUBJECT_ID == subject_id, "NDC"].astype(str))
        labs_pt = labs_df[labs_df.SUBJECT_ID == subject_id]
        last_vals = {}
        if len(labs_pt):
            idx = labs_pt.groupby("ITEMID")["CHARTTIME"].idxmax()
            last_rows = labs_pt.loc[idx]
            last_vals = dict(zip(last_rows["ITEMID"].astype(str), last_rows["IMPUTED_VALUE_DECAYED"]))
        return cls(dx, rx, last_vals)


# Sigmoid relaxation scale for continuous (GT/LT/GTE/LTE/EQ) criteria.
# Larger delta -> sharper (closer to a hard boolean threshold).
SIGMOID_DELTA = 2.0


def _match_single(state: PatientClinicalState, c: Criterion) -> float:
    """m(P_i, T_{j,c}) in [0, 1] -- boolean for EXISTS/NOT_EXISTS, sigmoid-relaxed otherwise."""
    if c.entity_type == "diagnosis":
        present = c.entity_code in state.diagnosis_codes
    elif c.entity_type == "medication":
        present = c.entity_code in state.medication_codes
    elif c.entity_type == "lab":
        present = c.entity_code in state.lab_last_values
    else:
        raise ValueError(f"Unknown entity_type: {c.entity_type}")

    if c.operator == Operator.EXISTS:
        return 1.0 if present else 0.0
    if c.operator == Operator.NOT_EXISTS:
        return 1.0 if not present else 0.0

    # Remaining operators only make sense for continuous lab values.
    if c.entity_type != "lab" or not present:
        return 0.0
    x = state.lab_last_values[c.entity_code]
    if c.operator == Operator.GT:
        return float(1 / (1 + np.exp(-SIGMOID_DELTA * (x - c.value))))
    if c.operator == Operator.GTE:
        return float(1 / (1 + np.exp(-SIGMOID_DELTA * (x - c.value) - 1e-6)))
    if c.operator == Operator.LT:
        return float(1 / (1 + np.exp(-SIGMOID_DELTA * (c.value - x))))
    if c.operator == Operator.LTE:
        return float(1 / (1 + np.exp(-SIGMOID_DELTA * (c.value - x) - 1e-6)))
    if c.operator == Operator.EQ:
        return float(np.exp(-SIGMOID_DELTA * abs(x - c.value)))
    raise ValueError(f"Unhandled operator: {c.operator}")


def compute_matching_indices(state: PatientClinicalState, trial: Trial,
                              criteria_subset: Optional[List[Criterion]] = None):
    """
    Returns (M_inc, M_exc) for a (patient, trial) pair.

      M_inc = weighted average of severity-weighted inclusion matches
      M_exc = max over exclusion matches (continuous relaxation of the
              logical OR used in the paper -- see the notation-precision
              fix from the methodology review)

    `criteria_subset`, if given, restricts the *inclusion* criteria used --
    this is what lets `derive_weak_positive_pairs` below train on a
    restricted rule subset while a held-out set of criteria (or an
    independently curated enrollment dataset) is reserved for evaluation,
    avoiding the train/eval circularity flagged in the methodology review.
    """
    inc_criteria = criteria_subset if criteria_subset is not None else trial.inclusion_criteria
    if inc_criteria:
        weights = np.array([c.severity_weight for c in inc_criteria])
        matches = np.array([_match_single(state, c) for c in inc_criteria])
        m_inc = float((weights * matches).sum() / weights.sum())
    else:
        m_inc = 0.0

    exc_criteria = trial.exclusion_criteria
    m_exc = max((_match_single(state, c) for c in exc_criteria), default=0.0)

    return m_inc, m_exc


def derive_weak_positive_pairs(patient_states: Dict[int, PatientClinicalState],
                                trial_store: TrialStore,
                                inc_threshold: float = 0.8,
                                train_criteria_fraction: float = 0.7,
                                seed: int = 0):
    """
    Builds a TRAINING-ONLY weak-supervision pair set: patient P_i is a weak
    positive for trial T_j if M_inc(P_i, T_j) computed on a *random subset*
    of that trial's inclusion criteria exceeds `inc_threshold`, and the
    patient violates no exclusion criterion.

    Held-out criteria are never used here -- a genuine evaluation set
    (P@k, ETE@k against an independently curated / real enrollment
    outcome sample) must be constructed separately and MUST NOT reuse this
    function, or P@k would just measure whether the model recovers the same
    rule it was trained to satisfy.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for trial in trial_store:
        inc = trial.inclusion_criteria
        if not inc:
            continue
        n_train = max(1, int(round(len(inc) * train_criteria_fraction)))
        train_subset = list(rng.choice(inc, size=n_train, replace=False)) if len(inc) > 1 else inc

        for pid, state in patient_states.items():
            m_inc, m_exc = compute_matching_indices(state, trial, criteria_subset=train_subset)
            if m_inc >= inc_threshold and m_exc == 0.0:
                pairs.append((pid, trial.trial_id))

    logging.info(f"[WeakSupervision] Derived {len(pairs)} weak positive (patient, trial) pairs "
                 f"from {len(trial_store)} trials using a {train_criteria_fraction:.0%} criteria subset.")
    return pairs
