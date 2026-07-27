# """
# trial_graph.py

# Structured representation of clinical-trial eligibility criteria, and the
# patient <-> trial matching-index functions (M_inc, M_exc) from the paper's
# "Trial Embedding Space Construction" section.

# Scope note: extracting (Entity, Operator, Value) triplets from raw free-text
# eligibility criteria is a clinical NER-RE task (e.g. via MedCAT / scispaCy /
# a fine-tuned LLM extractor) and is intentionally OUT OF SCOPE of this file.
# This module consumes the ALREADY-STRUCTURED output of that step (see
# `mock_data.generate_mock_trials` for the expected schema) and focuses on
# what the paper's methodology actually specifies mathematically: how those
# triplets become matching indices and training pairs.
# """
# import json
# import logging
# from dataclasses import dataclass
# from enum import Enum
# from typing import Dict, List, Optional, Set

# import numpy as np


# class Operator(Enum):
#     EXISTS = "EXISTS"
#     NOT_EXISTS = "NOT_EXISTS"
#     GT = "GT"
#     GTE = "GTE"
#     LT = "LT"
#     LTE = "LTE"
#     EQ = "EQ"


# @dataclass
# class Criterion:
#     entity_type: str          # 'diagnosis' | 'medication' | 'lab'
#     entity_code: str          # must key into the same node maps used for the patient graph
#     operator: Operator
#     value: Optional[float]
#     is_inclusion: bool
#     severity_weight: float = 1.0


# @dataclass
# class Trial:
#     trial_id: str
#     criteria: List[Criterion]

#     @property
#     def inclusion_criteria(self) -> List[Criterion]:
#         return [c for c in self.criteria if c.is_inclusion]

#     @property
#     def exclusion_criteria(self) -> List[Criterion]:
#         return [c for c in self.criteria if not c.is_inclusion]


# class TrialStore:
#     """Loads structured trial-criteria records (JSON or in-memory list of dicts)."""

#     def __init__(self, trials: List[Trial]):
#         self.trials: Dict[str, Trial] = {t.trial_id: t for t in trials}

#     @classmethod
#     @classmethod
#     def from_records(cls, records: List[dict]) -> "TrialStore":
#         trials = []
#         for rec in records:
#             # Check both trial_id and nct_id
#             t_id = rec.get("trial_id") or rec.get("nct_id", f"TRIAL_{len(trials)}")
            
#             criteria = [
#                 Criterion(
#                     entity_type=c["entity_type"],
#                     entity_code=str(c["entity_code"]),
#                     operator=Operator(c["operator"]),
#                     value=c.get("value"),
#                     is_inclusion=c["is_inclusion"],
#                     severity_weight=c.get("severity_weight", 1.0),
#                 )
#                 for c in rec["criteria"]
#             ]
#             trials.append(Trial(trial_id=t_id, criteria=criteria))
#         return cls(trials)

#     @classmethod
#     def from_json(cls, path: str) -> "TrialStore":
#         with open(path, "r") as f:
#             records = json.load(f)
#         return cls.from_records(records)

#     def __iter__(self):
#         return iter(self.trials.values())

#     def __getitem__(self, trial_id: str) -> Trial:
#         return self.trials[trial_id]

#     def __len__(self):
#         return len(self.trials)


# class PatientClinicalState:
#     """
#     A lightweight per-patient snapshot of known diagnoses / medications / last
#     normalized lab values, used only to EVALUATE matching indices (M_inc,
#     M_exc) against trial criteria -- not part of the learned graph itself.
#     """

#     def __init__(self, diagnosis_codes: Set[str], medication_codes: Set[str],
#                  lab_last_values: Dict[str, float]):
#         self.diagnosis_codes = diagnosis_codes
#         self.medication_codes = medication_codes
#         self.lab_last_values = lab_last_values

#     @classmethod
#     def build_from_tables(cls, subject_id: int, diag_df, rx_df, labs_df) -> "PatientClinicalState":
#         dx = set(diag_df.loc[diag_df.SUBJECT_ID == subject_id, "ICD10_CODE"].astype(str))
#         rx = set(rx_df.loc[rx_df.SUBJECT_ID == subject_id, "NDC"].astype(str))
#         labs_pt = labs_df[labs_df.SUBJECT_ID == subject_id]
#         last_vals = {}
#         if len(labs_pt):
#             idx = labs_pt.groupby("ITEMID")["CHARTTIME"].idxmax()
#             last_rows = labs_pt.loc[idx]
#             last_vals = dict(zip(last_rows["ITEMID"].astype(str), last_rows["IMPUTED_VALUE_DECAYED"]))
#         return cls(dx, rx, last_vals)


# # Sigmoid relaxation scale for continuous (GT/LT/GTE/LTE/EQ) criteria.
# # Larger delta -> sharper (closer to a hard boolean threshold).
# SIGMOID_DELTA = 2.0


# def _match_single(state: PatientClinicalState, c: Criterion) -> float:
#     """m(P_i, T_{j,c}) in [0, 1] -- boolean for EXISTS/NOT_EXISTS, sigmoid-relaxed otherwise."""
#     if c.entity_type == "diagnosis":
#         present = c.entity_code in state.diagnosis_codes
#     elif c.entity_type == "medication":
#         present = c.entity_code in state.medication_codes
#     elif c.entity_type == "lab":
#         present = c.entity_code in state.lab_last_values
#     elif c.entity_type == "procedure":
#         # Procedures are not currently extracted into PatientClinicalState from MIMIC-III
#         present = False
#     else:
#         # Gracefully handle any other unexpected entity_type without breaking Stage B execution
#         present = False

#     if c.operator == Operator.EXISTS:
#         return 1.0 if present else 0.0
#     if c.operator == Operator.NOT_EXISTS:
#         return 1.0 if not present else 0.0

#     # Remaining operators only make sense for continuous lab values.
#     if c.entity_type != "lab" or not present:
#         return 0.0
#     x = state.lab_last_values[c.entity_code]
#     if c.operator == Operator.GT:
#         return float(1 / (1 + np.exp(-SIGMOID_DELTA * (x - c.value))))
#     if c.operator == Operator.GTE:
#         return float(1 / (1 + np.exp(-SIGMOID_DELTA * (x - c.value) - 1e-6)))
#     if c.operator == Operator.LT:
#         return float(1 / (1 + np.exp(-SIGMOID_DELTA * (c.value - x))))
#     if c.operator == Operator.LTE:
#         return float(1 / (1 + np.exp(-SIGMOID_DELTA * (c.value - x) - 1e-6)))
#     if c.operator == Operator.EQ:
#         return float(np.exp(-SIGMOID_DELTA * abs(x - c.value)))
#     raise ValueError(f"Unhandled operator: {c.operator}")

# def compute_matching_indices(state: PatientClinicalState, trial: Trial,
#                               criteria_subset: Optional[List[Criterion]] = None):
#     """
#     Returns (M_inc, M_exc) for a (patient, trial) pair.

#       M_inc = weighted average of severity-weighted inclusion matches
#       M_exc = max over exclusion matches (continuous relaxation of the
#               logical OR used in the paper -- see the notation-precision
#               fix from the methodology review)

#     `criteria_subset`, if given, restricts the *inclusion* criteria used --
#     this is what lets `derive_weak_positive_pairs` below train on a
#     restricted rule subset while a held-out set of criteria (or an
#     independently curated enrollment dataset) is reserved for evaluation,
#     avoiding the train/eval circularity flagged in the methodology review.
#     """
#     inc_criteria = criteria_subset if criteria_subset is not None else trial.inclusion_criteria
#     if inc_criteria:
#         weights = np.array([c.severity_weight for c in inc_criteria])
#         matches = np.array([_match_single(state, c) for c in inc_criteria])
#         m_inc = float((weights * matches).sum() / weights.sum())
#     else:
#         m_inc = 0.0

#     exc_criteria = trial.exclusion_criteria
#     m_exc = max((_match_single(state, c) for c in exc_criteria), default=0.0)

#     return m_inc, m_exc


# def derive_weak_positive_pairs(
#     patient_states: Dict[int, PatientClinicalState],
#     trial_store: TrialStore,
#     inc_threshold: float = 0.01,
#     train_criteria_fraction: float = 0.7,
#     seed: int = 0
# ):
#     weak_pairs = []
    
#     # 1. Try standard matching loop
#     for pid, state in patient_states.items():
#         for tid, trial in trial_store.trials.items():
#             m_inc, _ = compute_matching_indices(state, trial)
#             if m_inc >= inc_threshold:
#                 weak_pairs.append((pid, tid))

#     # 2. FALLBACK: If code vocabulary mismatch results in 0 pairs,
#     # generate round-robin weak positive pairs so Stage B alignment fine-tuning can run!
#     if not weak_pairs:
#         logging.warning(
#             "[WeakSupervision] 0 pairs derived due to ICD-9 vs ICD-10 code mismatch! "
#             "Applying synthetic round-robin pair assignment to force Stage B execution."
#         )
#         trial_ids = list(trial_store.trials.keys())
#         patient_indices = list(range(len(patient_states)))
        
#         for idx, p_idx in enumerate(patient_indices):
#             # Assign each patient to a trial in round-robin fashion
#             assigned_trial = trial_ids[idx % len(trial_ids)]
#             weak_pairs.append((p_idx, assigned_trial))

#     logging.info(f"[WeakSupervision] Derived {len(weak_pairs)} weak positive (patient, trial) pairs.")
#     return weak_pairs

# trial_graph.py
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SIGMOID_DELTA = 1.0


class Operator(Enum):
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"
    GT = "GT"
    GTE = "GTE"
    LT = "LT"
    LTE = "LTE"
    EQ = "EQ"
    BETWEEN = "BETWEEN"


@dataclass
class Criterion:
    entity_type: str
    entity_code: str
    operator: Operator
    value: Optional[float] = None
    max_value: Optional[float] = None
    weight: float = 1.0


@dataclass
class Trial:
    trial_id: str
    inclusion_criteria: List[Criterion] = field(default_factory=list)
    exclusion_criteria: List[Criterion] = field(default_factory=list)


class PatientClinicalState:

    def __init__(self, subject_id: int):
        self.subject_id = subject_id
        self.diagnosis_codes = set()
        self.medication_codes = set()
        self.lab_last_values = {}

    @classmethod
    def build_from_tables(cls, subject_id: int, diag_df, rx_df, labs_df):
        state = cls(subject_id)
        p_diag = diag_df[diag_df['SUBJECT_ID'] == subject_id]
        state.diagnosis_codes = set(p_diag['ICD10_CODE'].astype(str).unique())

        p_rx = rx_df[rx_df['SUBJECT_ID'] == subject_id]
        state.medication_codes = set(p_rx['NDC'].astype(str).unique())

        p_labs = labs_df[labs_df['SUBJECT_ID'] == subject_id]
        for _, row in p_labs.iterrows():
            state.lab_last_values[str(row['ITEMID'])] = float(row['VALUENUM'])
        return state


def _match_single(state: PatientClinicalState, c: Criterion) -> float:
    if c.entity_type == "diagnosis":
        present = c.entity_code in state.diagnosis_codes
    elif c.entity_type == "medication":
        present = c.entity_code in state.medication_codes
    elif c.entity_type == "lab":
        present = c.entity_code in state.lab_last_values
    else:
        present = False

    if c.operator == Operator.EXISTS:
        return 1.0 if present else 0.0
    if c.operator == Operator.NOT_EXISTS:
        return 1.0 if not present else 0.0

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
    if c.operator == Operator.BETWEEN:
        lo, hi = c.value, c.max_value
        if lo is None or hi is None:
            return 0.0
        inside = (1 / (1 + np.exp(-SIGMOID_DELTA * (x - lo)))) * (1 / (1 + np.exp(-SIGMOID_DELTA * (hi - x))))
        return float(inside)
    
    return 0.0


def compute_matching_indices(state: PatientClinicalState, trial: Trial) -> Tuple[float, float]:
    inc_scores = [_match_single(state, c) * c.weight for c in trial.inclusion_criteria]
    m_inc = float(np.mean(inc_scores)) if inc_scores else 1.0

    exc_scores = [_match_single(state, c) * c.weight for c in trial.exclusion_criteria]
    m_exc = float(np.mean(exc_scores)) if exc_scores else 0.0

    return m_inc, m_exc


class TrialStore:

    def __init__(self):
        self.trials: Dict[str, Trial] = {}

    def add_trial(self, trial: Trial):
        self.trials[trial.trial_id] = trial

    def __getitem__(self, tid: str) -> Trial:
        return self.trials[tid]

    def __iter__(self):
        return iter(self.trials.values())

    @classmethod
    def from_records(cls, records: List[dict]) -> "TrialStore":
        store = cls()
        for rec in records:
            tid = rec["nct_id"]
            inc_list, exc_list = [], []
            for c in rec.get("criteria", []):
                crit = Criterion(
                    entity_type=c["entity_type"],
                    entity_code=str(c["entity_code"]),
                    operator=Operator(c["operator"]),
                    value=c.get("value"),
                    max_value=c.get("max_value"),
                    weight=c.get("severity_weight", 1.0),
                )
                if c.get("is_inclusion", True):
                    inc_list.append(crit)
                else:
                    exc_list.append(crit)
            store.add_trial(Trial(trial_id=tid, inclusion_criteria=inc_list, exclusion_criteria=exc_list))
        return store


def derive_weak_positive_pairs(
    patient_states: Dict[int, PatientClinicalState],
    trial_store: TrialStore,
    inc_threshold: float = 0.01,
) -> List[Tuple[int, str]]:
    weak_pairs = []
    p_keys = sorted(patient_states.keys())
    
    for pid_idx, sid in enumerate(p_keys):
        state = patient_states[sid]
        for tid, trial in trial_store.trials.items():
            m_inc, _ = compute_matching_indices(state, trial)
            if m_inc >= inc_threshold:
                weak_pairs.append((pid_idx, tid))

    logging.info(f"[WeakSupervision] Derived {len(weak_pairs)} weak positive (patient, trial) pairs.")
    return weak_pairs