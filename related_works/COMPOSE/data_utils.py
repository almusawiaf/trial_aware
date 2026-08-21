"""
data_utils.py -- everything that turns the shared, preprocessed data
(diagnoses_clean.parquet, prescriptions_clean.parquet, labs_clean.parquet,
structured_clinical_trials.json) into what the COMPOSE-adaptation model
needs: per-patient visit sequences, per-criterion training labels, and
train/val/test patient splits.

IMPORTANT: ground-truth matching (used for evaluation, and for deriving
training labels) is delegated to the SAME functions the main pipeline
uses (trial_graph.py / matching_engine.py under models/claude_active).
We import them directly rather than reimplementing the matching logic,
so a COMPOSE-vs-Stage-A/B comparison is judged by one single, consistent
definition of "does this patient satisfy this criterion".
"""
import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
MAIN_MODEL_DIR = os.path.join(REPO_ROOT, "models", "claude_active")
COMPOSE_EVAL_DIR = os.path.join(MAIN_MODEL_DIR, "evaluate", "compose_based")

# Make the main pipeline's modules importable. Order matters: the
# compose_based directory's matching_engine.py (with compute_strict_trial_match)
# must win over the plain models/claude_active copy, so it is inserted
# LAST (== ends up at sys.path[0], searched first).
for p in (MAIN_MODEL_DIR, COMPOSE_EVAL_DIR):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from trial_graph import (  # noqa: E402  (import after sys.path setup, intentional)
    PatientClinicalState,
    TrialStore,
    Trial,
    Criterion,
    Operator,
    compute_matching_indices,
    _match_single,
)
from matching_engine import compute_strict_trial_match  # noqa: E402

from config import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_patient_tables():
    for p in (config.DIAG_PATH, config.RX_PATH, config.LABS_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. This baseline reads the SAME processed tables as "
                f"the main pipeline -- run models/claude_active's run.py (or the "
                f"main data_pipeline) first, or point config.MAIN_DATA_DIR at the "
                f"directory that already contains them."
            )
    diag_df = pd.read_parquet(config.DIAG_PATH)
    rx_df = pd.read_parquet(config.RX_PATH)
    labs_df = pd.read_parquet(config.LABS_PATH)
    return diag_df, rx_df, labs_df


def load_trial_store() -> TrialStore:
    if not os.path.exists(config.TRAIN_TRIALS_PATH):
        raise FileNotFoundError(
            f"Missing {config.TRAIN_TRIALS_PATH}. Run the main pipeline's trial "
            f"structuring step first (see models/claude_active/trial_graph.py)."
        )
    with open(config.TRAIN_TRIALS_PATH, "r") as f:
        records = json.load(f)
    return TrialStore.from_records(records)


def get_subject_ids(diag_df: pd.DataFrame):
    return sorted(diag_df["SUBJECT_ID"].unique(), key=int)


def build_patient_states(subject_ids, diag_df, rx_df, labs_df):
    """Same call the main evaluate.py makes -- identical patient states."""
    return {
        sid: PatientClinicalState.build_from_tables(sid, diag_df, rx_df, labs_df)
        for sid in subject_ids
    }


# ----------------------------------------------------------------------
# Visit sequences (COMPOSE's EHR memory network reads a SEQUENCE of visits)
# ----------------------------------------------------------------------
def build_visit_sequences(subject_ids, diag_df, rx_df, labs_df):
    """
    Returns: dict[sid] -> list of visits, each visit a dict:
        {'diagnosis': [codes...], 'medication': [codes...], 'lab': [codes...]}

    NOTE ON DEVIATION FROM ORIGINAL COMPOSE:
    * Visit order is HADM_ID ascending, used as a chronological proxy --
      the shared parquet tables retain HADM_ID but not admission
      timestamps, so exact chronology is not recoverable here.
    * Lab codes have no HADM_ID in the shared pipeline's labs_clean.parquet
      (labs are stored as a decayed daily trajectory, not tied to a single
      admission -- see data_pipeline/preprocessor.py). We therefore treat
      the patient's lab profile as static across all of their visits,
      exactly as PatientClinicalState does for the rest of the pipeline
      (a single "last known labs" snapshot, not a per-visit one).
    * Demographics (age/gender) are not present in these three tables in
      the shared pipeline, so the demographic vector is all-zero. This
      matches the fact that models/claude_active's own graph features
      (PATIENT_FEAT_DIM) do not encode explicit demographics either.
    """
    diag_col = "ICD10_CODE" if "ICD10_CODE" in diag_df.columns else "ICD9_CODE"
    med_col = "NDC" if "NDC" in rx_df.columns else None
    lab_col = "ITEMID" if "ITEMID" in labs_df.columns else None

    diag_by_sid = {sid: g for sid, g in diag_df.groupby("SUBJECT_ID")}
    rx_by_sid = {sid: g for sid, g in rx_df.groupby("SUBJECT_ID")} if med_col else {}
    labs_by_sid = {sid: g for sid, g in labs_df.groupby("SUBJECT_ID")} if lab_col else {}

    sequences = {}
    for sid in subject_ids:
        # union of HADM_IDs seen in diagnosis/medication tables for this patient
        hadm_ids = set()
        pdiag = diag_by_sid.get(sid)
        if pdiag is not None and "HADM_ID" in pdiag.columns:
            hadm_ids.update(pdiag["HADM_ID"].unique().tolist())
        prx = rx_by_sid.get(sid)
        if prx is not None and "HADM_ID" in prx.columns:
            hadm_ids.update(prx["HADM_ID"].unique().tolist())

        hadm_ids = sorted(hadm_ids)[: config.MAX_VISITS] if hadm_ids else [None]

        # static, patient-level lab profile (see docstring above)
        lab_codes = []
        plabs = labs_by_sid.get(sid)
        if plabs is not None:
            lab_codes = plabs[lab_col].astype(str).unique().tolist()[: config.MAX_CODES_PER_CATEGORY]

        visits = []
        for hadm in hadm_ids:
            diag_codes, med_codes = [], []
            if pdiag is not None:
                sub = pdiag if hadm is None else pdiag[pdiag["HADM_ID"] == hadm]
                diag_codes = sub[diag_col].astype(str).unique().tolist()[: config.MAX_CODES_PER_CATEGORY]
            if prx is not None and med_col:
                sub = prx if hadm is None else prx[prx["HADM_ID"] == hadm]
                med_codes = sub[med_col].astype(str).unique().tolist()[: config.MAX_CODES_PER_CATEGORY]

            visits.append({"diagnosis": diag_codes, "medication": med_codes, "lab": lab_codes})

        if not visits:
            visits = [{"diagnosis": [], "medication": [], "lab": []}]

        sequences[sid] = visits

    return sequences


# ----------------------------------------------------------------------
# Per-criterion training labels
# ----------------------------------------------------------------------
def label_for_criterion(state: PatientClinicalState, criterion: Criterion, rng: random.Random) -> int:
    """
    0 = match, 1 = mismatch, 2 = unknown -- SAME label convention as
    original COMPOSE (see model.py's get_loss: similarity_label built
    from these same three classes).

    The underlying per-criterion score comes from trial_graph._match_single,
    i.e. the exact function the main pipeline itself uses to build M_inc/M_exc.
    """
    score = _match_single(state, criterion)
    if score >= config.MATCH_LABEL_THRESHOLD:
        base_label = 0
    elif score <= config.MISMATCH_LABEL_THRESHOLD:
        base_label = 1
    else:
        base_label = 2

    if base_label != 2 and rng.random() < config.UNKNOWN_RESAMPLE_RATIO * 0.15:
        return 2
    return base_label


def build_training_triples(subject_ids, patient_states, trial_store, rng: random.Random,
                            n_pos_trials_per_patient=3, n_neg_trials_per_patient=3):
    """
    Build (subject_id, trial_id, criterion, label) training quadruples.

    We do not enumerate the full patient x trial x criterion cube (that is
    only needed once, at evaluation time, over the whole population) --
    for TRAINING we sample a handful of trials per patient: some with
    non-trivial inclusion overlap ("weak positives", same notion the main
    pipeline uses for Stage B) and some random ("weak negatives"), then
    take every criterion of the sampled trials as one training example.
    """
    trial_ids = list(trial_store.keys())
    triples = []

    for sid in subject_ids:
        state = patient_states[sid]

        scored = []
        for tid in trial_ids:
            trial = trial_store[tid]
            m_inc, _ = compute_matching_indices(state, trial)
            scored.append((tid, m_inc))
        scored.sort(key=lambda x: x[1], reverse=True)

        pos_trials = [tid for tid, _ in scored[:n_pos_trials_per_patient]]
        remaining = [tid for tid, _ in scored[n_pos_trials_per_patient:]]
        neg_trials = rng.sample(remaining, k=min(n_neg_trials_per_patient, len(remaining))) if remaining else []

        for tid in pos_trials + neg_trials:
            trial = trial_store[tid]
            for c in list(trial.inclusion_criteria) + list(trial.exclusion_criteria):
                if c.entity_type == "administrative":
                    continue
                label = label_for_criterion(state, c, rng)
                triples.append((sid, tid, c, label))

    logging.info(f"[Data] Built {len(triples)} (patient, trial, criterion) training triples "
                 f"from {len(subject_ids)} patients.")
    return triples


def split_patients(subject_ids, seed):
    rng = random.Random(seed)
    ids = list(subject_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    logging.info(f"[Split] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    return train_ids, val_ids, test_ids


# ----------------------------------------------------------------------
# Evaluation ground truth -- identical protocol to models/claude_active's
# evaluate/compose_based/evaluate.py (build_evaluation_matrices), so that
# COMPOSE's numbers land in the exact same (patient, trial) matrix shape
# as Stage A / Stage B.
# ----------------------------------------------------------------------
def build_ground_truth_matrices(subject_ids, patient_states, trial_store,
                                 inc_threshold, exc_threshold, strict_match_threshold):
    trial_ids = list(trial_store.keys())
    num_patients, num_trials = len(subject_ids), len(trial_ids)

    y_true = np.zeros((num_patients, num_trials))
    y_true_strict = np.full((num_patients, num_trials), np.nan)

    for p_idx, sid in enumerate(subject_ids):
        state = patient_states[sid]
        for t_idx, tid in enumerate(trial_ids):
            trial = trial_store[tid]
            m_inc, m_exc = compute_matching_indices(state, trial)
            y_true[p_idx, t_idx] = 1.0 if (m_inc >= inc_threshold and m_exc < exc_threshold) else 0.0

            strict_result = compute_strict_trial_match(state, trial, hierarchy=None,
                                                         match_threshold=strict_match_threshold)
            if strict_result is not None:
                y_true_strict[p_idx, t_idx] = 1.0 if strict_result else 0.0

    return y_true, y_true_strict, trial_ids
