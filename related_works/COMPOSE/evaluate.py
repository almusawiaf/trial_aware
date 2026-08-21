"""
evaluate.py -- evaluates the trained COMPOSE baseline using EXACTLY the
same ground-truth definition, thresholds, strict-matching rule, and
bootstrap-CI procedure as models/claude_active/evaluate/compose_based/evaluate.py,
so the numbers in the printed comparison table are directly comparable
to Stage A / Stage B.

Usage:
    cd related_works/COMPOSE
    python train.py      # first, if you haven't already
    python evaluate.py

Writes:
    related_works/COMPOSE/results/compose_evaluation_results_seed{SEED}.json
    related_works/COMPOSE/results/compose_scores_seed{SEED}.npz
"""
import json
import logging
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_utils import (
    load_patient_tables, load_trial_store, get_subject_ids,
    build_patient_states, build_visit_sequences, build_ground_truth_matrices,
)
from dataset import ComposeDataset
from model import ComposeModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def bootstrap_ci_and_pvalue(y_true, scores_compose, scores_other, n_bootstrap=1000, seed=0, other_name="stage_b"):
    """Same patient-level bootstrap procedure as
    models/claude_active/evaluate/compose_based/evaluate.py, reused here
    so COMPOSE-vs-Stage-B confidence intervals are computed identically."""
    rng = np.random.default_rng(seed)
    num_patients = y_true.shape[0]

    auc_c, auc_o, auc_d, pr_c, pr_o, pr_d = [], [], [], [], [], []
    for i in range(n_bootstrap):
        idx = rng.integers(0, num_patients, size=num_patients)
        yt = y_true[idx].ravel()
        sc = scores_compose[idx].ravel()
        so = scores_other[idx].ravel()
        if yt.min() == yt.max():
            continue
        ac, ao = roc_auc_score(yt, sc), roc_auc_score(yt, so)
        pc, po = average_precision_score(yt, sc), average_precision_score(yt, so)
        auc_c.append(ac); auc_o.append(ao); auc_d.append(ao - ac)
        pr_c.append(pc); pr_o.append(po); pr_d.append(po - pc)

    def ci(s):
        return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

    def pvalue(diffs):
        diffs = np.array(diffs)
        sign = np.sign(np.mean(diffs))
        if sign == 0:
            return 1.0
        return float(min(1.0, 2 * np.mean(diffs * sign <= 0)))

    return {
        "auc_compose_ci": ci(auc_c), f"auc_{other_name}_ci": ci(auc_o),
        "auc_diff_ci": ci(auc_d), "auc_pvalue": pvalue(auc_d),
        "pr_compose_ci": ci(pr_c), f"pr_{other_name}_ci": ci(pr_o),
        "pr_diff_ci": ci(pr_d), "pr_pvalue": pvalue(pr_d),
        "n_valid_bootstrap_draws": len(auc_d),
    }


@torch.no_grad()
def compute_patient_memories(model, subject_ids, visit_sequences, vocabs, device, batch_size=64):
    """Encode every patient's EHR memory ONCE (it does not depend on the
    trial/criterion being scored), so scoring P patients x T trials only
    re-runs the (cheap) criterion/query side, not the EHR encoder."""
    ds = ComposeDataset([], visit_sequences, vocabs)  # only used for _encode_visits
    memories = []
    model.eval()
    for start in range(0, len(subject_ids), batch_size):
        batch_ids = subject_ids[start:start + batch_size]
        ehr_ids_b, ehr_mask_b, visit_mask_b = [], [], []
        for sid in batch_ids:
            e_ids, e_mask, v_mask = ds._encode_visits(sid)
            ehr_ids_b.append(e_ids); ehr_mask_b.append(e_mask); visit_mask_b.append(v_mask)
        ehr_ids_b = torch.stack(ehr_ids_b).to(device)
        ehr_mask_b = torch.stack(ehr_mask_b).to(device)
        visit_mask_b = torch.stack(visit_mask_b).to(device)
        demo_b = torch.zeros(len(batch_ids), config.DEMO_DIM, device=device)

        mem = model.encode_ehr(ehr_ids_b, ehr_mask_b, visit_mask_b, demo_b)  # (b, num_slots+1, mem_dim)
        memories.append(mem.cpu())
    return torch.cat(memories, dim=0)  # (P, num_slots+1, mem_dim)


@torch.no_grad()
def score_trial_against_all_patients(model, trial, memories, vocabs, device, eta):
    """Returns (P,) continuous compatibility score = M_inc_pred - eta * M_exc_pred
    for one trial against every patient, using the SAME aggregation formula
    (inc_sim - eta * exc_sim) as models/claude_active's evaluate.py, but with
    M_inc_pred / M_exc_pred coming from the trained classifier's P(match)
    output instead of cosine similarity between embeddings."""
    ds = ComposeDataset([], {}, vocabs)
    P = memories.size(0)
    memories = memories.to(device)

    def aggregate(criteria, empty_value):
        """Weighted mean of P(match) across criteria, all patients at once.
        empty_value: what to return if the trial has no assessable criteria
        of this kind (1.0 for inclusion -- 'nothing to fail' -- 0.0 for
        exclusion -- 'nothing to trigger' -- mirrors trial_graph.compute_matching_indices)."""
        num = torch.zeros(P, device=device)
        den = 0.0
        for c in criteria:
            if c.entity_type == "administrative":
                continue
            crit_ids, crit_mask, crit_value = ds._encode_criterion(c)
            crit_ids = crit_ids.unsqueeze(0).repeat(P, 1).to(device)
            crit_mask = crit_mask.unsqueeze(0).repeat(P, 1).to(device)
            crit_value = crit_value.unsqueeze(0).repeat(P, 1).to(device)
            entity_types = [c.entity_type] * P

            criteria_embd = model.encode_criterion(crit_ids, crit_mask, crit_value, entity_types)
            output, response, query, attention = model.query_network(memories, criteria_embd)
            pred = torch.softmax(output, dim=-1)[:, 0]  # P(match) for this criterion, all patients
            w = float(c.severity_weight)
            num += pred * w
            den += w
        if den == 0:
            return torch.full((P,), empty_value, device=device)
        return num / den

    m_inc = aggregate(trial.inclusion_criteria, empty_value=1.0)
    m_exc = aggregate(trial.exclusion_criteria, empty_value=0.0)

    return (m_inc - eta * m_exc).cpu().numpy()


def strict_accuracy(y_strict, scores, threshold):
    mask = ~np.isnan(y_strict)
    if mask.sum() == 0:
        return None, 0
    y_flat = y_strict[mask]
    pred_flat = (scores[mask] >= threshold).astype(float)
    return float((y_flat == pred_flat).mean()), int(mask.sum())


def load_main_pipeline_results():
    """Best-effort load of Stage A / Stage B numbers from the main
    pipeline's own evaluation_results_seed{SEED}.json, so the printed
    table can show all three models side by side. Returns None if the
    main pipeline hasn't been run/evaluated yet -- that's fine, this
    script still reports COMPOSE's own numbers either way."""
    path = os.path.join(config.MAIN_DATA_DIR, f"evaluation_results_seed{config.SEED}.json")
    if not os.path.exists(path):
        logging.warning(f"Main pipeline results not found at {path} -- "
                         f"printing COMPOSE-only numbers. Run models/claude_active's "
                         f"train.py + evaluate.py with the same RUN_SEED to get the full table.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    config.ensure_directories()
    device = config.DEVICE

    if not os.path.exists(config.CKPT_PATH):
        raise FileNotFoundError(f"No checkpoint at {config.CKPT_PATH}. Run train.py first.")

    # weights_only=False: this checkpoint also stores our own Vocab objects
    # (see vocab.py), not just tensors -- safe here since it's a checkpoint
    # this same codebase wrote, not an untrusted third-party file.
    ckpt = torch.load(config.CKPT_PATH, map_location=device, weights_only=False)
    vocabs = ckpt["vocabs"]

    model = ComposeModel(vocabs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logging.info(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    diag_df, rx_df, labs_df = load_patient_tables()
    trial_store = load_trial_store()
    subject_ids = get_subject_ids(diag_df)
    visit_sequences = build_visit_sequences(subject_ids, diag_df, rx_df, labs_df)
    patient_states = build_patient_states(subject_ids, diag_df, rx_df, labs_df)

    y_true, y_true_strict, trial_ids = build_ground_truth_matrices(
        subject_ids, patient_states, trial_store,
        inc_threshold=0.15, exc_threshold=0.8,
        strict_match_threshold=config.STRICT_MATCH_THRESHOLD,
    )
    logging.info(f"Evaluating {len(subject_ids)} patients x {len(trial_ids)} trials "
                 f"({int(y_true.sum())} positive pairs, {y_true.mean()*100:.2f}%).")

    # 1. Encode every patient's memory once.
    memories = compute_patient_memories(model, subject_ids, visit_sequences, vocabs, device)

    # 2. Score every trial against every patient.
    scores_compose = np.zeros((len(subject_ids), len(trial_ids)))
    for t_idx, tid in enumerate(trial_ids):
        trial = trial_store[tid]
        scores_compose[:, t_idx] = score_trial_against_all_patients(
            model, trial, memories, vocabs, device, eta=config.ETA_EXCLUSION_PENALTY)
        if (t_idx + 1) % 20 == 0:
            logging.info(f"  scored {t_idx + 1}/{len(trial_ids)} trials")

    np.savez(config.SCORES_PATH, scores=scores_compose, y_true=y_true, y_true_strict=y_true_strict,
             subject_ids=np.array(subject_ids), trial_ids=np.array(trial_ids, dtype=object))

    # 3. Metrics on the FULL population (for direct comparison to Stage A/B,
    #    which are also reported on the full population).
    auc_full_pop = roc_auc_score(y_true.ravel(), scores_compose.ravel())
    pr_full_pop = average_precision_score(y_true.ravel(), scores_compose.ravel())

    # 4. Metrics restricted to the held-out TEST patients only (no leakage --
    #    these patients' (patient, trial, criterion) triples were never used
    #    to train COMPOSE). This is the fair number if you want a strictly
    #    supervised-learning-style comparison.
    with open(os.path.join(config.OUT_DIR, f"patient_split_seed{config.SEED}.txt")) as f:
        lines = f.read().splitlines()
    test_ids = set(int(x) for x in lines[2].split(":")[1].split(",") if x)
    test_mask = np.array([sid in test_ids for sid in subject_ids])

    auc_test_only = roc_auc_score(y_true[test_mask].ravel(), scores_compose[test_mask].ravel())
    pr_test_only = average_precision_score(y_true[test_mask].ravel(), scores_compose[test_mask].ravel())

    FIXED_THRESHOLD = 0.0
    acc_full, n_eval_full = strict_accuracy(y_true_strict, scores_compose, FIXED_THRESHOLD)
    acc_test, n_eval_test = strict_accuracy(y_true_strict[test_mask], scores_compose[test_mask], FIXED_THRESHOLD)

    logging.info("=" * 70)
    logging.info("COMPOSE BASELINE -- EVALUATION RESULTS")
    logging.info("=" * 70)
    logging.info(f"[Full population, {len(subject_ids)} patients]  ROC-AUC={auc_full_pop:.4f}  PR-AUC={pr_full_pop:.4f}")
    logging.info(f"[Held-out TEST patients only, {test_mask.sum()} patients, no train leakage]  "
                 f"ROC-AUC={auc_test_only:.4f}  PR-AUC={pr_test_only:.4f}")
    logging.info(f"Strict COMPOSE-style accuracy @ threshold=0.0: full={acc_full}  test-only={acc_test}")
    logging.info("=" * 70)

    # 5. Compare against Stage A / Stage B if available.
    main_results = load_main_pipeline_results()
    comparison = {
        "compose_roc_auc_full_population": float(auc_full_pop),
        "compose_pr_auc_full_population": float(pr_full_pop),
        "compose_roc_auc_test_only": float(auc_test_only),
        "compose_pr_auc_test_only": float(pr_test_only),
        "compose_strict_acc_full": acc_full,
        "compose_strict_acc_test_only": acc_test,
        "num_patients": len(subject_ids),
        "num_test_patients": int(test_mask.sum()),
        "num_trials": len(trial_ids),
        "num_positive_pairs": int(y_true.sum()),
        "seed": config.SEED,
    }

    if main_results is not None:
        comparison["stage_a_roc_auc"] = main_results.get("stage_a_roc_auc")
        comparison["stage_b_roc_auc"] = main_results.get("stage_b_roc_auc")
        comparison["stage_a_pr_auc"] = main_results.get("stage_a_pr_auc")
        comparison["stage_b_pr_auc"] = main_results.get("stage_b_pr_auc")

        # NOTE: this table compares point-estimate ROC-AUC/PR-AUC only. A
        # bootstrap CI comparing COMPOSE directly against Stage A/B would
        # need Stage A/B's raw (patient, trial) SCORE MATRICES, which the
        # main pipeline's evaluate.py does not currently persist to disk
        # (only the scalar metrics in evaluation_results_seed{SEED}.json).
        # See README.md ("Getting a statistically rigorous 3-way
        # comparison") for the one-line change + compare_models.py that
        # adds this if you need it for the paper.

        logging.info("=" * 70)
        logging.info("COMPARISON TABLE (ROC-AUC / PR-AUC)")
        logging.info("=" * 70)
        logging.info(f"{'Model':<28}{'ROC-AUC':>12}{'PR-AUC':>12}")
        logging.info(f"{'Stage A (our, no align)':<28}{main_results.get('stage_a_roc_auc', float('nan')):>12.4f}"
                     f"{main_results.get('stage_a_pr_auc', float('nan')):>12.4f}")
        logging.info(f"{'Stage B (our, full model)':<28}{main_results.get('stage_b_roc_auc', float('nan')):>12.4f}"
                     f"{main_results.get('stage_b_pr_auc', float('nan')):>12.4f}")
        logging.info(f"{'COMPOSE (full population)':<28}{auc_full_pop:>12.4f}{pr_full_pop:>12.4f}")
        logging.info(f"{'COMPOSE (test-only)':<28}{auc_test_only:>12.4f}{pr_test_only:>12.4f}")
        logging.info("=" * 70)
        logging.info("NOTE: Stage A/B are evaluated on the full population by design (unsupervised "
                     "contrastive learning, no train/test split needed). COMPOSE is supervised, so "
                     "its 'test-only' row is the fair, no-leakage comparison; its 'full population' "
                     "row is the one directly matching how Stage A/B's numbers were computed.")
    else:
        logging.info("Run models/claude_active's train.py + evaluate.py with the same RUN_SEED, "
                     "then re-run this script, to get the full 3-way comparison table.")

    with open(config.RESULTS_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    logging.info(f"Results saved to {config.RESULTS_PATH}")


if __name__ == "__main__":
    main()
