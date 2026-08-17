# evaluate.py
import json
import logging
import os
import sys
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# config.py and matching_engine.py live right next to this file and must
# win any name collision (compose_based's versions add
# STRICT_MATCH_THRESHOLD / compute_strict_trial_match that the
# models/claude_active copies don't have) -- Python already searches this
# script's own directory first automatically, so we only need to APPEND
# the parent directory as a fallback for trial_graph.py, which doesn't
# have a local copy here. Do not insert(0, ...): that would shadow the
# local config.py/matching_engine.py with the wrong (base) versions.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from trial_graph import PatientClinicalState, TrialStore, compute_matching_indices
from matching_engine import compute_strict_trial_match


def bootstrap_ci_and_pvalue(y_true, scores_base, scores_full, n_bootstrap=1000, seed=0):
    """
    Answers two questions a single number can't:
      1. How much would this metric wobble if we'd sampled a different set
         of patients from the same population? (confidence interval)
      2. Is 'full model beats baseline' a real, reliable effect, or could it
         just as easily have gone the other way by chance? (p-value)

    Resamples at the PATIENT level (not flattened pair level) with
    replacement -- each bootstrap draw keeps every trial-column for the
    patients it picks, preserving the fact that one patient's scores across
    550 trials are correlated with each other, not independent data points.
    This matters: bootstrapping the flattened array would understate the
    true uncertainty and make everything look more significant than it is.
    """
    rng = np.random.default_rng(seed)
    num_patients = y_true.shape[0]

    auc_base_samples, auc_full_samples, auc_diff_samples = [], [], []
    pr_base_samples, pr_full_samples, pr_diff_samples = [], [], []

    for i in range(n_bootstrap):
        if i > 0 and i % 50 == 0:
            logging.info(f"[Bootstrap] {i}/{n_bootstrap} resamples done...")

        idx = rng.integers(0, num_patients, size=num_patients)
        yt = y_true[idx].ravel()
        sb = scores_base[idx].ravel()
        sf = scores_full[idx].ravel()

        # A resample with zero positives (or zero negatives) makes AUC
        # undefined -- skip that draw rather than crash or silently distort results.
        if yt.min() == yt.max():
            continue

        ab = roc_auc_score(yt, sb)
        af = roc_auc_score(yt, sf)
        pb = average_precision_score(yt, sb)
        pf = average_precision_score(yt, sf)

        auc_base_samples.append(ab)
        auc_full_samples.append(af)
        auc_diff_samples.append(af - ab)
        pr_base_samples.append(pb)
        pr_full_samples.append(pf)
        pr_diff_samples.append(pf - pb)

    def ci(samples):
        return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

    def two_sided_pvalue(diff_samples):
        # Fraction of bootstrap draws where the sign disagrees with the
        # observed direction of the effect -- doubled for a two-sided test.
        diffs = np.array(diff_samples)
        observed_sign = np.sign(np.mean(diffs))
        if observed_sign == 0:
            return 1.0
        p_one_sided = np.mean(diffs * observed_sign <= 0)
        return float(min(1.0, 2 * p_one_sided))

    return {
        'auc_base_ci': ci(auc_base_samples),
        'auc_full_ci': ci(auc_full_samples),
        'auc_diff_ci': ci(auc_diff_samples),
        'auc_pvalue': two_sided_pvalue(auc_diff_samples),
        'pr_base_ci': ci(pr_base_samples),
        'pr_full_ci': ci(pr_full_samples),
        'pr_diff_ci': ci(pr_diff_samples),
        'pr_pvalue': two_sided_pvalue(pr_diff_samples),
        'n_valid_bootstrap_draws': len(auc_diff_samples),
    }

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_evaluation_matrices(cfg: Config):
    """
    Build ground truth and prediction matrices for evaluation.
    Returns:
        y_true: (num_patients, num_trials) SOFT binary matrix (m_inc/m_exc
            threshold-based eligibility, the original definition)
        scores_baseline: (num_patients, num_trials) similarity scores from Stage A
        scores_full: (num_patients, num_trials) similarity scores from Stage B
        y_true_strict: (num_patients, num_trials) STRICT COMPOSE-style
            all-or-nothing match matrix. NaN where a trial had zero
            assessable criteria for that patient (excluded from accuracy,
            not counted as a match or a miss). See
            matching_engine.compute_strict_trial_match for the exact rule.
    """
    
    # ============================================================
    # 1. Load trials from training file (since embeddings contain these)
    # ============================================================
    
    train_trials_path = cfg.TRAIN_TRIALS_PATH
    
    if os.path.exists(train_trials_path):
        logging.info(f"Loading trials from: {train_trials_path}")
        with open(train_trials_path, "r") as f:
            eval_trials_data = json.load(f)
        logging.info(f"Loaded {len(eval_trials_data)} trials for evaluation")
        trial_store = TrialStore.from_records(eval_trials_data)
    else:
        logging.error(f"Trials not found at {train_trials_path}")
        return None, None, None, None

    # 2. Load patient data tables
    diag_path = os.path.join(cfg.OUTPUT_DIR, "diagnoses_clean.parquet")
    rx_path = os.path.join(cfg.OUTPUT_DIR, "prescriptions_clean.parquet")
    labs_path = os.path.join(cfg.OUTPUT_DIR, "labs_clean.parquet")
    
    if not all(os.path.exists(p) for p in [diag_path, rx_path, labs_path]):
        logging.error(f"Missing patient data files in {cfg.OUTPUT_DIR}")
        return None, None, None, None
    
    diag_df = pd.read_parquet(diag_path)
    rx_df = pd.read_parquet(rx_path)
    labs_df = pd.read_parquet(labs_path)

    # 3. Get sorted subject IDs and build patient clinical states
    subject_ids = sorted(diag_df['SUBJECT_ID'].unique(), key=int)
    logging.info(f"Building clinical states for {len(subject_ids)} patients...")
    
    patient_states = {
        sid: PatientClinicalState.build_from_tables(sid, diag_df, rx_df, labs_df)
        for sid in subject_ids
    }

    # 4. Load patient and trial embeddings
    baseline_path = cfg.BASELINE_EMBED_PATH
    
    if not os.path.exists(baseline_path):
        logging.error(f"Baseline embeddings not found at {baseline_path}")
        return None, None, None, None
    
    if not os.path.exists(cfg.PATIENT_EMBED_PATH):
        logging.error(f"Patient embeddings not found at {cfg.PATIENT_EMBED_PATH}")
        return None, None, None, None
    
    if not os.path.exists(cfg.TRIAL_EMBED_PATH):
        logging.error(f"Trial embeddings not found at {cfg.TRIAL_EMBED_PATH}")
        return None, None, None, None

    if not os.path.exists(cfg.TRIAL_EMBED_BASELINE_PATH):
        logging.error(
            f"Baseline trial embeddings not found at {cfg.TRIAL_EMBED_BASELINE_PATH}. "
            "Re-run train.py (it now saves this file) before evaluating."
        )
        return None, None, None, None

    h_baseline = torch.load(baseline_path, map_location='cpu')
    h_full = torch.load(cfg.PATIENT_EMBED_PATH, map_location='cpu')
    trial_embeds = torch.load(cfg.TRIAL_EMBED_PATH, map_location='cpu')
    # NEW: separate, honest "before training" trial embeddings -- paired with
    # h_baseline in the same pre-alignment embedding space. Do NOT reuse
    # `trial_embeds` (Stage B) for the baseline score; that was the bug.
    trial_embeds_baseline = torch.load(cfg.TRIAL_EMBED_BASELINE_PATH, map_location='cpu')
    
    # 5. Debug: Print trial embedding information (baseline vs full model)
    for tid, (z_inc, z_exc) in trial_embeds.items():
        z_inc_b, z_exc_b = trial_embeds_baseline.get(tid, (None, None))
        logging.info(f"Trial {tid}:")
        logging.info(f"  [Stage B] z_inc norm: {z_inc.norm().item():.4f}  z_exc norm: {z_exc.norm().item():.4f}  "
                     f"cos(z_inc,z_exc): {F.cosine_similarity(z_inc, z_exc, dim=0).item():.4f}")
        if z_inc_b is not None:
            logging.info(f"  [Baseline] z_inc norm: {z_inc_b.norm().item():.4f}  z_exc norm: {z_exc_b.norm().item():.4f}  "
                         f"cos(z_inc,z_exc): {F.cosine_similarity(z_inc_b, z_exc_b, dim=0).item():.4f}")

    # 6. Filter to trials that have BOTH baseline and full embeddings
    trial_ids = [
        tid for tid in trial_store.trials.keys()
        if tid in trial_embeds and tid in trial_embeds_baseline
    ]
    
    if not trial_ids:
        logging.error("No matching trial IDs found between trial store and embeddings")
        return None, None, None, None
    
    logging.info(f"Evaluating {len(subject_ids)} patients against {len(trial_ids)} clinical trials...")

    # 7. Build ground truth eligibility matrix (y_true)
    num_patients = len(subject_ids)
    num_trials = len(trial_ids)
    
    y_true = np.zeros((num_patients, num_trials))
    m_inc_matrix = np.zeros((num_patients, num_trials))
    m_exc_matrix = np.zeros((num_patients, num_trials))
    # NEW: strict COMPOSE-style trial-level match matrix. NaN means "this
    # trial had zero assessable criteria for this patient" -- excluded from
    # accuracy computation later, not silently treated as a match or a miss.
    y_true_strict = np.full((num_patients, num_trials), np.nan)

    inc_threshold = getattr(cfg, 'HARD_NEG_INC_THRESHOLD', 0.3)
    exc_threshold = getattr(cfg, 'HARD_NEG_EXC_THRESHOLD', 0.3)
    strict_match_threshold = getattr(cfg, 'STRICT_MATCH_THRESHOLD', 0.5)

    logging.info(f"Using thresholds: M_inc >= {inc_threshold}, M_exc < {exc_threshold}")
    logging.info(f"Using STRICT per-criterion match threshold: {strict_match_threshold} "
                 f"(COMPOSE-style all-or-nothing trial matching)")

    for p_idx, sid in enumerate(subject_ids):
        state = patient_states[sid]
        for t_idx, tid in enumerate(trial_ids):
            trial = trial_store[tid]
            m_inc, m_exc = compute_matching_indices(state, trial)
            
            m_inc_matrix[p_idx, t_idx] = m_inc
            m_exc_matrix[p_idx, t_idx] = m_exc
            
            # Ground truth: eligible if inclusion score is high AND exclusion score is low
            y_true[p_idx, t_idx] = 1.0 if (m_inc >= inc_threshold and m_exc < exc_threshold) else 0.0

            # NEW: strict all-or-nothing match, computed independently of
            # the soft m_inc/m_exc thresholds above -- see
            # compute_strict_trial_match's docstring for exactly what
            # counts as a match here.
            strict_result = compute_strict_trial_match(state, trial, hierarchy=None, match_threshold=strict_match_threshold)
            if strict_result is not None:
                y_true_strict[p_idx, t_idx] = 1.0 if strict_result else 0.0
            # else: stays NaN -- trial had no assessable criteria for this patient
    
    # Log ground truth statistics
    total_eligible = y_true.sum()
    logging.info(f"Total eligible (patient, trial) pairs: {total_eligible} ({total_eligible/(num_patients*num_trials)*100:.2f}%)")
    logging.info(f"Trials with at least one eligible patient: {(y_true.sum(axis=0) > 0).sum()}")

    # NEW: strict matrix statistics
    n_assessable_strict = np.sum(~np.isnan(y_true_strict))
    n_total_strict = y_true_strict.size
    n_strict_matches = np.nansum(y_true_strict == 1.0)
    logging.info(f"[Strict/COMPOSE-style] Assessable (patient, trial) pairs: {int(n_assessable_strict)}/{n_total_strict} "
                 f"({n_assessable_strict/n_total_strict*100:.2f}%) -- rest had zero assessable criteria")
    if n_assessable_strict > 0:
        logging.info(f"[Strict/COMPOSE-style] Full matches among assessable pairs: {int(n_strict_matches)} "
                     f"({n_strict_matches/n_assessable_strict*100:.2f}%)")

    # 8. Calculate similarity scores for both models
    scores_baseline = np.zeros((num_patients, num_trials))
    scores_full = np.zeros((num_patients, num_trials))
    
    eta = getattr(cfg, 'ETA_EXCLUSION_PENALTY', 1.0)
    logging.info(f"Using exclusion penalty eta = {eta}")
    
    for t_idx, tid in enumerate(trial_ids):
        # Baseline trial vectors come from the pre-Stage-B space (matches h_baseline)
        z_inc_base, z_exc_base = trial_embeds_baseline[tid]
        z_inc_base = z_inc_base.squeeze().cpu()
        z_exc_base = z_exc_base.squeeze().cpu()

        # Full-model trial vectors come from the trained Stage B space (matches h_full)
        z_inc_full, z_exc_full = trial_embeds[tid]
        z_inc_full = z_inc_full.squeeze().cpu()
        z_exc_full = z_exc_full.squeeze().cpu()

        for p_idx in range(num_patients):
            # Stage A (Baseline) score -- both sides from the pre-alignment space
            z_patient_base = h_baseline[p_idx].squeeze().cpu()
            inc_sim_base = F.cosine_similarity(z_patient_base.unsqueeze(0), z_inc_base.unsqueeze(0)).item()
            exc_sim_base = F.cosine_similarity(z_patient_base.unsqueeze(0), z_exc_base.unsqueeze(0)).item()
            scores_baseline[p_idx, t_idx] = inc_sim_base - eta * exc_sim_base

            # Stage B (Full Model) score -- both sides from the trained space
            z_patient_full = h_full[p_idx].squeeze().cpu()
            inc_sim_full = F.cosine_similarity(z_patient_full.unsqueeze(0), z_inc_full.unsqueeze(0)).item()
            exc_sim_full = F.cosine_similarity(z_patient_full.unsqueeze(0), z_exc_full.unsqueeze(0)).item()
            scores_full[p_idx, t_idx] = inc_sim_full - eta * exc_sim_full

    # 9. Debug: Print score statistics
    logging.info(f"Scores baseline - mean: {scores_baseline.mean():.4f}, std: {scores_baseline.std():.4f}")
    logging.info(f"Scores full - mean: {scores_full.mean():.4f}, std: {scores_full.std():.4f}")
    logging.info(f"y_true - mean: {y_true.mean():.4f} ({y_true.sum()} positive pairs)")

    return y_true, scores_baseline, scores_full, y_true_strict

def evaluate_retrieval():
    """Main evaluation function."""
    cfg = Config()
    
    # Ensure output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Build evaluation matrices
    result = build_evaluation_matrices(cfg)
    if result[0] is None:
        logging.error("Evaluation failed - missing embeddings or data")
        return
    
    y_true, scores_base, scores_full, y_true_strict = result
    
    # Calculate metrics
    try:
        auc_base = roc_auc_score(y_true.ravel(), scores_base.ravel())
        auc_full = roc_auc_score(y_true.ravel(), scores_full.ravel())
        
        pr_base = average_precision_score(y_true.ravel(), scores_base.ravel())
        pr_full = average_precision_score(y_true.ravel(), scores_full.ravel())
    except Exception as e:
        logging.error(f"Error computing metrics: {e}")
        return

    # Log results
    logging.info("=" * 60)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"[Results] Stage A (Baseline) ROC-AUC: {auc_base:.4f}")
    logging.info(f"[Results] Stage B (Full Model) ROC-AUC: {auc_full:.4f}")
    logging.info(f"[Results] Stage A (Baseline) PR-AUC: {pr_base:.4f}")
    logging.info(f"[Results] Stage B (Full Model) PR-AUC: {pr_full:.4f}")
    
    # Calculate improvement
    auc_improvement = ((auc_full - auc_base) / max(auc_base, 0.001)) * 100
    pr_improvement = ((pr_full - pr_base) / max(pr_base, 0.001)) * 100
    
    logging.info(f"ROC-AUC Improvement: {auc_improvement:+.2f}%")
    logging.info(f"PR-AUC Improvement: {pr_improvement:+.2f}%")
    logging.info("=" * 60)

    # NEW: strict, COMPOSE-style trial-level accuracy. Our model outputs one
    # continuous score per (patient, trial), not a per-criterion decision
    # like COMPOSE's classifier -- so we binarize our score with a threshold
    # and compare it against y_true_strict (computed by the RULE-BASED
    # matching engine, since that's the only part of this pipeline that
    # scores individual criteria). This is the same AGGREGATION FORMULA
    # COMPOSE uses, applied to a different kind of prediction -- document
    # that distinction if you report this number.
    def strict_accuracy(y_strict, scores, threshold):
        mask = ~np.isnan(y_strict)
        if mask.sum() == 0:
            return None, 0
        y_flat = y_strict[mask]
        pred_flat = (scores[mask] >= threshold).astype(float)
        return float((y_flat == pred_flat).mean()), int(mask.sum())

    # Fixed threshold at 0.0: score = inc_sim - eta*exc_sim is symmetric
    # around 0, so this is a principled default, NOT tuned on this data.
    # This is the ONLY strict-accuracy number that's valid to report as a
    # final result without a separate validation split.
    FIXED_THRESHOLD = 0.0
    acc_base_fixed, n_eval_base = strict_accuracy(y_true_strict, scores_base, FIXED_THRESHOLD)
    acc_full_fixed, n_eval_full = strict_accuracy(y_true_strict, scores_full, FIXED_THRESHOLD)

    logging.info("=" * 60)
    logging.info("STRICT COMPOSE-STYLE TRIAL-LEVEL ACCURACY")
    logging.info("=" * 60)
    logging.info(f"  (evaluated on {n_eval_base} assessable pairs out of {y_true_strict.size} total)")
    logging.info(f"  Stage A (Baseline) accuracy @ fixed threshold=0.0: {acc_base_fixed:.4f}" if acc_base_fixed is not None else "  Stage A: no assessable pairs")
    logging.info(f"  Stage B (Full Model) accuracy @ fixed threshold=0.0: {acc_full_fixed:.4f}" if acc_full_fixed is not None else "  Stage B: no assessable pairs")

    # DIAGNOSTIC ONLY -- best-possible accuracy if the threshold were
    # cherry-picked on THIS SAME data. This is what you'd see if you (or a
    # reviewer) swept thresholds after the fact -- useful to know as an
    # upper bound, but NEVER report this number as your headline result.
    # Real threshold selection must happen on a held-out validation split
    # you have not looked at, or it's just threshold-shopping.
    candidate_thresholds = np.percentile(scores_full[~np.isnan(y_true_strict)], np.arange(1, 100, 1)) if n_eval_full > 0 else []
    best_acc_full, best_thresh = None, None
    for t in candidate_thresholds:
        acc, _ = strict_accuracy(y_true_strict, scores_full, t)
        if best_acc_full is None or acc > best_acc_full:
            best_acc_full, best_thresh = acc, t
    if best_acc_full is not None:
        logging.info(f"  [DIAGNOSTIC ONLY, NOT a valid reportable result] Best-in-hindsight Stage B accuracy: "
                     f"{best_acc_full:.4f} at threshold={best_thresh:.4f} -- this threshold was chosen by looking at "
                     f"this same evaluation data, so it does NOT generalize. Re-derive it on a separate validation "
                     f"split before using it for anything you report.")
    logging.info("=" * 60)

    # NEW: confidence intervals + significance test. A single number from a
    # single run can't tell you if an improvement (or a tie, like we're
    # currently seeing) is real or just noise -- this can.
    logging.info("Running patient-level bootstrap (1000 resamples) for CIs and significance...")
    stats = bootstrap_ci_and_pvalue(y_true, scores_base, scores_full, n_bootstrap=1000, seed=cfg.SEED)
    logging.info("=" * 60)
    logging.info("STATISTICAL SIGNIFICANCE (95% CI from patient-level bootstrap)")
    logging.info("=" * 60)
    logging.info(f"  Stage A ROC-AUC: {auc_base:.4f}  95% CI [{stats['auc_base_ci'][0]:.4f}, {stats['auc_base_ci'][1]:.4f}]")
    logging.info(f"  Stage B ROC-AUC: {auc_full:.4f}  95% CI [{stats['auc_full_ci'][0]:.4f}, {stats['auc_full_ci'][1]:.4f}]")
    logging.info(f"  ROC-AUC diff (B-A): {auc_full - auc_base:+.4f}  95% CI [{stats['auc_diff_ci'][0]:+.4f}, {stats['auc_diff_ci'][1]:+.4f}]  p={stats['auc_pvalue']:.4f}")
    logging.info(f"  Stage A PR-AUC:  {pr_base:.4f}  95% CI [{stats['pr_base_ci'][0]:.4f}, {stats['pr_base_ci'][1]:.4f}]")
    logging.info(f"  Stage B PR-AUC:  {pr_full:.4f}  95% CI [{stats['pr_full_ci'][0]:.4f}, {stats['pr_full_ci'][1]:.4f}]")
    logging.info(f"  PR-AUC diff (B-A):  {pr_full - pr_base:+.4f}  95% CI [{stats['pr_diff_ci'][0]:+.4f}, {stats['pr_diff_ci'][1]:+.4f}]  p={stats['pr_pvalue']:.4f}")
    if stats['auc_diff_ci'][0] <= 0 <= stats['auc_diff_ci'][1]:
        logging.info("  -> ROC-AUC difference CI includes 0: cannot claim Stage B reliably beats Stage A on this run/seed alone.")
    else:
        logging.info("  -> ROC-AUC difference CI excludes 0: Stage B vs A difference looks real for this run/seed.")
    logging.info("=" * 60)

    # Save results
    results = {
        'seed': cfg.SEED,
        'stage_a_roc_auc': float(auc_base),
        'stage_b_roc_auc': float(auc_full),
        'stage_a_pr_auc': float(pr_base),
        'stage_b_pr_auc': float(pr_full),
        'roc_improvement_pct': float(auc_improvement),
        'pr_improvement_pct': float(pr_improvement),
        'num_patients': y_true.shape[0],
        'num_trials': y_true.shape[1],
        'num_positive_pairs': int(y_true.sum()),
        'bootstrap': stats,
        'strict_composE_style': {
            'note': 'accuracy computed against matching_engine.compute_strict_trial_match '
                    '(all inclusion criteria matched AND all exclusion criteria mismatched). '
                    'Only the fixed_threshold_0.0 numbers are valid to report as final results; '
                    'best_in_hindsight is diagnostic only, computed by sweeping thresholds on this '
                    'same evaluation data, and does not generalize.',
            'n_assessable_pairs': n_eval_base,
            'n_total_pairs': int(y_true_strict.size),
            'stage_a_accuracy_fixed_threshold_0.0': acc_base_fixed,
            'stage_b_accuracy_fixed_threshold_0.0': acc_full_fixed,
            'stage_b_best_in_hindsight_accuracy': best_acc_full,
            'stage_b_best_in_hindsight_threshold': float(best_thresh) if best_thresh is not None else None,
        },
    }

    results_path = os.path.join(cfg.OUTPUT_DIR, f'evaluation_results_seed{cfg.SEED}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {results_path}")
    return results


if __name__ == "__main__":
    evaluate_retrieval()