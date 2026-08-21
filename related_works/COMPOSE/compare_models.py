"""
compare_models.py -- OPTIONAL. Produces a statistically rigorous, three-way
comparison (COMPOSE vs Stage A vs Stage B) with patient-level bootstrap
95% CIs and p-values, exactly like the CIs already computed inside
models/claude_active/evaluate/compose_based/evaluate.py.

This needs Stage A's and Stage B's raw (patient, trial) SCORE MATRICES,
which the main pipeline's evaluate.py does not save by default (it only
writes the scalar ROC-AUC/PR-AUC to evaluation_results_seed{SEED}.json).
To enable this script, add the following two lines to
models/claude_active/evaluate/compose_based/evaluate.py, right before its
`return y_true, scores_baseline, scores_full, y_true_strict` statement
inside build_evaluation_matrices():

    np.savez(os.path.join(cfg.OUTPUT_DIR, f"raw_score_matrices_seed{cfg.SEED}.npz"),
             y_true=y_true, scores_baseline=scores_baseline, scores_full=scores_full)

Then re-run models/claude_active's evaluate.py once, and this script.

Usage:
    cd related_works/COMPOSE
    python evaluate.py            # writes results/compose_scores_seed{SEED}.npz
    python compare_models.py
"""
import json
import logging
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def bootstrap_three_way(y_true, scores_a, scores_b, scores_compose, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    P = y_true.shape[0]
    rows = {k: [] for k in [
        "auc_a", "auc_b", "auc_compose",
        "pr_a", "pr_b", "pr_compose",
        "auc_b_minus_compose", "pr_b_minus_compose",
    ]}
    for i in range(n_bootstrap):
        idx = rng.integers(0, P, size=P)
        yt = y_true[idx].ravel()
        if yt.min() == yt.max():
            continue
        sa, sb, sc = scores_a[idx].ravel(), scores_b[idx].ravel(), scores_compose[idx].ravel()
        rows["auc_a"].append(roc_auc_score(yt, sa))
        rows["auc_b"].append(roc_auc_score(yt, sb))
        rows["auc_compose"].append(roc_auc_score(yt, sc))
        rows["pr_a"].append(average_precision_score(yt, sa))
        rows["pr_b"].append(average_precision_score(yt, sb))
        rows["pr_compose"].append(average_precision_score(yt, sc))
        rows["auc_b_minus_compose"].append(rows["auc_b"][-1] - rows["auc_compose"][-1])
        rows["pr_b_minus_compose"].append(rows["pr_b"][-1] - rows["pr_compose"][-1])

    def ci(key):
        return float(np.percentile(rows[key], 2.5)), float(np.percentile(rows[key], 97.5))

    def pvalue(key):
        diffs = np.array(rows[key])
        sign = np.sign(np.mean(diffs))
        if sign == 0:
            return 1.0
        return float(min(1.0, 2 * np.mean(diffs * sign <= 0)))

    return {
        "auc_a_ci": ci("auc_a"), "auc_b_ci": ci("auc_b"), "auc_compose_ci": ci("auc_compose"),
        "pr_a_ci": ci("pr_a"), "pr_b_ci": ci("pr_b"), "pr_compose_ci": ci("pr_compose"),
        "auc_b_minus_compose_ci": ci("auc_b_minus_compose"), "auc_b_minus_compose_pvalue": pvalue("auc_b_minus_compose"),
        "pr_b_minus_compose_ci": ci("pr_b_minus_compose"), "pr_b_minus_compose_pvalue": pvalue("pr_b_minus_compose"),
        "n_valid_draws": len(rows["auc_a"]),
    }


def main():
    stage_ab_path = os.path.join(config.MAIN_DATA_DIR, f"raw_score_matrices_seed{config.SEED}.npz")
    compose_path = config.SCORES_PATH

    for p in (stage_ab_path, compose_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. See the module docstring at the top of this file for the "
                f"one-line change needed in models/claude_active/evaluate/compose_based/evaluate.py, "
                f"and make sure you've run `python evaluate.py` in related_works/COMPOSE first."
            )

    stage_ab = np.load(stage_ab_path)
    compose = np.load(compose_path, allow_pickle=True)

    y_true = stage_ab["y_true"]
    scores_a, scores_b = stage_ab["scores_baseline"], stage_ab["scores_full"]
    scores_compose = compose["scores"]

    assert y_true.shape == scores_compose.shape, (
        f"Shape mismatch: main pipeline evaluated {y_true.shape}, COMPOSE evaluated "
        f"{scores_compose.shape}. Both scripts must be run against the SAME structured "
        f"trials JSON and the same patient population -- re-run both with the same RUN_SEED."
    )

    stats = bootstrap_three_way(y_true, scores_a, scores_b, scores_compose,
                                 n_bootstrap=config.N_BOOTSTRAP, seed=config.SEED)

    logging.info("=" * 70)
    logging.info("THREE-WAY BOOTSTRAP COMPARISON (95% CI, patient-level resampling)")
    logging.info("=" * 70)
    logging.info(f"Stage A     ROC-AUC 95% CI: {stats['auc_a_ci']}")
    logging.info(f"Stage B     ROC-AUC 95% CI: {stats['auc_b_ci']}")
    logging.info(f"COMPOSE     ROC-AUC 95% CI: {stats['auc_compose_ci']}")
    logging.info(f"Stage B - COMPOSE diff 95% CI: {stats['auc_b_minus_compose_ci']}  "
                 f"p={stats['auc_b_minus_compose_pvalue']:.4f}")
    logging.info(f"Stage A     PR-AUC 95% CI: {stats['pr_a_ci']}")
    logging.info(f"Stage B     PR-AUC 95% CI: {stats['pr_b_ci']}")
    logging.info(f"COMPOSE     PR-AUC 95% CI: {stats['pr_compose_ci']}")
    logging.info(f"Stage B - COMPOSE diff 95% CI: {stats['pr_b_minus_compose_ci']}  "
                 f"p={stats['pr_b_minus_compose_pvalue']:.4f}")
    logging.info("=" * 70)

    out_path = os.path.join(config.OUT_DIR, f"three_way_comparison_seed{config.SEED}.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
