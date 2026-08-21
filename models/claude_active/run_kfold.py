"""
run_kfold.py

Runs Stage B training + evaluation across K trial folds (see
make_trial_folds.py) and reports mean +/- std ROC-AUC/PR-AUC across folds.
Every trial is evaluated exactly once (in whichever fold it landed in as
held-out), and trained on in the other K-1 folds -- so unlike a single
80/20 split, the result doesn't depend on which trials happened to land in
the one held-out 20%.

Run make_trial_folds.py once before this, if you haven't already.

Usage:
    python run_kfold.py --k 5
    python run_kfold.py --k 5 --seed 0     # fix the training seed; only fold varies
"""
import argparse
import json
import logging
import os
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
EVALUATE_PY = os.path.join(SCRIPT_DIR, "evaluate", "evaluate.py")

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_one_fold(fold: int, seed: int, lambda_anchor: float, align_lr: float):
    env = os.environ.copy()
    env["RUN_FOLD"] = str(fold)
    env["RUN_SEED"] = str(seed)
    env["RUN_LAMBDA_ANCHOR"] = str(lambda_anchor)
    env["RUN_ALIGN_LR"] = str(align_lr)

    logging.info(f"[Fold {fold}] Running train.py ...")
    subprocess.run([sys.executable, TRAIN_PY], env=env, check=True)

    logging.info(f"[Fold {fold}] Running evaluate.py ...")
    subprocess.run([sys.executable, EVALUATE_PY], env=env, check=True)

    cfg = Config()  # OUTPUT_DIR doesn't depend on fold/seed, safe to read here
    # evaluate.py's results filename is seed-based only (not fold-aware) --
    # tag it with the fold number ourselves right after each run so fold 1's
    # result doesn't get overwritten by fold 2's run at the same seed.
    src = os.path.join(cfg.OUTPUT_DIR, f"evaluation_results_seed{seed}.json")
    with open(src, "r") as f:
        result = json.load(f)

    dst = os.path.join(cfg.OUTPUT_DIR, f"kfold_seed{seed}_fold{fold}.json")
    with open(dst, "w") as f:
        json.dump(result, f, indent=2)

    result["fold"] = fold
    result["seed"] = seed
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Training seed, held fixed across folds")
    parser.add_argument("--lambda-anchor", type=float, default=0.3)
    parser.add_argument("--align-lr", type=float, default=3e-4)
    args = parser.parse_args()

    cfg = Config()
    fold_dir = os.path.join(cfg.TRIALS_DATA_DIR, "folds")
    if not os.path.isdir(fold_dir):
        logging.error(f"{fold_dir} not found -- run make_trial_folds.py first.")
        sys.exit(1)

    all_results = []
    for fold in range(args.k):
        result = run_one_fold(fold, args.seed, args.lambda_anchor, args.align_lr)
        all_results.append(result)

    auc_a = np.array([r['stage_a_roc_auc'] for r in all_results])
    auc_b = np.array([r['stage_b_roc_auc'] for r in all_results])
    pr_a = np.array([r['stage_a_pr_auc'] for r in all_results])
    pr_b = np.array([r['stage_b_pr_auc'] for r in all_results])
    diff = auc_b - auc_a

    logging.info("=" * 60)
    logging.info(f"K-FOLD CROSS-VALIDATION SUMMARY (k={args.k}, seed={args.seed}, "
                 f"lambda_anchor={args.lambda_anchor}, align_lr={args.align_lr})")
    logging.info("=" * 60)
    for i, r in enumerate(all_results):
        logging.info(f"  Fold {i}: A={r['stage_a_roc_auc']:.4f}  B={r['stage_b_roc_auc']:.4f}  "
                     f"diff={r['stage_b_roc_auc']-r['stage_a_roc_auc']:+.4f}  "
                     f"(n_eval_trials={r.get('num_trials', '?')})")
    logging.info("-" * 60)
    logging.info(f"  Stage A ROC-AUC: {auc_a.mean():.4f} +/- {auc_a.std(ddof=1):.4f}")
    logging.info(f"  Stage B ROC-AUC: {auc_b.mean():.4f} +/- {auc_b.std(ddof=1):.4f}")
    logging.info(f"  B-A diff:        {diff.mean():+.4f} +/- {diff.std(ddof=1):.4f}  "
                 f"({sum(diff>0)}/{args.k} folds positive)")
    logging.info(f"  Stage A PR-AUC:  {pr_a.mean():.4f} +/- {pr_a.std(ddof=1):.4f}")
    logging.info(f"  Stage B PR-AUC:  {pr_b.mean():.4f} +/- {pr_b.std(ddof=1):.4f}")
    logging.info("=" * 60)

    from scipy import stats
    t, p = stats.ttest_rel(auc_b, auc_a)
    logging.info(f"  Paired t-test across folds (ROC-AUC): t={t:.3f}, p={p:.4f}")
    logging.info("=" * 60)

    summary_path = os.path.join(cfg.OUTPUT_DIR, f"kfold_summary_seed{args.seed}_lambda{args.lambda_anchor}_lr{args.align_lr}.json")
    with open(summary_path, "w") as f:
        json.dump({
            'k': args.k, 'seed': args.seed,
            'lambda_anchor': args.lambda_anchor, 'align_lr': args.align_lr,
            'stage_a_roc_auc_mean': float(auc_a.mean()), 'stage_a_roc_auc_std': float(auc_a.std(ddof=1)),
            'stage_b_roc_auc_mean': float(auc_b.mean()), 'stage_b_roc_auc_std': float(auc_b.std(ddof=1)),
            'diff_mean': float(diff.mean()), 'diff_std': float(diff.std(ddof=1)),
            'folds_positive': int(sum(diff > 0)),
            'paired_ttest_t': float(t), 'paired_ttest_p': float(p),
            'per_fold_results': all_results,
        }, f, indent=2)
    logging.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()