"""
run_multi_seed.py

Runs Stage B training + evaluation across several random seeds and reports
mean +/- std for both ROC-AUC and PR-AUC. A single-seed number can't tell you
whether a result is reliable; this can.

Uses subprocess (not in-process re-calls) because train.py uses
ProcessPoolExecutor internally, which doesn't play nicely with being called
repeatedly inside one long-lived Python process. This is slower per-run
overhead but much more robust.

Usage:
    python run_multi_seed.py --seeds 0 1 2 3 4
    python run_multi_seed.py --seeds 0 1 2 3 4 --skip-train   # if you already
                                                                 # trained all seeds
                                                                 # and just want
                                                                 # to re-aggregate
"""
import argparse
import json
import logging
import os
import subprocess
import sys

import numpy as np

# NEW: resolve train.py/evaluate.py by absolute path so they're always
# FOUND regardless of launch directory. Deliberately no cwd override when
# running them (see run_anchor_sweep.py for the full explanation) --
# train.py's own relative paths (config.py's ./data) assume it
# runs from the parent folder, not from wherever this .py file happens to live.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
EVALUATE_PY = os.path.join(SCRIPT_DIR, "evaluate.py")

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_one_seed(seed: int, skip_train: bool):
    env = os.environ.copy()
    env["RUN_SEED"] = str(seed)

    if not skip_train:
        logging.info(f"[Seed {seed}] Running train.py ...")
        subprocess.run([sys.executable, TRAIN_PY], env=env, check=True)

    logging.info(f"[Seed {seed}] Running evaluate.py ...")
    subprocess.run([sys.executable, EVALUATE_PY], env=env, check=True)

    cfg = Config()  # cfg.SEED reads RUN_SEED from this process's own env, not the child's
    results_path = os.path.join(cfg.OUTPUT_DIR, f'evaluation_results_seed{seed}.json')
    with open(results_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--skip-train", action="store_true",
                         help="Only re-run evaluate.py (e.g. results files already exist)")
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        result = run_one_seed(seed, args.skip_train)
        all_results.append(result)

    auc_a = np.array([r['stage_a_roc_auc'] for r in all_results])
    auc_b = np.array([r['stage_b_roc_auc'] for r in all_results])
    pr_a = np.array([r['stage_a_pr_auc'] for r in all_results])
    pr_b = np.array([r['stage_b_pr_auc'] for r in all_results])

    logging.info("=" * 60)
    logging.info(f"MULTI-SEED SUMMARY ({len(args.seeds)} seeds: {args.seeds})")
    logging.info("=" * 60)
    logging.info(f"  Stage A ROC-AUC: {auc_a.mean():.4f} +/- {auc_a.std():.4f}  (values: {np.round(auc_a, 4).tolist()})")
    logging.info(f"  Stage B ROC-AUC: {auc_b.mean():.4f} +/- {auc_b.std():.4f}  (values: {np.round(auc_b, 4).tolist()})")
    logging.info(f"  Stage A PR-AUC:  {pr_a.mean():.4f} +/- {pr_a.std():.4f}  (values: {np.round(pr_a, 4).tolist()})")
    logging.info(f"  Stage B PR-AUC:  {pr_b.mean():.4f} +/- {pr_b.std():.4f}  (values: {np.round(pr_b, 4).tolist()})")
    logging.info("-" * 60)
    logging.info("  If Stage B's mean is only a fraction of a std above Stage A's mean, "
                  "the improvement is not distinguishable from seed-to-seed noise.")
    logging.info("=" * 60)

    cfg = Config()
    summary_path = os.path.join(cfg.OUTPUT_DIR, "multi_seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            'seeds': args.seeds,
            'stage_a_roc_auc_mean': float(auc_a.mean()), 'stage_a_roc_auc_std': float(auc_a.std()),
            'stage_b_roc_auc_mean': float(auc_b.mean()), 'stage_b_roc_auc_std': float(auc_b.std()),
            'stage_a_pr_auc_mean': float(pr_a.mean()), 'stage_a_pr_auc_std': float(pr_a.std()),
            'stage_b_pr_auc_mean': float(pr_b.mean()), 'stage_b_pr_auc_std': float(pr_b.std()),
            'per_seed_results': all_results,
        }, f, indent=2)
    logging.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()