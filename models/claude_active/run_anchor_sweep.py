"""
run_anchor_sweep.py

Systematically tests whether ANY combination of LAMBDA_ANCHOR / ALIGN_LR lets
Stage B genuinely beat the Stage A baseline, instead of guessing at one value
and hoping.

Why sweep these two specifically: LAMBDA_ANCHOR controls how tightly the
encoder is held to its pre-training representation (too tight = Stage B can't
learn anything new on top of the baseline, which is what we're currently
seeing; too loose = the encoder can drift and forget what made the baseline
decent in the first place). ALIGN_LR controls how fast the criterion/trial
encoders learn -- worth checking whether the current rate is even large
enough to move the needle in 60 epochs.

Runs each (lambda_anchor, align_lr) combination across multiple seeds (so a
lucky/unlucky single run doesn't mislead you), and reports mean +/- std of
the Stage B - Stage A gap for every combination, sorted best-first.

Usage:
    python run_anchor_sweep.py --lambda-anchor 0.1 0.3 0.5 --align-lr 1e-4 3e-4 --seeds 0 1 2
    (defaults are reasonable if you just run it with no arguments)

Cost warning: this runs the FULL train.py + evaluate.py once per
(lambda_anchor, align_lr, seed) combination. Default settings = 3 anchor
values x 1 lr value x 3 seeds = 9 full runs. Start small if each run is slow.
"""
import argparse
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys

import numpy as np

# NEW: resolve train.py/evaluate.py relative to THIS script's own location,
# not the current working directory -- so this file is always FOUND
# regardless of where it's launched from.
#
# IMPORTANT: we deliberately do NOT pass cwd=SCRIPT_DIR to subprocess.run
# below. train.py itself uses relative paths (e.g. config.py's
# OUTPUT_DIR = "./data") that assume it's being run from the
# PARENT folder (TRIAL_AWARE_2), even though the .py file lives one level
# down in C/. Forcing cwd to this script's own folder would find the file
# correctly but then break every relative path *inside* it. So: absolute
# path to WHICH script to run, but leave WHERE it runs from alone
# (inherited from however this sweep script itself was launched).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
EVALUATE_PY = os.path.join(SCRIPT_DIR, "evaluate", "evaluate.py")

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_one_combo(lambda_anchor: float, align_lr: float, seed: int):
    env = os.environ.copy()
    env["RUN_SEED"] = str(seed)
    env["RUN_LAMBDA_ANCHOR"] = str(lambda_anchor)
    env["RUN_ALIGN_LR"] = str(align_lr)

    tag = f"anchor{lambda_anchor}_lr{align_lr}_seed{seed}"
    logging.info(f"[{tag}] Running train.py ...")
    # Absolute path to WHICH script, but no cwd override -- see the note
    # above SCRIPT_DIR for why.
    subprocess.run([sys.executable, TRAIN_PY], env=env, check=True)

    logging.info(f"[{tag}] Running evaluate.py ...")
    subprocess.run([sys.executable, EVALUATE_PY], env=env, check=True)

    cfg = Config()  # reads THIS process's env, not the child's -- fine, SEED is the only path-relevant field
    src = os.path.join(cfg.OUTPUT_DIR, f"evaluation_results_seed{seed}.json")
    with open(src, "r") as f:
        result = json.load(f)

    # Copy the result under a name that encodes the swept params, since the
    # next combo run with the same seed would otherwise overwrite this file.
    dst = os.path.join(cfg.OUTPUT_DIR, f"sweep_{tag}.json")
    shutil.copy(src, dst)

    result["lambda_anchor"] = lambda_anchor
    result["align_lr"] = align_lr
    result["seed"] = seed
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-anchor", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--align-lr", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    all_results = []
    combos = list(itertools.product(args.lambda_anchor, args.align_lr))
    logging.info(f"Sweeping {len(combos)} config(s) x {len(args.seeds)} seed(s) = "
                 f"{len(combos) * len(args.seeds)} total runs")

    for lambda_anchor, align_lr in combos:
        for seed in args.seeds:
            result = run_one_combo(lambda_anchor, align_lr, seed)
            all_results.append(result)

    # Aggregate: group by (lambda_anchor, align_lr), average across seeds
    summary_rows = []
    for lambda_anchor, align_lr in combos:
        matching = [r for r in all_results if r["lambda_anchor"] == lambda_anchor and r["align_lr"] == align_lr]
        auc_a = np.array([r["stage_a_roc_auc"] for r in matching])
        auc_b = np.array([r["stage_b_roc_auc"] for r in matching])
        pr_a = np.array([r["stage_a_pr_auc"] for r in matching])
        pr_b = np.array([r["stage_b_pr_auc"] for r in matching])
        diff = auc_b - auc_a
        summary_rows.append({
            "lambda_anchor": lambda_anchor,
            "align_lr": align_lr,
            "auc_a_mean": float(auc_a.mean()), "auc_a_std": float(auc_a.std()),
            "auc_b_mean": float(auc_b.mean()), "auc_b_std": float(auc_b.std()),
            "pr_a_mean": float(pr_a.mean()), "pr_b_mean": float(pr_b.mean()),
            "diff_mean": float(diff.mean()), "diff_std": float(diff.std()),
        })

    summary_rows.sort(key=lambda r: r["diff_mean"], reverse=True)

    logging.info("=" * 90)
    logging.info("SWEEP RESULTS (sorted best Stage B - Stage A gap first)")
    logging.info("=" * 90)
    logging.info(f"{'LAMBDA_ANCHOR':>14} {'ALIGN_LR':>10} {'AUC_A':>8} {'AUC_B':>8} {'B-A diff':>12} {'PR_A':>8} {'PR_B':>8}")
    for row in summary_rows:
        flag = ""
        # Rough "is this a real improvement, not just noise" check: mean gap
        # is at least 1 std above zero. This is a quick heuristic for
        # deciding what's worth a real bootstrap significance test (see
        # evaluate.py) -- not a substitute for one.
        if row["diff_mean"] - row["diff_std"] > 0:
            flag = "  <-- gap consistently positive across seeds"
        logging.info(
            f"{row['lambda_anchor']:>14.3f} {row['align_lr']:>10.1e} "
            f"{row['auc_a_mean']:>8.4f} {row['auc_b_mean']:>8.4f} "
            f"{row['diff_mean']:>+7.4f}\u00b1{row['diff_std']:.4f} "
            f"{row['pr_a_mean']:>8.4f} {row['pr_b_mean']:>8.4f}{flag}"
        )
    logging.info("=" * 90)
    logging.info("Next step: take the best config's exact seed-level result and run it through "
                  "evaluate.py's bootstrap p-value (already computed per-seed in each sweep_*.json) "
                  "before concluding it's a real win.")

    cfg = Config()
    summary_path = os.path.join(cfg.OUTPUT_DIR, "anchor_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"summary": summary_rows, "per_run_results": all_results}, f, indent=2)
    logging.info(f"Full sweep summary saved to {summary_path}")


if __name__ == "__main__":
    main()