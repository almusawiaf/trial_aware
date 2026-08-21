"""
make_trial_folds.py

Builds K reproducible cross-validation folds over ALL trials (recombining
the existing 550-train / 138-eval split back into one pool of 688, then
re-splitting K ways). This replaces a single 80/20 split -- which risks the
held-out 20% happening to be an unusually easy or hard subset -- with K
different held-out sets, so every trial gets evaluated exactly once and
every trial gets trained on in K-1 of the K folds.

Run ONCE, before run_kfold.py. Output goes in
data/10000_trials/folds/fold{i}_train.json and fold{i}_eval.json for
i in [0, K).

Usage:
    python make_trial_folds.py --k 5
"""
import argparse
import json
import logging
import os
import random

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    # Fixed, independent of any training seed -- fold ASSIGNMENT should not
    # change if you later sweep training seeds, or "seed 0's fold 2" would
    # silently mean a different set of trials each time you touched RUN_SEED.
    parser.add_argument("--fold-assignment-seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = Config()

    with open(cfg.TRAIN_TRIALS_PATH, "r") as f:
        train_trials = json.load(f)
    with open(cfg.EVAL_TRIALS_PATH, "r") as f:
        eval_trials = json.load(f)

    all_trials = train_trials + eval_trials
    logging.info(f"Recombined {len(train_trials)} + {len(eval_trials)} = {len(all_trials)} total trials")

    # Dedupe defensively by trial_id/nct_id in case of any overlap -- there
    # shouldn't be any (they came from disjoint files), but silently training
    # and evaluating on a duplicated trial would reintroduce exactly the kind
    # of leakage this whole exercise is trying to eliminate.
    seen = set()
    deduped = []
    for t in all_trials:
        tid = t.get("trial_id") or t.get("nct_id")
        if tid in seen:
            logging.warning(f"Duplicate trial_id found across train/eval files: {tid} -- skipping duplicate")
            continue
        seen.add(tid)
        deduped.append(t)
    all_trials = deduped
    logging.info(f"{len(all_trials)} unique trials after dedup")

    rng = random.Random(args.fold_assignment_seed)
    shuffled = all_trials[:]
    rng.shuffle(shuffled)

    k = args.k
    fold_dir = os.path.join(cfg.TRIALS_DATA_DIR, "folds")
    os.makedirs(fold_dir, exist_ok=True)

    fold_sizes = []
    for i in range(k):
        eval_fold = shuffled[i::k]  # every k-th trial, offset by i -- even fold sizes
        train_fold = [t for t in shuffled if t not in eval_fold]
        # NOTE: `not in eval_fold` on dicts is O(n) per check -- fine at
        # n=688, would need a set-of-ids approach at larger scale.

        train_path = os.path.join(fold_dir, f"fold{i}_train.json")
        eval_path = os.path.join(fold_dir, f"fold{i}_eval.json")
        with open(train_path, "w") as f:
            json.dump(train_fold, f, indent=2)
        with open(eval_path, "w") as f:
            json.dump(eval_fold, f, indent=2)

        fold_sizes.append((len(train_fold), len(eval_fold)))
        logging.info(f"Fold {i}: {len(train_fold)} train / {len(eval_fold)} eval -> {train_path}, {eval_path}")

    total_eval = sum(sz[1] for sz in fold_sizes)
    logging.info(f"Total eval trials across all folds: {total_eval} "
                 f"(should equal {len(all_trials)} -- every trial evaluated exactly once)")
    assert total_eval == len(all_trials), "Fold sizes don't sum to total trial count -- check the split logic"


if __name__ == "__main__":
    main()