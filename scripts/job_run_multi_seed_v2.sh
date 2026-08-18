#!/bin/bash

#SBATCH --job-name=multi_seed_v2        # Job name
#SBATCH --output=logs/multiseed_v2_%j.out
#SBATCH --error=logs/multiseed_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00                 # 5 seeds x (~20-35 min train+eval on CPU) -- generous buffer
#SBATCH --partition=cpu                 # CPU partition -- check exact name with `sinfo` on your cluster

mkdir -p logs

cd "$SLURM_SUBMIT_DIR" || exit 1
find . -type d -name "__pycache__" -exec rm -r {} +

# --- Corrected hyperparameters from the anchor sweep -----------------------
# The sweep showed align_lr=1e-4 (the old default) is an UNSTABLE regime --
# sometimes helps, sometimes collapses well below baseline, depending on
# seed. align_lr=3e-4 was robustly positive across every seed and every
# lambda_anchor tested. lambda_anchor=0.3 gave the largest, tightest gain
# at that learning rate. These are now our reported final hyperparameters.
export RUN_LAMBDA_ANCHOR=0.3
export RUN_ALIGN_LR=0.0003

# --- Preserve the OLD (default-hyperparameter) results before overwriting --
# config.py's checkpoint/results filenames only encode the seed, not
# lambda_anchor/align_lr, so re-running with different hyperparameters at
# the same seeds would silently clobber the earlier evidence of the
# instability -- and that instability finding is worth keeping for the paper.
BACKUP_DIR="data/archive_default_lr1e-4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
for f in data/evaluation_results_seed*.json data/multi_seed_summary.json \
         data/patient_embeddings_seed*.pt data/trial_embeddings_seed*.pt \
         data/patient_embeddings_baseline_seed*.pt data/trial_embeddings_baseline_seed*.pt; do
  [ -e "$f" ] && mv "$f" "$BACKUP_DIR/"
done
echo "Backed up pre-existing default-hyperparameter results to $BACKUP_DIR"

# --- Run ---------------------------------------------------------------
python models/claude_active/run_multi_seed.py --seeds 0 1 2 3 4

# --- Tag the new outputs so a future run (different hyperparams again)
#     can't silently clobber THESE either -----------------------------------
TAG="lambda0.3_lr3e-4"
mv data/multi_seed_summary.json "data/multi_seed_summary_${TAG}.json"
for f in data/evaluation_results_seed*.json; do
  [ -e "$f" ] && mv "$f" "data/$(basename "$f" .json)_${TAG}.json"
done
echo "New results tagged with suffix _${TAG}"