#!/bin/bash

#SBATCH --job-name=sweep_gcl
#SBATCH --output=logs/sweep_gcl_%j.out
#SBATCH --error=logs/sweep_gcl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs
cd "$SLURM_SUBMIT_DIR" || exit 1
find . -type d -name "__pycache__" -exec rm -r {} +
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Explicit for clarity even though "1" is now config.py's default --
# every run in this sweep uses a genuinely GCL-pretrained Stage A, not
# the old random-init baseline.
export RUN_ENABLE_GCL=1

# Wider grid than the original sweep, and shifted toward HIGHER
# lambda_anchor: the single-seed test showed that lambda=0.3 (tuned
# against a weak random-init Stage A) actively hurts ROC-AUC once Stage A
# is a strong GCL-pretrained baseline -- the working theory is the anchor
# needs to hold tighter to a good starting point than it did to a bad one.
# Dropped lambda=0.1 from the old grid since it was already clearly worse
# even under the old (weak-baseline) regime.
#
# 4 lambda_anchor x 2 align_lr x 3 seeds = 24 runs. At ~18 min/run
# (GCL pretraining itself only adds ~70s, most of the cost is still the
# 60-epoch alignment loop), budget ~7-8 hours; --time above has buffer.
python models/claude_active/run_anchor_sweep.py \
    --lambda-anchor 0.3 0.5 0.7 1.0 \
    --align-lr 0.0001 0.0003 \
    --seeds 0 1 2