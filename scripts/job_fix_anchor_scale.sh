#!/bin/bash

#SBATCH --job-name=fix_anchor
#SBATCH --output=logs/fix_anchor_%j.out
#SBATCH --error=logs/fix_anchor_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs
cd "$SLURM_SUBMIT_DIR" || exit 1
find . -type d -name "__pycache__" -exec rm -r {} +
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN_ENABLE_GCL=1

# Fix candidate #1: normalized anchor loss (now bounded [0,4] like the
# rest of the loss, instead of raw unbounded MSE). Since the SCALE
# changed, the lambda values that were catastrophic under the old
# unnormalized loss (0.3, 0.5) don't mean the same thing anymore -- a
# bounded loss term typically needs a LARGER multiplier to exert
# comparable influence to the other (implicitly weight=1) loss terms.
# Testing a wider, higher range accordingly.
python models/claude_active/run_anchor_sweep.py \
    --lambda-anchor 0.5 1.0 2.0 5.0 \
    --align-lr 0.0003 \
    --seeds 0 1 2