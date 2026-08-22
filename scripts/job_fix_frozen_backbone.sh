#!/bin/bash

#SBATCH --job-name=fix_frozen
#SBATCH --output=logs/fix_frozen_%j.out
#SBATCH --error=logs/fix_frozen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

mkdir -p logs
cd "$SLURM_SUBMIT_DIR" || exit 1
find . -type d -name "__pycache__" -exec rm -r {} +
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN_ENABLE_GCL=1

# Fix candidate #2: freeze the GNN backbone entirely during Stage B.
# LAMBDA_ANCHOR is moot here (encoder can't drift from Stage A at all,
# by construction) -- still passing a value since the script requires
# one, but it has no effect. align_lr only affects CriterionEncoder/
# TrialEncoder here, not the frozen backbone.
python models/claude_active/run_anchor_sweep.py \
    --freeze-backbone \
    --lambda-anchor 0.5 \
    --align-lr 0.0003 \
    --seeds 0 1 2