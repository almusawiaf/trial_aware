#!/bin/bash

#SBATCH --job-name=test_gcl
#SBATCH --output=logs/test_gcl_%j.out
#SBATCH --error=logs/test_gcl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs
cd "$SLURM_SUBMIT_DIR" || exit 1
find . -type d -name "__pycache__" -exec rm -r {} +
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single seed, single (full, non-fold) trial split, GCL pretraining ON
# (the new default). This is a cheap sanity check before committing to a
# full 5-fold x multi-seed re-run: does Stage A actually move off 0.746
# now that it's a genuinely pretrained encoder instead of random init?
export RUN_SEED=0
export RUN_LAMBDA_ANCHOR=0.3
export RUN_ALIGN_LR=0.0003
export RUN_ENABLE_GCL=1

python models/claude_active/train.py
python models/claude_active/evaluate/evaluate.py