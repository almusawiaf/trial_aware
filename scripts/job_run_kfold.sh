#!/bin/bash

#SBATCH --job-name=kfold_cv
#SBATCH --output=logs/kfold_%j.out
#SBATCH --error=logs/kfold_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-06:00:00                 # 5 folds x (~20-35 min train+eval on CPU) -- generous buffer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                  

mkdir -p logs
cd "$SLURM_SUBMIT_DIR" || exit 1
find . -type d -name "__pycache__" -exec rm -r {} +

# lambda_anchor=0.3, align_lr=3e-4 are the corrected hyperparameters from
# the earlier sweep -- kept fixed here since fold is the thing varying.
# seed is also held fixed (0) so fold identity is the only thing changing
# across these 5 runs.
python models/claude_active/run_kfold.py --k 5 --seed 0 --lambda-anchor 0.3 --align-lr 0.0003