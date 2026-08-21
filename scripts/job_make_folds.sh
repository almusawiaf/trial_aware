#!/bin/bash

#SBATCH --job-name=make_folds
#SBATCH --output=logs/makefolds_%j.out
#SBATCH --error=logs/makefolds_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=cpu

mkdir -p logs
cd "$SLURM_SUBMIT_DIR" || exit 1

python models/claude_active/make_trial_folds.py --k 5