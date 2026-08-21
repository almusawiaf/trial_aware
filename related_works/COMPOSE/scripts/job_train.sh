#!/bin/bash

#SBATCH --job-name=compose_training
#SBATCH --output=logs/train_compose_%j.out
#SBATCH --error=logs/train_compose_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

# IMPORTANT: submit this job from related_works/COMPOSE/, e.g.:
#     cd related_works/COMPOSE
#     sbatch scripts/job_train.sh
# $SLURM_SUBMIT_DIR is wherever `sbatch` was invoked from -- this script's
# python call (`python train.py`) is a bare relative path that assumes
# train.py sits directly in the current directory, exactly like this
# baseline's layout (related_works/COMPOSE/train.py, related_works/COMPOSE/scripts/job_train.sh).
# If you submit from the repo root instead, either `cd related_works/COMPOSE`
# below or change the python call to the full relative path.
cd "$SLURM_SUBMIT_DIR" || exit 1

# module load cuda/12.1
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate trial_aware

# This baseline reads the MAIN pipeline's already-processed data
# (../../data/diagnoses_clean.parquet, ../../data/10000_trials/structured_clinical_trials.json,
# see config.py). Make sure that data already exists -- i.e. the main
# pipeline's run.py (and trial structuring step) has been run at least
# once -- before submitting this job. This script does NOT regenerate it.

find . -type d -name "__pycache__" -exec rm -r {} +

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use the same seed you use for the main pipeline's Stage A/B runs so the
# eventual comparison table lines up (see scripts/job_evaluate.sh and
# README.md).
export RUN_SEED=${RUN_SEED:-0}

python train.py