#!/bin/bash

#SBATCH --job-name=trial_pipeline        # Combined pipeline job name
#SBATCH --output=logs/pipeline_%j.out   # Standard output log (%j inserts job ID)
#SBATCH --error=logs/pipeline_%j.err    # Standard error log
#SBATCH --nodes=1                       # Run all tasks on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=8               # Number of CPU cores
#SBATCH --mem=128G                      # Total memory
#SBATCH --time=4-00:00:00               # Time limit (days-hours:min:sec)
#SBATCH --partition=gpu                 # GPU partition
#SBATCH --gres=gpu:1                    # 1 GPU requested

# 1. Ensure logs directory exists
mkdir -p logs

# 2. Clean PyCache to prevent stale module imports
find . -type d -name "__pycache__" -exec rm -r {} +

# 3. Memory management environment flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ----------------------------------------------------
# Step 1: Run Stage B Fine-Tuning (train.py)
# ----------------------------------------------------
echo "=================================================="
echo "Starting Stage B Training at $(date)"
echo "=================================================="

python train.py

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training script failed with exit code $TRAIN_EXIT_CODE. Aborting evaluation."
    exit $TRAIN_EXIT_CODE
fi

# ----------------------------------------------------
# Step 2: Run Evaluation (evaluate.py)
# ----------------------------------------------------
echo "=================================================="
echo "Stage B Training Completed Successfully!"
echo "Starting Model Evaluation at $(date)"
echo "=================================================="

python evaluate.py

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation script failed with exit code $EVAL_EXIT_CODE."
    exit $EVAL_EXIT_CODE
fi

echo "=================================================="
echo "Full Pipeline Completed Successfully at $(date)"
echo "=================================================="