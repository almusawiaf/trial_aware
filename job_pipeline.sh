#!/bin/bash

#SBATCH --job-name=full_pipeline          # Full pipeline job name
#SBATCH --output=logs/full_pipeline_%j.out   # Standard output log
#SBATCH --error=logs/full_pipeline_%j.err    # Standard error log
#SBATCH --nodes=1                         # Run all tasks on a single node
#SBATCH --ntasks=1                        # Run a single task
#SBATCH --cpus-per-task=8                 # Number of CPU cores
#SBATCH --mem=128G                        # Total memory
#SBATCH --time=4-00:00:00                 # Time limit (days-hours:min:sec)
#SBATCH --partition=gpu                   # GPU partition
#SBATCH --gres=gpu:1                      # 1 GPU requested

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

# 1. Ensure logs directory exists
mkdir -p logs

# 2. Clean PyCache to prevent stale module imports
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true

# 3. Load conda environment
source /lustre/home/almusawiaf/anaconda3/etc/profile.d/conda.sh
conda activate envGNN4

# 4. Memory management environment flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "  FULL PIPELINE - START"
echo "============================================================"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo "============================================================"

# ============================================================
# Step 1: Run Data Preprocessing & Graph Construction (run.py)
# ============================================================
echo ""
echo "=================================================="
echo "Step 1: Running Preprocessing & Graph Construction"
echo "Started at: $(date)"
echo "=================================================="

python run.py

RUN_EXIT_CODE=$?

if [ $RUN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: run.py failed with exit code $RUN_EXIT_CODE. Aborting."
    exit $RUN_EXIT_CODE
fi

echo "✅ run.py completed successfully at $(date)"

# ============================================================
# Step 2: Run Training (train.py)
# ============================================================
echo ""
echo "=================================================="
echo "Step 2: Running Stage A + Stage B Training"
echo "Started at: $(date)"
echo "=================================================="

python train.py

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: train.py failed with exit code $TRAIN_EXIT_CODE. Aborting."
    exit $TRAIN_EXIT_CODE
fi

echo "✅ train.py completed successfully at $(date)"

# ============================================================
# Step 3: Run Evaluation (evaluate.py)
# ============================================================
echo ""
echo "=================================================="
echo "Step 3: Running Evaluation"
echo "Started at: $(date)"
echo "=================================================="

python evaluate.py

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: evaluate.py failed with exit code $EVAL_EXIT_CODE."
    exit $EVAL_EXIT_CODE
fi

echo "✅ evaluate.py completed successfully at $(date)"

# ============================================================
# COMPLETION SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "  ✅ FULL PIPELINE COMPLETED SUCCESSFULLY"
echo "============================================================"
echo "  Completed at: $(date)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Output logs: logs/full_pipeline_${SLURM_JOB_ID}.out"
echo "  Error logs: logs/full_pipeline_${SLURM_JOB_ID}.err"
echo "============================================================"

# Print result summary
echo ""
echo "📊 Results Summary:"
if [ -f "processed_data/evaluation_results.json" ]; then
    echo "  Evaluation results saved to: processed_data/evaluation_results.json"
    cat processed_data/evaluation_results.json
else
    echo "  ⚠️ Evaluation results not found"
fi