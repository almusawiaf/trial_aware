#!/bin/bash

#SBATCH --job-name=evaluate_model_compose       # Job name
#SBATCH --output=logs/eval_compose_%j.out       # Standard output log (%j inserts job ID)
#SBATCH --error=logs/eval_compose_%j.err        # Standard error log
#SBATCH --nodes=1                       # Run all tasks on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # CPU cores (evaluation needs modest compute)
#SBATCH --mem=32G                       # Memory (adjust if evaluating millions of nodes)
#SBATCH --time=2-02:00:00                 # Time limit (2 hours is usually plenty)
#SBATCH --partition=gpu             # Partition name (use 'gpu' if required by your system)
#SBATCH --gres=gpu:1                  # Uncomment only if your environment strictly requires a GPU

# 1. Create a logs directory if it doesn't exist
mkdir -p logs

# Always run from the repo root regardless of where sbatch was invoked from
cd "$SLURM_SUBMIT_DIR" || exit 1

# 2. Clean PyCache to ensure clean imports
find . -type d -name "__pycache__" -exec rm -r {} +

# 3. Load modules and activate environment (uncomment and customize as needed)
# module load cuda/12.1
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate your_env_name

# 4. Run the evaluation script (simple evaluator -- ROC-AUC / PR-AUC, baseline vs full model)
export RUN_LAMBDA_ANCHOR=0.3
export RUN_ALIGN_LR=0.0003
export RUN_SEED=0
# python models/claude_active/evaluate/evaluate.py

# # For the statistically rigorous version (bootstrap CI + p-value), run instead/also:
python models/claude_active/evaluate/compose_based/evaluate.py