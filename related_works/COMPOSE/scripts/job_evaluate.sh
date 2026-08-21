#!/bin/bash

#SBATCH --job-name=compose_evaluate            # Job name
#SBATCH --output=logs/eval_compose_%j.out       # Standard output log (%j inserts job ID)
#SBATCH --error=logs/eval_compose_%j.err        # Standard error log
#SBATCH --nodes=1                       # Run all tasks on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # CPU cores (evaluation needs modest compute)
#SBATCH --mem=32G                       # Memory (adjust if evaluating many trials)
#SBATCH --time=02:00:00                 # Time limit (2 hours is usually plenty)
#SBATCH --partition=gpu                 # Partition name (use 'gpu' if required by your system)
#SBATCH --gres=gpu:1                    # Uncomment only if your environment strictly requires a GPU

# 1. Create a logs directory if it doesn't exist
mkdir -p logs

# IMPORTANT: submit this job from related_works/COMPOSE/, e.g.:
#     cd related_works/COMPOSE
#     sbatch scripts/job_evaluate.sh
# Same relative-path assumption as job_train.sh: `python evaluate.py`
# expects evaluate.py to sit directly in the current directory.
cd "$SLURM_SUBMIT_DIR" || exit 1

# 2. Clean PyCache to ensure clean imports
find . -type d -name "__pycache__" -exec rm -r {} +

# 3. Load modules and activate environment (uncomment and customize as needed)
# module load cuda/12.1
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate trial_aware

# 4. Run evaluation. MUST use the same RUN_SEED that scripts/job_train.sh
# used, so this job loads checkpoints/compose_seed{RUN_SEED}.pt and, if
# present, picks up ../../data/evaluation_results_seed{RUN_SEED}.json to
# print the full Stage A / Stage B / COMPOSE comparison table.
export RUN_SEED=${RUN_SEED:-0}

python evaluate.py

# 5. OPTIONAL: statistically rigorous 3-way bootstrap comparison against
# Stage A/B (needs the one-line change to
# models/claude_active/evaluate/compose_based/evaluate.py described in
# README.md -- uncomment once that's in place):
# python compare_models.py