#!/bin/bash

#SBATCH --job-name=evaluate_model       # Job name
#SBATCH --output=logs/eval_%j.out       # Standard output log (%j inserts job ID)
#SBATCH --error=logs/eval_%j.err        # Standard error log
#SBATCH --nodes=1                       # Run all tasks on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # CPU cores (evaluation needs modest compute)
#SBATCH --mem=32G                       # Memory (adjust if evaluating millions of nodes)
#SBATCH --time=02:00:00                 # Time limit (2 hours is usually plenty)
#SBATCH --partition=cpu             # Partition name (use 'gpu' if required by your system)
# #SBATCH --gres=gpu:1                  # Uncomment only if your environment strictly requires a GPU

# 1. Create a logs directory if it doesn't exist
mkdir -p logs

# 2. Clean PyCache to ensure clean imports
find . -type d -name "__pycache__" -exec rm -r {} +

# 3. Load modules and activate environment (uncomment and customize as needed)
# module load cuda/12.1
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate your_env_name

# 4. Run the evaluation script
RUN_SEED=0 python evaluate.py