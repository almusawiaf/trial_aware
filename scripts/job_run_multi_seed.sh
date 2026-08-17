#!/bin/bash

#SBATCH --job-name=multi_seed           # Job name
#SBATCH --output=logs/multiseed_%j.out  # Standard output log (%j inserts job ID)
#SBATCH --error=logs/multiseed_%j.err   # Standard error log
#SBATCH --nodes=1                       # Run all tasks on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=32              # Number of CPU cores per task
#SBATCH --mem=256G                      # Total memory
#SBATCH --time=08:00:00                 # 8 hours -- 5 seeds x (~13 min train + ~1 min eval)
#SBATCH --partition=gpu                 # Partition/queue name
#SBATCH --gres=gpu:1                    # Request 1 GPU

mkdir -p logs

# Always run from the repo root regardless of where sbatch was invoked from
cd "$SLURM_SUBMIT_DIR" || exit 1

# module load cuda/12.1
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate your_env_name

find . -type d -name "__pycache__" -exec rm -r {} +
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python models/claude_active/run_multi_seed.py --seeds 0 1 2 3 4