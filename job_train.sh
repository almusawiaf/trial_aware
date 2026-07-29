#!/bin/bash

#SBATCH --job-name=graph_cleaning      # Job name
#SBATCH --output=logs/run_%j.out       # Standard output log (%j inserts job ID)
#SBATCH --error=logs/run_%j.err        # Standard error log
#SBATCH --nodes=1                      # Run all tasks on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=32              # Number of CPU cores per task
#SBATCH --mem=256G                      # Total memory (e.g., 32GB, adjust based on graph size)
#SBATCH --time=4-00:00:00                # Time limit hrs:min:sec (e.g., 4 hours)
#SBATCH --partition=gpu                # Partition/queue name (e.g., gpu, compute, etc.)
#SBATCH --gres=gpu:1                   # Request 1 GPU (remove if running strictly on CPU)

# 1. Create a logs directory if it doesn't exist
mkdir -p logs

# 2. Load necessary cluster modules (e.g., CUDA if using GPU)
# module load cuda/12.1                # Uncomment and adjust version if needed

# 3. Initialize and activate your conda environment
# Replace 'base' or path below with the name of your specific conda environment
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate your_env_name

# export GEMINI_API_KEY="AQ.Ab8RN6Lqny2NU-RqFdgi61sQbCE9yP2gHc-KBG1uyMe-m7b2vA"

# 4. Run the phase 1 + 2 cleaning and graph building script
# python run.py

find . -type d -name "__pycache__" -exec rm -r {} +
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py