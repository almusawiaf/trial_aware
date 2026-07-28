#!/bin/bash

#SBATCH --job-name=trial_json       # Job name
#SBATCH --output=logs/eval_%j.out       # Standard output log (%j inserts job ID)
#SBATCH --error=logs/eval_%j.err        # Standard error log
#SBATCH --nodes=1                       # Run all tasks on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # CPU cores (evaluation needs modest compute)
#SBATCH --mem=32G                       # Memory (adjust if evaluating millions of nodes)
#SBATCH --time=02:00:00                 # Time limit (2 hours is usually plenty)
#SBATCH --partition=cpu             # Partition name (use 'gpu' if required by your system)

mkdir -p logs

find . -type d -name "__pycache__" -exec rm -r {} +


python generate_trial_json.py