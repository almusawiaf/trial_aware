#!/bin/bash

#SBATCH --job-name=training
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

# module load cuda/12.1
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate your_env_name

# python data_pipeline/run.py     # phase 1+2 cleaning and graph building, if needed

find . -type d -name "__pycache__" -exec rm -r {} +
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python models/claude_active/train.py