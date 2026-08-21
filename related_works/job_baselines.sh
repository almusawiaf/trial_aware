#!/bin/bash
#SBATCH --job-name=ta_baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# GPU is optional. The GNNs and MLPs run fine on CPU at this graph size
# (~10^4 nodes); request one only if your cohort is much larger.
##SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs results

module load anaconda3 2>/dev/null || true
source activate trial_aware

export TA_NJOBS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# --- point these at your pipeline outputs ----------------------------------
MIMIC_DIR=${MIMIC_DIR:-/lustre/home/almusawiaf/PhD_Projects/Trial_Aware_2/data}
TRIALS_DIR=${TRIALS_DIR:-/lustre/home/almusawiaf/PhD_Projects/Trial_Aware_2/data/10000_trials}

echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  started=$(date)"
python -c "import torch, xgboost, sklearn; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"

# ---------------------------------------------------------------------------
# Stage 1 -- label audit and a single fast seed. Read results/results.md before
# committing to the full sweep: if the ORACLE and HONEST regimes are not far
# apart, something is leaking and the long run would only waste the allocation.
# ---------------------------------------------------------------------------
python run_all.py \
    --data real \
    --mimic-dir "$MIMIC_DIR" \
    --trials-dir "$TRIALS_DIR" \
    --models all \
    --regimes honest oracle \
    --seeds 0 \
    --no-tune \
    --bootstrap 200 \
    --out results/stage1_quick

# ---------------------------------------------------------------------------
# Stage 2 -- the reportable run: five seeds, equal 20-draw tuning budget for
# every model, full bootstrap.
# ---------------------------------------------------------------------------
python run_all.py \
    --data real \
    --mimic-dir "$MIMIC_DIR" \
    --trials-dir "$TRIALS_DIR" \
    --models all \
    --regimes honest oracle \
    --seeds 0 1 2 3 4 \
    --tune-trials 20 \
    --bootstrap 1000 \
    --reference "GraphSAGE" \
    --out results/stage2_full

echo "finished=$(date)"
echo "Report: results/stage2_full/results.md"
