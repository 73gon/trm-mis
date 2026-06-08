#!/bin/bash -l
#SBATCH --job-name=decbase_er
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "Starting ${SLURM_JOB_NAME} on $(hostname) at $(date)"

python -u -m experiments.decoder_baseline \
    --eval_data_path data/difusco_benchmark/datasets/er_700_800/test \
    --eval_num_graphs 128 \
    --eval_batch_size 4 \
    --use_pe 1 \
    --use_enhanced_features 1 \
    --pe_dim 16 \
    --label ER

echo "Done at $(date)"
