#!/bin/bash -l
#SBATCH --job-name=eval_overfit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "Starting ${SLURM_JOB_NAME} on $(hostname) at $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data path:  ${DATA_PATH}"
echo "Run name:   ${RUN_NAME}"

python -u -m experiments.eval_overfit_ckpt \
    --checkpoint "${CHECKPOINT}" \
    --data_path "${DATA_PATH}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --num_graphs "${NUM_GRAPHS:-10000000}" \
    --output_json "logs/${RUN_NAME}_${SLURM_JOB_ID}.json" \
    --run_name "${RUN_NAME}" \
    --wandb \
    ${EXTRA_ARGS:-}

echo "Done at $(date)"
