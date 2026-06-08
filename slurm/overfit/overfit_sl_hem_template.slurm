#!/bin/bash -l
#SBATCH --job-name=overfit_hem
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "Starting ${SLURM_JOB_NAME} on $(hostname) at $(date)"
echo "Init from:  ${INIT_FROM}"
echo "Data path:  ${DATA_PATH}"
echo "Run name:   ${RUN_NAME}"
echo "alpha=${HEM_ALPHA:-2.0} iters=${ITERATIONS:-100000} N=${NUM_GRAPHS:-25000}"

python -u -m experiments.overfit_sl_hem \
    --data_path "${DATA_PATH}" \
    --num_graphs "${NUM_GRAPHS:-25000}" \
    --iterations "${ITERATIONS:-100000}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --lr "${LR:-1e-4}" \
    --pos_weight "${POS_WEIGHT:-5.0}" \
    --feasibility_weight "${FEAS_W:-2.0}" \
    --init_from "${INIT_FROM}" \
    --hem_alpha "${HEM_ALPHA:-2.0}" \
    --hem_eval_every "${HEM_EVAL_EVERY:-50000}" \
    --checkpoint_path "checkpoints/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    --wandb \
    ${EXTRA_ARGS:-}

echo "Done at $(date)"
