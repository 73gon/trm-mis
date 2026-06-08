#!/bin/bash -l
#SBATCH --job-name=overfit_pe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "Starting ${SLURM_JOB_NAME} on $(hostname) at $(date)"
echo "Train data: ${DATA_PATH}"
echo "Eval data:  ${EVAL_DATA_PATH}"
echo "Run name:   ${RUN_NAME}"
echo "N=${NUM_GRAPHS:-25000} iters=${ITERATIONS:-400000} eval_every=${EVAL_EVERY:-5000}"

python -u -m experiments.overfit_sl_pe \
    --data_path "${DATA_PATH}" \
    --eval_data_path "${EVAL_DATA_PATH}" \
    --num_graphs "${NUM_GRAPHS:-25000}" \
    --iterations "${ITERATIONS:-400000}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --lr "${LR:-3e-4}" \
    --pos_weight "${POS_WEIGHT:-5.0}" \
    --feasibility_weight "${FEAS_W:-2.0}" \
    --eval_every "${EVAL_EVERY:-5000}" \
    --eval_max_steps "${EVAL_MAX_STEPS:-12}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
    --benchmark_md "${BENCHMARK_MD:-docs/BENCHMARK.md}" \
    --benchmark_dataset_label "${BENCHMARK_DATASET:-?}" \
    --checkpoint_path "checkpoints/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    --wandb \
    --disable_early_stop \
    ${EXTRA_ARGS:-}

echo "Done at $(date)"
