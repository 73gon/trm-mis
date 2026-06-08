#!/bin/bash -l
#SBATCH --job-name=smallmis_n300_v8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=sarmatia
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

RUN_NAME=${RUN_NAME:-smallmis_n300_v8}
DATA_PATH=${DATA_PATH:-data/smallmis_n300_multi/train}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-data/smallmis_n300_multi/test}
NUM_GRAPHS=${NUM_GRAPHS:-3000}
ITERATIONS=${ITERATIONS:-160000}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
LR=${LR:-3e-4}
POS_WEIGHT=${POS_WEIGHT:-1.0}
FEAS_W=${FEAS_W:-2.0}
EVAL_EVERY=${EVAL_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-50}
PRECISION=${PRECISION:-bf16}
OPTIMIZER=${OPTIMIZER:-adamw}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
DROPOUT=${DROPOUT:-0.15}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.0}
EVAL_LOSS_PATIENCE=${EVAL_LOSS_PATIENCE:-25}
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "Starting ${RUN_NAME} on $(hostname) at $(date)"
echo "Train=${DATA_PATH} Eval=${EVAL_DATA_PATH}"
echo "N=${NUM_GRAPHS} iters=${ITERATIONS} batch=${BATCH_SIZE} lr=${LR} pos_w=${POS_WEIGHT} precision=${PRECISION} optim=${OPTIMIZER}"
nvidia-smi -L || true
free -g | head -2

python -u -m experiments.overfit_sl_pe \
    --data_path "${DATA_PATH}" \
    --eval_data_path "${EVAL_DATA_PATH}" \
    --num_graphs "${NUM_GRAPHS}" \
    --iterations "${ITERATIONS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --dropout "${DROPOUT}" \
    --label_smoothing "${LABEL_SMOOTHING}" \
    --pos_weight "${POS_WEIGHT}" \
    --feasibility_weight "${FEAS_W}" \
    --eval_every "${EVAL_EVERY}" \
    --eval_max_steps 12 \
    --log_every "${LOG_EVERY}" \
    --precision "${PRECISION}" \
    --optimizer "${OPTIMIZER}" \
    --gpu_util_poll \
    --decoder_baseline degree \
    --benchmark_md docs/BENCHMARK.md \
    --benchmark_dataset_label "smallmis_n300" \
    --checkpoint_path "checkpoints/${RUN_NAME}" \
    --run_name "${RUN_NAME}" \
    --wandb \
    --eval_loss_patience "${EVAL_LOSS_PATIENCE}" \
    ${EXTRA_ARGS}

echo "Done at $(date)"
