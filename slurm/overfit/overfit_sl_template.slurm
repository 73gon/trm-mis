#!/bin/bash
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Required env vars passed via `sbatch --export`:
#   DATASET      (satlib | er_700_800)
#   NUM_GRAPHS   (e.g. 1, 250, 2500, ...)
#   ITERATIONS   (e.g. 3000)
#   STAGE_TAG    (e.g. stage2_250)
#   RUN_NAME     (wandb name, e.g. overfit_sl_satlib_250)
#   CHKPT_DIR    (checkpoint dir)
#   ITER_BUDGET_TAG (for wandb tag, e.g. 3k)
# Optional:
#   EXTRA_ARGS   (extra flags for overfit_sl, e.g. "--batch_size 50")

export PYTHONUNBUFFERED=1
export WANDB_TAGS="overfit,sl,${DATASET},${STAGE_TAG},${ITER_BUDGET_TAG}"
echo "Starting ${RUN_NAME} on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

mkdir -p "${CHKPT_DIR}"

GPU_LOG="logs/${RUN_NAME}_${SLURM_JOB_ID}_gpu.log"
(
  while true; do
    ts=$(date +'%Y-%m-%d %H:%M:%S')
    util=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n 1)
    echo "$ts  util_mem=$util" >> "$GPU_LOG"
    sleep 30
  done
) &
GPU_WATCHER_PID=$!
trap "kill $GPU_WATCHER_PID 2>/dev/null" EXIT

python -u -m experiments.overfit_sl \
    --data_path "data/difusco_benchmark/datasets/${DATASET}/train" \
    --num_graphs "${NUM_GRAPHS}" \
    --iterations "${ITERATIONS}" \
    --lr "${LR:-1e-3}" --dropout 0.0 \
    --use_pe 1 --use_enhanced_features 1 --precomputed 1 \
    --feasibility_weight "${FEAS_W:-2.0}" --feasibility_loss_type hinge \
    --n_supervision 1 \
    --log_every 20 \
    --checkpoint_path "${CHKPT_DIR}" \
    --run_name "${RUN_NAME}" \
    --wandb \
    ${EXTRA_ARGS}

RC=$?
echo "overfit_sl exited rc=${RC}"
echo "Done at $(date)"
awk -F'util_mem=' '{split($2,a,","); sum+=a[1]; n++} END {if(n) printf "GPU avg=%.1f%% samples=%d\n", sum/n, n}' "$GPU_LOG" 2>/dev/null || true
exit $RC
