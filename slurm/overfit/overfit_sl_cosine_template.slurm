#!/bin/bash
#SBATCH --job-name=sweep1_cosine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

GPU_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.log"
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
    --lr "${LR:-3e-4}" --dropout 0.0 \
    --use_pe 1 --use_enhanced_features 1 --precomputed 1 \
    --feasibility_weight "${FEAS_W:-2.0}" --feasibility_loss_type hinge \
    --n_supervision 1 \
    --batch_size 16 \
    --lr_schedule cosine_restart \
    --cosine_t0 "${COS_T0:-20000}" \
    --cosine_tmult "${COS_TMULT:-2}" \
    --cosine_eta_min "${COS_ETA:-1e-5}" \
    --pos_weight "${POS_W:-15.0}" \
    --log_every 20 \
    --checkpoint_path "${CHKPT_DIR}" \
    --run_name "${RUN_NAME}" \
    --wandb \
    ${EXTRA_ARGS}

RC=$?
echo "overfit_sl exited rc=${RC}"
echo "Done at $(date)"
