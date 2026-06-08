#!/bin/bash
#SBATCH --job-name=sweep1_grok_N25k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs "${CHKPT_DIR}"
echo "Starting ${SLURM_JOB_NAME} on $(hostname) at $(date)"

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
    --batch_size 16 \
    --lr 3e-4 --dropout 0.0 \
    --use_pe 1 --use_enhanced_features 1 --precomputed 1 \
    --feasibility_weight 2.0 --feasibility_loss_type hinge \
    --pos_weight 5.0 \
    --n_supervision 1 \
    --log_every 20 \
    --checkpoint_path "${CHKPT_DIR}" \
    --run_name "${RUN_NAME}" \
    --wandb

RC=$?
echo "overfit_sl exited rc=${RC}"
echo "Done at $(date)"
GPU_AVG=$(awk -F'util_mem=' 'NF>1{split($2,a,","); s+=a[1]; n++} END{if(n>0)printf "%.1f", s/n}' "$GPU_LOG")
echo "GPU avg=${GPU_AVG}% samples=$(wc -l < "$GPU_LOG")"
