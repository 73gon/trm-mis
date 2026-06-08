#!/bin/bash
#SBATCH --job-name=ovf_sat10
#SBATCH --output=logs/overfit_sl_satlib_10_%j.log
#SBATCH --error=logs/overfit_sl_satlib_10_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="overfit,sl,satlib,N10,stageB"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

CHKPT_DIR=checkpoints/overfit_sl_satlib_10
mkdir -p "$CHKPT_DIR"

GPU_LOG="logs/overfit_sl_satlib_10_${SLURM_JOB_ID}_gpu.log"
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

# Stage B: overfit on 10 SATLIB graphs simultaneously (single batch of 10).
# Success gate: raw_AR averaged across all 10 >= 0.99, raw_feasibility == 1.0.
python -u -m experiments.overfit_sl \
    --data_path data/difusco_benchmark/datasets/satlib/train \
    --num_graphs 10 \
    --iterations 4000 \
    --lr 1e-3 \
    --dropout 0.0 \
    --use_pe 1 --use_enhanced_features 1 --precomputed 1 \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --n_supervision 1 \
    --log_every 20 \
    --checkpoint_path "$CHKPT_DIR" \
    --run_name overfit_sl_satlib_10 \
    --wandb

echo "Done at $(date)"
echo "=== GPU util summary ==="
awk -F'util_mem=' '{split($2,a,","); sum+=a[1]; n++} END {if(n) printf "avg=%.1f%%  samples=%d\n", sum/n, n}' "$GPU_LOG" 2>/dev/null || true
