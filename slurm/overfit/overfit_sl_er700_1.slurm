#!/bin/bash
#SBATCH --job-name=ovf_er1
#SBATCH --output=logs/overfit_sl_er700_1_%j.log
#SBATCH --error=logs/overfit_sl_er700_1_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="overfit,sl,er700,N1,stageA"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

CHKPT_DIR=checkpoints/overfit_sl_er700_1
mkdir -p "$CHKPT_DIR"

GPU_LOG="logs/overfit_sl_er700_1_${SLURM_JOB_ID}_gpu.log"
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

# Stage A on ER-700-800 (harder: sparser, bigger, ~6% positive class)
python -u -m experiments.overfit_sl \
    --data_path data/difusco_benchmark/datasets/er_700_800/train \
    --num_graphs 1 \
    --iterations 3000 \
    --lr 1e-3 \
    --dropout 0.0 \
    --use_pe 1 --use_enhanced_features 1 --precomputed 1 \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --n_supervision 1 \
    --log_every 10 \
    --checkpoint_path "$CHKPT_DIR" \
    --run_name overfit_sl_er700_1 \
    --wandb

echo "Done at $(date)"
echo "=== Last 20 GPU samples ==="
tail -n 20 "$GPU_LOG" 2>/dev/null || true
