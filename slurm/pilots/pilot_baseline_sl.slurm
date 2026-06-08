#!/bin/bash
#SBATCH --job-name=p_base_sl
#SBATCH --output=logs/pilot_baseline_sl_%j.log
#SBATCH --error=logs/pilot_baseline_sl_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="pilot,sl,satlib,baseline"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# 20-epoch supervised baseline (identity of train_mis.py defaults)
python -u train_mis.py \
    --batch_size 16 --epochs 20 --max_shards 50 \
    --use_pe 0 --use_enhanced_features 0 \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --pos_weight 5.0 \
    --checkpoint_path checkpoints/pilots/baseline_sl \
    --run_name pilot_sl_satlib_baseline

echo "Done at $(date)"
