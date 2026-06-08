#!/bin/bash
#SBATCH --job-name=p_focal_sl
#SBATCH --output=logs/pilot_focal_sl_%j.log
#SBATCH --error=logs/pilot_focal_sl_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="pilot,sl,satlib,focal"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Focal loss (γ=2, α=0.25) replacing weighted BCE
python -u -m experiments.train_sl_pilot \
    --batch_size 16 --epochs 20 --max_shards 50 \
    --use_pe 0 --use_enhanced_features 0 \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --pos_weight 5.0 \
    --focal_gamma 2.0 --focal_alpha 0.25 \
    --checkpoint_path checkpoints/pilots/focal_sl \
    --run_name pilot_sl_satlib_focal

echo "Done at $(date)"
