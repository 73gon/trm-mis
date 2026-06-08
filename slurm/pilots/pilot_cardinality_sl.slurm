#!/bin/bash
#SBATCH --job-name=p_card_sl
#SBATCH --output=logs/pilot_cardinality_sl_%j.log
#SBATCH --error=logs/pilot_cardinality_sl_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="pilot,sl,satlib,cardinality"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Cardinality penalty: (Σ probs - opt_size)^2 / opt_size^2 per graph
python -u -m experiments.train_sl_pilot \
    --batch_size 16 --epochs 20 --max_shards 50 \
    --use_pe 0 --use_enhanced_features 0 \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --pos_weight 5.0 \
    --cardinality_weight 0.1 \
    --checkpoint_path checkpoints/pilots/cardinality_sl \
    --run_name pilot_sl_satlib_cardinality

echo "Done at $(date)"
