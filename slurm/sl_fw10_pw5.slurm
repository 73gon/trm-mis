#!/bin/bash
#SBATCH --job-name=sl_fw10_pw5
#SBATCH --output=logs/sl_fw10_pw5_%j.log
#SBATCH --error=logs/sl_fw10_pw5_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Supervised: feasibility_weight=10.0, pos_weight=5.0
# High feasibility weight to strongly penalize the "select everything" strategy
# while pos_weight=5 still encourages finding MIS nodes
python -u train_mis.py \
    --batch_size 16 --epochs 100 \
    --pos_weight 5.0 \
    --feasibility_weight 10.0 \
    --feasibility_loss_type hinge \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/sl_fw10_pw5 \
    --run_name sl_fw10_pw5_158sh
