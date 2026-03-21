#!/bin/bash
#SBATCH --job-name=ssl_pignn1
#SBATCH --output=logs/ssl_pignn1_%j.log
#SBATCH --error=logs/ssl_pignn1_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# PI-GNN style loss: linear selection + quadratic feasibility
# fw=100 (heavy penalty like PI-GNN paper), sw=1
# This avoids log saturation and exponential overflow of default SSL
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 100 \
    --loss_mode pi_gnn \
    --feasibility_weight 100.0 \
    --selection_weight 1.0 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/ssl_pignn1 \
    --run_name ssl_pignn_fw100_sw1
