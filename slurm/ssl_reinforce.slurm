#!/bin/bash
#SBATCH --job-name=ssl_reinf
#SBATCH --output=logs/ssl_reinf_%j.log
#SBATCH --error=logs/ssl_reinf_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# REINFORCE: sample discrete solutions, compute reward
# Fundamentally different from continuous relaxation
# 8 samples per graph, lr=5e-4 (higher for RL)
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 100 \
    --loss_mode reinforce \
    --reinforce_samples 8 \
    --lr 5e-4 \
    --grad_clip 1.0 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/ssl_reinforce \
    --run_name ssl_reinforce_k8
