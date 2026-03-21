#!/bin/bash
#SBATCH --job-name=ssl_noise
#SBATCH --output=logs/ssl_noise_%j.log
#SBATCH --error=logs/ssl_noise_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# PI-GNN + exploration noise: breaks the equilibrium
# Noise forces different probs on different nodes
# fw=100, sw=1, noise=1.0
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 100 \
    --loss_mode pi_gnn \
    --feasibility_weight 100.0 \
    --selection_weight 1.0 \
    --noise_scale 1.0 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/ssl_noise \
    --run_name ssl_pignn_noise1
