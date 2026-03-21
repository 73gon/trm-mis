#!/bin/bash
#SBATCH --job-name=ssl_pignn3
#SBATCH --output=logs/ssl_pignn3_%j.log
#SBATCH --error=logs/ssl_pignn3_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# PI-GNN combo: degree-weighted + noise + higher LR
# Best combination of all tricks
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 100 \
    --loss_mode pi_gnn \
    --feasibility_weight 100.0 \
    --selection_weight 1.0 \
    --degree_weighted 1 \
    --noise_scale 0.5 \
    --lr 5e-4 \
    --grad_clip 2.0 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/ssl_pignn3 \
    --run_name ssl_pignn_combo
