#!/bin/bash
#SBATCH --job-name=ssl_pignn2
#SBATCH --output=logs/ssl_pignn2_%j.log
#SBATCH --error=logs/ssl_pignn2_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# PI-GNN style loss with degree-weighted selection
# fw=50, sw=1, degree_weighted=1
# Low-degree nodes get stronger selection signal → breaks symmetry
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 100 \
    --loss_mode pi_gnn \
    --feasibility_weight 50.0 \
    --selection_weight 1.0 \
    --degree_weighted 1 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/ssl_pignn2 \
    --run_name ssl_pignn_fw50_degw
