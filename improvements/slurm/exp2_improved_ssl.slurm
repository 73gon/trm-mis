#!/bin/bash
#SBATCH --job-name=imp_ssl
#SBATCH --output=logs/imp_ssl_%j.log
#SBATCH --error=logs/imp_ssl_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Experiment 2: Improved SSL Training
# - Lagrangian dual ascent (auto-balance feasibility vs selection)
# - Feasibility warmup (ramp fw from 0 over first 20%)
# - Weight decay 1e-5
# - Log-barrier loss (best from experiments)
# - Starting from existing SSL checkpoint for faster convergence
export PYTHONUNBUFFERED=1
echo "========================================="
echo "Experiment 2: Improved SSL Training"
echo "Starting on $(hostname) at $(date)"
echo "========================================="
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u improvements/train_ssl_improved.py \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-5 \
    --grad_clip 1.0 \
    --weight_decay 1e-5 \
    --feasibility_weight 2.0 \
    --selection_weight 5.0 \
    --feasibility_loss_type log_barrier \
    --use_lagrangian 1 \
    --lagrangian_lr 0.01 \
    --lagrangian_target 0.05 \
    --use_feasibility_warmup 1 \
    --warmup_fraction 0.2 \
    --use_pe 0 --use_enhanced_features 0 \
    --max_shards 10 \
    --pretrained checkpoints/mis_ssl/epoch_499.pt \
    --checkpoint_path checkpoints/improved_ssl \
    --run_name improved_ssl_lagrangian

echo ""
echo "Experiment 2 complete at $(date)"
