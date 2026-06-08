#!/bin/bash
#SBATCH --job-name=imp_ft
#SBATCH --output=logs/imp_ft_%j.log
#SBATCH --error=logs/imp_ft_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Experiment 3: Fine-tune improved SSL → Supervised
# Uses the best improved SSL checkpoint as starting point
# Then applies supervised BCE fine-tuning with labels
export PYTHONUNBUFFERED=1
echo "========================================="
echo "Experiment 3: Supervised Fine-tuning from Improved SSL"
echo "Starting on $(hostname) at $(date)"
echo "========================================="
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis.py \
    --max_shards 10 \
    --batch_size 16 --epochs 50 \
    --pretrained checkpoints/improved_ssl/best.pt \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/improved_finetune \
    --run_name imp_finetune_from_improved_ssl

echo ""
echo "Experiment 3 complete at $(date)"
