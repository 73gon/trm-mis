#!/bin/bash
#SBATCH --job-name=sup_finetune
#SBATCH --output=logs/sup_finetune_%j.log
#SBATCH --error=logs/sup_finetune_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Fine-tune from SSL checkpoint (epoch 499, best SSL model)
# SSL model already achieves 0.9651 PP approx ratio
# Supervised fine-tuning should improve node discrimination
python -u train_mis.py \
    --max_shards 10 \
    --batch_size 16 --epochs 50 \
    --pretrained checkpoints/mis_ssl/epoch_499.pt \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/mis_finetune \
    --run_name sup_finetune_ssl499_10sh
