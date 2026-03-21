#!/bin/bash
#SBATCH --job-name=sup_ft_high
#SBATCH --output=logs/sup_ft_high_%j.log
#SBATCH --error=logs/sup_ft_high_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Fine-tune from SSL + higher pos_weight to strongly select MIS nodes
python -u train_mis.py \
    --max_shards 10 \
    --batch_size 16 --epochs 50 \
    --pretrained checkpoints/mis_ssl/epoch_499.pt \
    --pos_weight 5.0 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/mis_ft_high \
    --run_name sup_ft_high5_ssl499_10sh
