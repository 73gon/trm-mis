#!/bin/bash
#SBATCH --job-name=p_entr_ssl
#SBATCH --output=logs/pilot_entropy_ssl_%j.log
#SBATCH --error=logs/pilot_entropy_ssl_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="pilot,ssl,satlib,entropy"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Entropy regularization (push probs toward 0/1)
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 20 --max_shards 50 \
    --use_pe 0 --use_enhanced_features 0 \
    --entropy_weight 0.01 \
    --checkpoint_path checkpoints/pilots/entropy_ssl \
    --run_name pilot_ssl_satlib_entropy

echo "Done at $(date)"
