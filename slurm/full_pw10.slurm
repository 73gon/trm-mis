#!/bin/bash
#SBATCH --job-name=full_pw10
#SBATCH --output=logs/full_pw10_%j.log
#SBATCH --error=logs/full_pw10_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Sweep: pos_weight=10.0 (explore even higher)
# Full SATLIB: 158 shards (~39,500 graphs), 100 epochs
python -u train_mis.py \
    --batch_size 16 --epochs 100 \
    --pos_weight 10.0 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/mis_full_pw10 \
    --run_name full_pw10_158sh_100ep
