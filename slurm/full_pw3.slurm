#!/bin/bash
#SBATCH --job-name=full_pw3
#SBATCH --output=logs/full_pw3_%j.log
#SBATCH --error=logs/full_pw3_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Sweep: pos_weight=3.0 (between auto~2.0 and best quick=5.0)
# Full SATLIB: 158 shards (~39,500 graphs), 100 epochs
python -u train_mis.py \
    --batch_size 16 --epochs 100 \
    --pos_weight 3.0 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/mis_full_pw3 \
    --run_name full_pw3_158sh_100ep
