#!/bin/bash
#SBATCH --job-name=sup_autoweight
#SBATCH --output=logs/sup_autoweight_%j.log
#SBATCH --error=logs/sup_autoweight_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Key fix: pos_weight=None (auto from dataset, ~2.0)
# Previous runs had pos_weight=1.0 (no class weighting) → too conservative
python -u train_mis.py \
    --max_shards 10 \
    --batch_size 16 --epochs 50 \
    --use_pe 0 --use_enhanced_features 0 \
    --run_name sup_autoweight_10sh
