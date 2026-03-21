#!/bin/bash
#SBATCH --job-name=sup_auto_sel
#SBATCH --output=logs/sup_auto_sel_%j.log
#SBATCH --error=logs/sup_auto_sel_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Auto pos_weight + selection loss to push probabilities up
python -u train_mis.py \
    --max_shards 10 \
    --batch_size 16 --epochs 50 \
    --selection_weight 2.0 \
    --use_pe 0 --use_enhanced_features 0 \
    --run_name sup_auto_sel2_10sh
