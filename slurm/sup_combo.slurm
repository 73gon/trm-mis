#!/bin/bash
#SBATCH --job-name=sup_combo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sup_combo_%j.log
#SBATCH --error=logs/sup_combo_%j.err
#SBATCH --time=04:00:00


# Combined: Label smoothing + hybrid SSL loss
export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis.py \
    --max_shards 10 \
    --batch_size 16 --epochs 50 \
    --selection_weight 2.0 \
    --label_smoothing 0.1 \
    --feasibility_weight 1.0 \
    --feasibility_loss_type log_barrier \
    --use_pe 0 --use_enhanced_features 0 \
    --run_name sup_combo_10sh
