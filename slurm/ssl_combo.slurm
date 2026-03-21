#!/bin/bash
#SBATCH --job-name=ssl_combo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ssl_combo_%j.log
#SBATCH --error=logs/ssl_combo_%j.err
#SBATCH --time=04:00:00


# Combined: Deep supervision + entropy reg + temp annealing
# All three mechanisms together to break equilibrium
export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis_ssl.py \
    --max_shards 10 \
    --batch_size 16 --epochs 50 \
    --use_deep_supervision 1 \
    --entropy_weight 0.5 \
    --temp_start 2.0 \
    --temp_end 0.5 \
    --feasibility_loss_type log_barrier \
    --use_pe 0 --use_enhanced_features 0 \
    --run_name ssl_combo_10sh
