#!/bin/bash
#SBATCH --job-name=ssl_lossch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ssl_losssched_%j.log
#SBATCH --error=logs/ssl_losssched_%j.err
#SBATCH --time=04:00:00


# B3: Loss weight scheduling — curriculum on fw/sw
# Start selection-heavy (sw=10, fw=0.5), shift to normal (sw=5, fw=2)
export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis_ssl.py \
    --max_shards 10 \
    --batch_size 16 --epochs 50 \
    --use_loss_schedule 1 \
    --fw_start 0.5 \
    --sw_start 10.0 \
    --feasibility_loss_type log_barrier \
    --use_pe 0 --use_enhanced_features 0 \
    --run_name ssl_losssched_10sh
