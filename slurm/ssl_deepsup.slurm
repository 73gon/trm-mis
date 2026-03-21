#!/bin/bash
#SBATCH --job-name=ssl_deepsup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ssl_deepsup_%j.log
#SBATCH --error=logs/ssl_deepsup_%j.err
#SBATCH --time=04:00:00


# B1: Deep supervision — loss at each H-cycle output
# More gradient signal to train intermediate representations
export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis_ssl.py \
    --max_shards 10 \
    --batch_size 16 --epochs 50 \
    --use_deep_supervision 1 \
    --feasibility_loss_type log_barrier \
    --use_pe 0 --use_enhanced_features 0 \
    --run_name ssl_deepsup_10sh
