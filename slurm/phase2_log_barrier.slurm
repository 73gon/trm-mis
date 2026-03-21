#!/bin/bash
#SBATCH --job-name=p2_log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase2_log_barrier_%j.log
#SBATCH --error=logs/phase2_log_barrier_%j.err
#SBATCH --time=04:00:00

# Phase 2: Log-barrier feasibility loss (professor's question)
export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis_ssl.py \
    --epochs 50 \
    --use_pe 0 \
    --use_enhanced_features 0 \
    --feasibility_loss_type log_barrier \
    --run_name "phase2_log_barrier" \
    --max_shards 10
