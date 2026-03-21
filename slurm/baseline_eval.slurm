#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/baseline_eval_%j.log
#SBATCH --error=logs/baseline_eval_%j.err
#SBATCH --time=01:00:00


# Phase A: Baseline evaluation — trivial strategies vs trained models
export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u eval/baseline_comparison.py \
    --max_shards 10 \
    --ssl_checkpoint checkpoints/mis_ssl/epoch_49.pt \
    --sup_checkpoint checkpoints/mis/epoch_49.pt \
    --n_random_trials 5
