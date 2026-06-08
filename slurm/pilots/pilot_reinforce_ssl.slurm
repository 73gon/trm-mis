#!/bin/bash
#SBATCH --job-name=p_reinf_ssl
#SBATCH --output=logs/pilot_reinforce_ssl_%j.log
#SBATCH --error=logs/pilot_reinforce_ssl_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="pilot,ssl,satlib,reinforce"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# REINFORCE-style SSL loss (Bernoulli sampling + reward)
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 20 --max_shards 50 \
    --use_pe 0 --use_enhanced_features 0 \
    --loss_mode reinforce \
    --checkpoint_path checkpoints/pilots/reinforce_ssl \
    --run_name pilot_ssl_satlib_reinforce

echo "Done at $(date)"
