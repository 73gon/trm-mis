#!/bin/bash
#SBATCH --job-name=p_tempa_ssl
#SBATCH --output=logs/pilot_tempanneal_ssl_%j.log
#SBATCH --error=logs/pilot_tempanneal_ssl_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="pilot,ssl,satlib,temp_anneal"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Temperature annealing: start soft (2.0), sharpen to crisp (0.5)
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 20 --max_shards 50 \
    --use_pe 0 --use_enhanced_features 0 \
    --temp_start 2.0 --temp_end 0.5 \
    --checkpoint_path checkpoints/pilots/tempanneal_ssl \
    --run_name pilot_ssl_satlib_temp_anneal

echo "Done at $(date)"
