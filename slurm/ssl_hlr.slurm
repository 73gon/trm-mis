#!/bin/bash
#SBATCH --job-name=ssl_hlr
#SBATCH --output=logs/ssl_hlr_%j.log
#SBATCH --error=logs/ssl_hlr_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Default SSL loss but with higher LR and less grad clipping
# Previous: lr=1e-4, grad_clip=0.5
# Now: lr=1e-3, grad_clip=5.0 → faster escape from local minimum
# Plus heavy feasibility weight to force differentiation
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 100 \
    --lr 1e-3 \
    --grad_clip 5.0 \
    --feasibility_weight 20.0 \
    --selection_weight 1.0 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/ssl_hlr \
    --run_name ssl_hlr_fw20_sw1
