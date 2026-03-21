#!/bin/bash
#SBATCH --job-name=sup_lb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/full_sup_logbarrier_%j.log
#SBATCH --error=logs/full_sup_logbarrier_%j.err
#SBATCH --time=24:00:00


# Full supervised training with log-barrier feasibility loss on SATLIB
# - All 158 shards (~39500 graphs)
# - 50 epochs with log-barrier feasibility, fw=1.0
# - Features disabled for speed (pe=0, enhanced=0)
# - wandb logging enabled

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis.py \
    --epochs 50 \
    --use_pe 0 \
    --use_enhanced_features 0 \
    --feasibility_weight 1.0 \
    --feasibility_loss_type log_barrier \
    --run_name "full_sup_logbarrier_fw1.0_satlib"
