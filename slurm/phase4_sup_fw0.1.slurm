#!/bin/bash
#SBATCH --job-name=p4_fw01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase4_sup_fw0.1_%j.log
#SBATCH --error=logs/phase4_sup_fw0.1_%j.err
#SBATCH --time=04:00:00

# Phase 4: Supervised with feasibility_weight=0.1
export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis.py \
    --epochs 50 \
    --use_pe 0 \
    --use_enhanced_features 0 \
    --feasibility_weight 0.1 \
    --run_name "phase4_sup_fw0.1" \
    --max_shards 10
