#!/bin/bash
#SBATCH --job-name=p1_runA
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase1_runA_%j.log
#SBATCH --error=logs/phase1_runA_%j.err
#SBATCH --time=02:00:00

# Phase 1 Run A: selection_weight=0, feasibility_weight=2
# Expect: feasibility_loss → 0, selection_loss unconstrained
export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis_ssl.py \
    --epochs 20 \
    --use_pe 0 \
    --use_enhanced_features 0 \
    --selection_weight 0.0 \
    --feasibility_weight 2.0 \
    --run_name "phase1_runA_sw0_fw2" \
    --max_shards 10
