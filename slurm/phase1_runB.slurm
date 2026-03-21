#!/bin/bash
#SBATCH --job-name=p1_runB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase1_runB_%j.log
#SBATCH --error=logs/phase1_runB_%j.err
#SBATCH --time=02:00:00

# Phase 1 Run B: feasibility_weight=0, selection_weight=5
# Expect: selection_loss → 0 (all probs → 1), feasibility_loss unconstrained
export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis_ssl.py \
    --epochs 20 \
    --use_pe 0 \
    --use_enhanced_features 0 \
    --feasibility_weight 0.0 \
    --selection_weight 5.0 \
    --run_name "phase1_runB_fw0_sw5" \
    --max_shards 10
