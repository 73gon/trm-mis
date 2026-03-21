#!/bin/bash
#SBATCH --job-name=p0_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase0_test_%j.log
#SBATCH --error=logs/phase0_test_%j.err
#SBATCH --time=00:30:00

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

python -u train_mis_ssl.py \
    --epochs 2 \
    --use_pe 0 \
    --use_enhanced_features 0 \
    --run_name "phase0_test_no_features" \
    --max_shards 10
