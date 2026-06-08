#!/bin/bash
#SBATCH --job-name=eval_ssl_orig
#SBATCH --output=logs/eval_ssl_orig_%j.log
#SBATCH --error=logs/eval_ssl_orig_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Quick eval for SSL orig checkpoint (job 1114 timed out before eval)
echo "=== SATLIB Test ==="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_ssl_orig/best.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type ssl \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_ssl_orig_satlib \
    --output improvements/results/cmp_ssl_orig_satlib.json

echo "=== ER-700-800 Test ==="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_ssl_orig/best.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type ssl \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_ssl_orig_er700 \
    --output improvements/results/cmp_ssl_orig_er700.json

echo "Done at $(date)"
