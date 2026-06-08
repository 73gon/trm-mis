#!/bin/bash
#SBATCH --job-name=cmp_ssl_orig
#SBATCH --output=logs/cmp_ssl_orig_%j.log
#SBATCH --error=logs/cmp_ssl_orig_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# ============================================================
# Comparison run: Original SSL (train_mis_ssl.py)
# Fast mode: no PE, no enhanced features
# Default loss: exponential, mu=5, fw=2, sw=5
# ============================================================
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 100 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/cmp_ssl_orig \
    --run_name cmp_ssl_orig

echo "=== Training complete. Starting eval ==="

# Eval on SATLIB test with wandb logging
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_ssl_orig/best.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type ssl \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_ssl_orig_satlib \
    --output improvements/results/cmp_ssl_orig_satlib.json

# Eval on ER-700-800 test with wandb logging
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_ssl_orig/best.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type ssl \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_ssl_orig_er700 \
    --output improvements/results/cmp_ssl_orig_er700.json

echo "Done at $(date)"
