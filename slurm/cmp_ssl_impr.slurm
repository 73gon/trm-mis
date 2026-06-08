#!/bin/bash
#SBATCH --job-name=cmp_ssl_impr
#SBATCH --output=logs/cmp_ssl_impr_%j.log
#SBATCH --error=logs/cmp_ssl_impr_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# ============================================================
# Comparison run: Improved SSL (Lagrangian dual ascent)
# Fast mode: no PE, no enhanced features
# log_barrier loss, Lagrangian auto-balancing, warmup
# ============================================================
python -u improvements/train_ssl_improved.py \
    --batch_size 16 --epochs 100 \
    --use_pe 0 --use_enhanced_features 0 \
    --feasibility_weight 2.0 --selection_weight 5.0 \
    --feasibility_loss_type log_barrier \
    --use_lagrangian 1 --lagrangian_lr 0.01 --lagrangian_target 0.05 \
    --use_feasibility_warmup 1 --warmup_fraction 0.2 \
    --checkpoint_path checkpoints/cmp_ssl_impr \
    --run_name cmp_ssl_improved

echo "=== Training complete. Starting eval ==="

# Eval on SATLIB test
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_ssl_impr/best.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type ssl \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_ssl_impr_satlib \
    --output improvements/results/cmp_ssl_impr_satlib.json

# Eval on ER-700-800 test
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_ssl_impr/best.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type ssl \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_ssl_impr_er700 \
    --output improvements/results/cmp_ssl_impr_er700.json

echo "Done at $(date)"
