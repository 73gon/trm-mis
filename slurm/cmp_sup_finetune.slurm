#!/bin/bash
#SBATCH --job-name=cmp_sup_ft
#SBATCH --output=logs/cmp_sup_ft_%j.log
#SBATCH --error=logs/cmp_sup_ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# ============================================================
# Comparison run: Supervised fine-tuned from SSL
# Fast mode: no PE, no enhanced features
# First trains SSL 100 epochs, then fine-tunes supervised 50 epochs
# ============================================================

echo "=== Phase 1: SSL pre-training (100 epochs) ==="
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 100 \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/cmp_ssl_pretrain \
    --run_name cmp_ssl_pretrain

echo "=== Phase 2: Supervised fine-tuning (50 epochs) ==="
python -u train_mis.py \
    --batch_size 16 --epochs 50 \
    --use_pe 0 --use_enhanced_features 0 \
    --pretrained checkpoints/cmp_ssl_pretrain/best.pt \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --pos_weight 5.0 \
    --checkpoint_path checkpoints/cmp_sup_ft \
    --run_name cmp_sup_finetune

echo "=== Training complete. Starting eval ==="

# Eval on SATLIB test
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_sup_ft/best.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type supervised \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_sup_ft_satlib \
    --output improvements/results/cmp_sup_ft_satlib.json

# Eval on ER-700-800 test
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_sup_ft/best.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type supervised \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_sup_ft_er700 \
    --output improvements/results/cmp_sup_ft_er700.json

echo "Done at $(date)"
