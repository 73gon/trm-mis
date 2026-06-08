#!/bin/bash
#SBATCH --job-name=cmp_sup_scratch
#SBATCH --output=logs/cmp_sup_scratch_%j.log
#SBATCH --error=logs/cmp_sup_scratch_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# ============================================================
# Comparison run: Supervised from scratch (train_mis.py)
# Fast mode: no PE, no enhanced features
# BCE loss + hinge feasibility_weight=1, pos_weight=5
# ============================================================
python -u train_mis.py \
    --batch_size 16 --epochs 100 \
    --use_pe 0 --use_enhanced_features 0 \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --pos_weight 5.0 \
    --checkpoint_path checkpoints/cmp_sup_scratch \
    --run_name cmp_sup_scratch

echo "=== Training complete. Starting eval ==="

# Eval on SATLIB test
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_sup_scratch/best.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type supervised \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_sup_scratch_satlib \
    --output improvements/results/cmp_sup_scratch_satlib.json

# Eval on ER-700-800 test
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_sup_scratch/best.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type supervised \
    --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_sup_scratch_er700 \
    --output improvements/results/cmp_sup_scratch_er700.json

echo "Done at $(date)"
