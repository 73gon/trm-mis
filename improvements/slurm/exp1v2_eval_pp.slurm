#!/bin/bash
#SBATCH --job-name=eval_pp2
#SBATCH --output=logs/eval_pp2_%j.log
#SBATCH --error=logs/eval_pp2_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Experiment 1 (v2): Faster evaluation with proper checkpoint detection
# Uses max 2 shards (~500 graphs) for quick comparison, 8 samples
export PYTHONUNBUFFERED=1
echo "========================================="
echo "Experiment 1v2: Enhanced Post-Processing"
echo "Starting on $(hostname) at $(date)"
echo "========================================="
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

mkdir -p improvements/results

# Test 1a: Fine-tuned checkpoint (input_dim=2, works directly)
echo ""
echo "=== Test 1a: Fine-tuned on SATLIB val (8 samples) ==="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/mis_finetune/epoch_49.pt \
    --data_path data/difusco_benchmark/datasets/satlib/train \
    --model_type ssl \
    --n_samples 8 \
    --batch_size 32 \
    --max_shards 1 \
    --output improvements/results/eval_finetune_satlib_8s.json

# Test 1b: Fine-tuned high pos_weight version
echo ""
echo "=== Test 1b: Fine-tuned high pw on SATLIB val (8 samples) ==="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/mis_ft_high/epoch_49.pt \
    --data_path data/difusco_benchmark/datasets/satlib/train \
    --model_type ssl \
    --n_samples 8 \
    --batch_size 32 \
    --max_shards 1 \
    --output improvements/results/eval_ft_high_satlib_8s.json

# Test 1c: SSL epoch_499 with proper feature detection
echo ""
echo "=== Test 1c: SSL epoch_499 on SATLIB val (8 samples) ==="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/mis_ssl/epoch_499.pt \
    --data_path data/difusco_benchmark/datasets/satlib/train \
    --model_type ssl \
    --n_samples 8 \
    --batch_size 32 \
    --max_shards 1 \
    --output improvements/results/eval_ssl499_satlib_8s.json

echo ""
echo "Experiment 1v2 complete at $(date)"
