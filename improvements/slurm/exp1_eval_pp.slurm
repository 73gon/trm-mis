#!/bin/bash
#SBATCH --job-name=eval_pp
#SBATCH --output=logs/eval_pp_%j.log
#SBATCH --error=logs/eval_pp_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Experiment 1: Evaluate existing best checkpoints with enhanced post-processing
# This needs NO retraining — purely inference-time improvement.
# Compares: baseline greedy vs enhanced greedy vs multi-threshold vs multi-sample
export PYTHONUNBUFFERED=1
echo "========================================="
echo "Experiment 1: Enhanced Post-Processing"
echo "Starting on $(hostname) at $(date)"
echo "========================================="
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

mkdir -p improvements/results

# Test 1a: SSL checkpoint (epoch 499) on SATLIB validation
echo ""
echo "=== Test 1a: SSL epoch_499 on SATLIB (16 samples) ==="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/mis_ssl/epoch_499.pt \
    --data_path data/difusco_benchmark/datasets/satlib/train \
    --model_type ssl \
    --n_samples 16 \
    --batch_size 16 \
    --max_shards 10 \
    --output improvements/results/eval_ssl499_satlib_16s.json

# Test 1b: Fine-tuned checkpoint on SATLIB validation
echo ""
echo "=== Test 1b: Fine-tuned on SATLIB (16 samples) ==="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/mis_finetune/epoch_49.pt \
    --data_path data/difusco_benchmark/datasets/satlib/train \
    --model_type ssl \
    --n_samples 16 \
    --batch_size 16 \
    --max_shards 10 \
    --output improvements/results/eval_finetune_satlib_16s.json

echo ""
echo "Experiment 1 complete at $(date)"
