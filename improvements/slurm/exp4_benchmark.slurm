#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=logs/benchmark_%j.log
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Experiment 4: Final SOTA Benchmark Comparison
# Run the best model on SATLIB test + ER-700-800 test
# Compare: baseline, old best, improved model
export PYTHONUNBUFFERED=1
echo "========================================="
echo "Experiment 4: Final SOTA Benchmark"
echo "Starting on $(hostname) at $(date)"
echo "========================================="
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

mkdir -p improvements/results

# Determine best checkpoint (try improved_finetune first, then improved_ssl, then old finetune)
if [ -f "checkpoints/improved_finetune/best.pt" ]; then
    BEST_CKPT="checkpoints/improved_finetune/best.pt"
    BEST_NAME="improved_finetune"
elif [ -f "checkpoints/improved_finetune/epoch_49.pt" ]; then
    BEST_CKPT="checkpoints/improved_finetune/epoch_49.pt"
    BEST_NAME="improved_finetune"
elif [ -f "checkpoints/improved_ssl/best.pt" ]; then
    BEST_CKPT="checkpoints/improved_ssl/best.pt"
    BEST_NAME="improved_ssl"
else
    BEST_CKPT="checkpoints/mis_ssl/epoch_499.pt"
    BEST_NAME="ssl_epoch499"
fi

OLD_CKPT="checkpoints/mis_ssl/epoch_499.pt"
OLD_FT_CKPT="checkpoints/mis_finetune/epoch_49.pt"

echo "Best checkpoint: $BEST_CKPT ($BEST_NAME)"

# =====================================================
# SATLIB TEST SET
# =====================================================
echo ""
echo "==================== SATLIB TEST ===================="

# Old SSL model
echo ""
echo "--- Old SSL (epoch_499) on SATLIB test ---"
python -u improvements/eval_multisample.py \
    --checkpoint "$OLD_CKPT" \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type ssl \
    --n_samples 32 \
    --batch_size 1 \
    --output improvements/results/bench_old_ssl_satlib.json

# Old fine-tuned
echo ""
echo "--- Old Fine-tuned on SATLIB test ---"
python -u improvements/eval_multisample.py \
    --checkpoint "$OLD_FT_CKPT" \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type ssl \
    --n_samples 32 \
    --batch_size 1 \
    --output improvements/results/bench_old_ft_satlib.json

# NEW best model
echo ""
echo "--- NEW best model ($BEST_NAME) on SATLIB test ---"
python -u improvements/eval_multisample.py \
    --checkpoint "$BEST_CKPT" \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type ssl \
    --n_samples 32 \
    --batch_size 1 \
    --output improvements/results/bench_new_${BEST_NAME}_satlib.json

# =====================================================
# ER-700-800 TEST SET
# =====================================================
echo ""
echo "==================== ER-700-800 TEST ===================="

# Old SSL model
echo ""
echo "--- Old SSL (epoch_499) on ER700-800 test ---"
python -u improvements/eval_multisample.py \
    --checkpoint "$OLD_CKPT" \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type ssl \
    --n_samples 32 \
    --batch_size 1 \
    --output improvements/results/bench_old_ssl_er700.json

# Old fine-tuned
echo ""
echo "--- Old Fine-tuned on ER700-800 test ---"
python -u improvements/eval_multisample.py \
    --checkpoint "$OLD_FT_CKPT" \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type ssl \
    --n_samples 32 \
    --batch_size 1 \
    --output improvements/results/bench_old_ft_er700.json

# NEW best model
echo ""
echo "--- NEW best model ($BEST_NAME) on ER700-800 test ---"
python -u improvements/eval_multisample.py \
    --checkpoint "$BEST_CKPT" \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type ssl \
    --n_samples 32 \
    --batch_size 1 \
    --output improvements/results/bench_new_${BEST_NAME}_er700.json

echo ""
echo "========================================="
echo "All benchmarks complete at $(date)"
echo "Results saved in improvements/results/"
echo "========================================="
