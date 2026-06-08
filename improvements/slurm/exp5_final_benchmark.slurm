#!/bin/bash
#SBATCH --job-name=final_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/final_bench_%j.log
#SBATCH --error=logs/final_bench_%j.err

echo "========================================="
echo "Experiment 5: Final Benchmark — SATLIB test + ER-700-800 test"
echo "Starting on $(hostname) at $(date)"
echo "========================================="

cd /home/mmardan/trm
source .venv/bin/activate

EVAL_SCRIPT="improvements/eval_multisample.py"
RESULTS_DIR="improvements/results"
mkdir -p "$RESULTS_DIR"

N_SAMPLES=32

# ============================================================
#  SATLIB TEST — 500 graphs, 2 checkpoints
# ============================================================

echo ""
echo "=== SATLIB TEST: mis_finetune (32 samples) ==="
python3 "$EVAL_SCRIPT" \
  --checkpoint checkpoints/mis_finetune/epoch_49.pt \
  --data_path data/difusco_benchmark/datasets/satlib/test \
  --model_type ssl \
  --n_samples $N_SAMPLES \
  --batch_size 32 \
  --output "$RESULTS_DIR/bench_finetune_satlib_test.json"

echo ""
echo "=== SATLIB TEST: mis_ft_high (32 samples) ==="
python3 "$EVAL_SCRIPT" \
  --checkpoint checkpoints/mis_ft_high/epoch_49.pt \
  --data_path data/difusco_benchmark/datasets/satlib/test \
  --model_type ssl \
  --n_samples $N_SAMPLES \
  --batch_size 32 \
  --output "$RESULTS_DIR/bench_ft_high_satlib_test.json"

# ============================================================
#  ER-700-800 TEST — 500 graphs, 2 checkpoints
# ============================================================

echo ""
echo "=== ER-700-800 TEST: mis_finetune (32 samples) ==="
python3 "$EVAL_SCRIPT" \
  --checkpoint checkpoints/mis_finetune/epoch_49.pt \
  --data_path data/difusco_benchmark/datasets/er_700_800/test \
  --model_type ssl \
  --n_samples $N_SAMPLES \
  --batch_size 16 \
  --output "$RESULTS_DIR/bench_finetune_er700_test.json"

echo ""
echo "=== ER-700-800 TEST: mis_ft_high (32 samples) ==="
python3 "$EVAL_SCRIPT" \
  --checkpoint checkpoints/mis_ft_high/epoch_49.pt \
  --data_path data/difusco_benchmark/datasets/er_700_800/test \
  --model_type ssl \
  --n_samples $N_SAMPLES \
  --batch_size 16 \
  --output "$RESULTS_DIR/bench_ft_high_er700_test.json"

echo ""
echo "========================================="
echo "Final Benchmark complete at $(date)"
echo "========================================="
