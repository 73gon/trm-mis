#!/bin/bash
#SBATCH --job-name=f_ssl_er
#SBATCH --output=logs/final_ssl_er700_%j.log
#SBATCH --error=logs/final_ssl_er700_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="final,ssl,er700,reinforce,entropy,temp_anneal,final_runs"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

CHKPT_DIR=checkpoints/final_ssl_er700
mkdir -p "$CHKPT_DIR"

# ER-700 SSL with REINFORCE + stronger entropy + temp anneal
# Use experiments.train_ssl_pilot wrapper (supports --data_path).
python -u -m experiments.train_ssl_pilot \
    --batch_size 8 --epochs 50 \
    --use_pe 1 --use_enhanced_features 1 \
    --data_path data/difusco_benchmark/datasets/er_700_800/train \
    --loss_mode reinforce \
    --entropy_weight 0.05 \
    --temp_start 3.0 --temp_end 0.3 \
    --checkpoint_path "$CHKPT_DIR" \
    --run_name final_ssl_er700

echo "=== Training done. Starting eval ==="
python -u improvements/eval_multisample.py \
    --checkpoint "$CHKPT_DIR/best.pt" \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type ssl \
    --n_samples 32 \
    --wandb --wandb_run_name final_ssl_er700_eval \
    --output improvements/results/final_ssl_er700.json || true

echo "Done at $(date)"
