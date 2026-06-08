#!/bin/bash
#SBATCH --job-name=f_ssl_sat
#SBATCH --output=logs/final_ssl_satlib_%j.log
#SBATCH --error=logs/final_ssl_satlib_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="final,ssl,satlib,reinforce,entropy,temp_anneal,final_runs"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

CHKPT_DIR=checkpoints/final_ssl_satlib
mkdir -p "$CHKPT_DIR"

# Final SSL: REINFORCE + stronger entropy (0.05) + wider temp anneal (3.0→0.3)
# Note: train_mis_ssl.py does not support --pretrained; SSL typically plateaus
# within 30-50 epochs so a single 48h run is enough.
python -u train_mis_ssl.py \
    --batch_size 16 --epochs 50 \
    --use_pe 1 --use_enhanced_features 1 \
    --loss_mode reinforce \
    --entropy_weight 0.05 \
    --temp_start 3.0 --temp_end 0.3 \
    --checkpoint_path "$CHKPT_DIR" \
    --run_name final_ssl_satlib

echo "=== Training done. Starting eval ==="
python -u improvements/eval_multisample.py \
    --checkpoint "$CHKPT_DIR/best.pt" \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type ssl \
    --n_samples 32 \
    --wandb --wandb_run_name final_ssl_satlib_eval \
    --output improvements/results/final_ssl_satlib.json || true

echo "Done at $(date)"
