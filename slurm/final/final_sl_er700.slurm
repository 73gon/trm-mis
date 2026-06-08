#!/bin/bash
#SBATCH --job-name=f_sl_er
#SBATCH --output=logs/final_sl_er700_%j.log
#SBATCH --error=logs/final_sl_er700_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="final,sl,er700,focal,cardinality,final_runs"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

CHKPT_DIR=checkpoints/final_sl_er700
mkdir -p "$CHKPT_DIR"
if [ -f "$CHKPT_DIR/best.pt" ]; then
    PRETRAINED="--pretrained $CHKPT_DIR/best.pt"
    echo "Resuming from $CHKPT_DIR/best.pt"
else
    PRETRAINED=""
    echo "Training from scratch"
fi

# ER-700-800 has ~6% positives (vs SATLIB ~33%), so bump pos_weight to 15
python -u -m experiments.train_sl_pilot \
    --batch_size 2 --epochs 100 \
    --use_pe 1 --use_enhanced_features 1 \
    --data_path data/difusco_benchmark/datasets/er_700_800/train \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --pos_weight 15.0 \
    --focal_gamma 2.0 --focal_alpha 0.25 \
    --cardinality_weight 0.1 \
    --checkpoint_path "$CHKPT_DIR" \
    --run_name final_sl_er700 \
    $PRETRAINED

echo "=== Training done. Starting eval ==="
python -u improvements/eval_multisample.py \
    --checkpoint "$CHKPT_DIR/best.pt" \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type supervised \
    --n_samples 32 \
    --wandb --wandb_run_name final_sl_er700_eval \
    --output improvements/results/final_sl_er700.json || true

echo "Done at $(date)"
