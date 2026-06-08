#!/bin/bash
#SBATCH --job-name=f_sl_sat
#SBATCH --output=logs/final_sl_satlib_%j.log
#SBATCH --error=logs/final_sl_satlib_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
export WANDB_TAGS="final,sl,satlib,focal,cardinality,final_runs"
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

CHKPT_DIR=checkpoints/final_sl_satlib
mkdir -p "$CHKPT_DIR"
if [ -f "$CHKPT_DIR/best.pt" ]; then
    PRETRAINED="--pretrained $CHKPT_DIR/best.pt"
    echo "Resuming from $CHKPT_DIR/best.pt"
else
    PRETRAINED=""
    echo "Training from scratch"
fi

# Final SL: focal (γ=2, α=0.25) + cardinality (λ=0.1) combo
# Full 158 SATLIB shards, 100 epochs, PE + enhanced features enabled
python -u -m experiments.train_sl_pilot \
    --batch_size 16 --epochs 100 \
    --use_pe 1 --use_enhanced_features 1 \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --pos_weight 5.0 \
    --focal_gamma 2.0 --focal_alpha 0.25 \
    --cardinality_weight 0.1 \
    --checkpoint_path "$CHKPT_DIR" \
    --run_name final_sl_satlib \
    $PRETRAINED

echo "=== Training done. Starting eval ==="
python -u improvements/eval_multisample.py \
    --checkpoint "$CHKPT_DIR/best.pt" \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type supervised \
    --n_samples 32 \
    --wandb --wandb_run_name final_sl_satlib_eval \
    --output improvements/results/final_sl_satlib.json || true

echo "Done at $(date)"
