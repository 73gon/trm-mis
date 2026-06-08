#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --output=logs/eval_all_%j.log
#SBATCH --error=logs/eval_all_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# ============================================================
# Eval ALL comparison checkpoints (run after training completes)
# Each checkpoint: SATLIB test + ER-700-800 test with wandb logging
# ============================================================

declare -A CKPTS
CKPTS[cmp_ssl_orig]="checkpoints/cmp_ssl_orig/best.pt:ssl"
CKPTS[cmp_ssl_impr]="checkpoints/cmp_ssl_impr/best.pt:ssl"
CKPTS[cmp_sup_scratch]="checkpoints/cmp_sup_scratch/best.pt:supervised"
CKPTS[cmp_sup_ft]="checkpoints/cmp_sup_ft/best.pt:supervised"

DATASETS=(
    "data/difusco_benchmark/datasets/satlib/test:satlib"
    "data/difusco_benchmark/datasets/er_700_800/test:er700"
)

for name in "${!CKPTS[@]}"; do
    IFS=':' read -r ckpt model_type <<< "${CKPTS[$name]}"

    if [ ! -f "$ckpt" ]; then
        echo "⚠️ Checkpoint not found: $ckpt — skipping $name"
        continue
    fi

    for ds_entry in "${DATASETS[@]}"; do
        IFS=':' read -r ds_path ds_short <<< "$ds_entry"

        echo ""
        echo "========================================================"
        echo "Evaluating: $name on $ds_short"
        echo "  Checkpoint: $ckpt"
        echo "  Dataset: $ds_path"
        echo "  Model type: $model_type"
        echo "========================================================"

        python -u improvements/eval_multisample.py \
            --checkpoint "$ckpt" \
            --data_path "$ds_path" \
            --model_type "$model_type" \
            --n_samples 32 \
            --wandb --wandb_run_name "eval_${name}_${ds_short}" \
            --output "improvements/results/${name}_${ds_short}.json"
    done
done

echo ""
echo "All evaluations done at $(date)"
