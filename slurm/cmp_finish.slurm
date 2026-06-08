#!/bin/bash
#SBATCH --job-name=cmp_finish
#SBATCH --output=logs/cmp_finish_%j.log
#SBATCH --error=logs/cmp_finish_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# ============================================================
# Step 1: Supervised fine-tuning from SSL pretrained checkpoint
# (Phase 2 that never ran due to timeout in job 1117)
# ============================================================
echo ""
echo "=========================================="
echo "STEP 1: Supervised fine-tune from SSL pretrain"
echo "=========================================="
python -u train_mis.py \
    --batch_size 16 --epochs 50 \
    --use_pe 0 --use_enhanced_features 0 \
    --pretrained checkpoints/cmp_ssl_pretrain/best.pt \
    --feasibility_weight 1.0 --feasibility_loss_type hinge \
    --pos_weight 5.0 \
    --checkpoint_path checkpoints/cmp_sup_ft \
    --run_name cmp_sup_finetune

# ============================================================
# Step 2: Eval cmp_ssl_impr (improved SSL) on both test sets
# ============================================================
echo ""
echo "=========================================="
echo "STEP 2: Eval improved SSL"
echo "=========================================="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_ssl_impr/best.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type ssl --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_ssl_impr_satlib \
    --output improvements/results/cmp_ssl_impr_satlib.json

python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_ssl_impr/best.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type ssl --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_ssl_impr_er700 \
    --output improvements/results/cmp_ssl_impr_er700.json

# ============================================================
# Step 3: Eval cmp_sup_scratch (supervised from scratch)
# ============================================================
echo ""
echo "=========================================="
echo "STEP 3: Eval supervised from scratch"
echo "=========================================="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_sup_scratch/best.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type supervised --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_sup_scratch_satlib \
    --output improvements/results/cmp_sup_scratch_satlib.json

python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_sup_scratch/best.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type supervised --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_sup_scratch_er700 \
    --output improvements/results/cmp_sup_scratch_er700.json

# ============================================================
# Step 4: Eval cmp_sup_ft (supervised fine-tuned from SSL)
# ============================================================
echo ""
echo "=========================================="
echo "STEP 4: Eval supervised fine-tuned"
echo "=========================================="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_sup_ft/best.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type supervised --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_sup_ft_satlib \
    --output improvements/results/cmp_sup_ft_satlib.json

python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/cmp_sup_ft/best.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type supervised --n_samples 32 \
    --wandb --wandb_run_name eval_cmp_sup_ft_er700 \
    --output improvements/results/cmp_sup_ft_er700.json

# ============================================================
# Step 5: Also eval existing best checkpoints (mis_finetune, mis_ft_high)
# with wandb logging for full comparison
# ============================================================
echo ""
echo "=========================================="
echo "STEP 5: Eval previous best models with wandb"
echo "=========================================="
python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/mis_finetune/epoch_49.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type supervised --n_samples 32 \
    --wandb --wandb_run_name eval_mis_finetune_satlib \
    --output improvements/results/eval_mis_finetune_satlib.json

python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/mis_finetune/epoch_49.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type supervised --n_samples 32 \
    --wandb --wandb_run_name eval_mis_finetune_er700 \
    --output improvements/results/eval_mis_finetune_er700.json

python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/mis_ft_high/epoch_49.pt \
    --data_path data/difusco_benchmark/datasets/satlib/test \
    --model_type supervised --n_samples 32 \
    --wandb --wandb_run_name eval_mis_ft_high_satlib \
    --output improvements/results/eval_mis_ft_high_satlib.json

python -u improvements/eval_multisample.py \
    --checkpoint checkpoints/mis_ft_high/epoch_49.pt \
    --data_path data/difusco_benchmark/datasets/er_700_800/test \
    --model_type supervised --n_samples 32 \
    --wandb --wandb_run_name eval_mis_ft_high_er700 \
    --output improvements/results/eval_mis_ft_high_er700.json

echo ""
echo "=========================================="
echo "ALL DONE at $(date)"
echo "=========================================="
