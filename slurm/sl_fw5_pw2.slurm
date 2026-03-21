#!/bin/bash
#SBATCH --job-name=sl_fw5_pw2
#SBATCH --output=logs/sl_fw5_pw2_%j.log
#SBATCH --error=logs/sl_fw5_pw2_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Supervised: feasibility_weight=5.0, pos_weight=auto (~2.0)
# Strong feasibility penalty to force model to learn independent set structure
python -u train_mis.py \
    --batch_size 16 --epochs 100 \
    --feasibility_weight 5.0 \
    --feasibility_loss_type hinge \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/sl_fw5_pw2 \
    --run_name sl_fw5_pwauto_158sh
