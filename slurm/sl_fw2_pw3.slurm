#!/bin/bash
#SBATCH --job-name=sl_fw2_pw3
#SBATCH --output=logs/sl_fw2_pw3_%j.log
#SBATCH --error=logs/sl_fw2_pw3_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
echo "Python: $(which python)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# Supervised: feasibility_weight=2.0, pos_weight=3.0
# Balanced: moderate pos_weight to avoid predicting everything positive,
# moderate feasibility to penalize adjacent selections
python -u train_mis.py \
    --batch_size 16 --epochs 100 \
    --pos_weight 3.0 \
    --feasibility_weight 2.0 \
    --feasibility_loss_type hinge \
    --use_pe 0 --use_enhanced_features 0 \
    --checkpoint_path checkpoints/sl_fw2_pw3 \
    --run_name sl_fw2_pw3_158sh
