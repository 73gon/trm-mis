#!/bin/bash
#SBATCH --job-name=prep_feat
#SBATCH --output=logs/preprocess_features_%j.log
#SBATCH --error=logs/preprocess_features_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

export PYTHONUNBUFFERED=1
echo "Starting on $(hostname) at $(date)"
source /home/mmardan/trm/.venv/bin/activate
cd /home/mmardan/trm

# SATLIB: ~10k graphs of ~1200 nodes. ER-700: ~7k graphs of ~750 nodes.
# Writes sibling *.cache.pt files; skipped on re-run if newer than source.
for path in \
    data/difusco_benchmark/datasets/satlib/train \
    data/difusco_benchmark/datasets/satlib/test \
    data/difusco_benchmark/datasets/er_700_800/train \
    data/difusco_benchmark/datasets/er_700_800/test
do
    if [ -d "$path" ]; then
        echo "=== $path ==="
        python -u -m experiments.preprocess_features \
            --data_path "$path" \
            --pe_dim 16 \
            --workers 16
    else
        echo "(skipping missing $path)"
    fi
done

echo "Done at $(date)"
