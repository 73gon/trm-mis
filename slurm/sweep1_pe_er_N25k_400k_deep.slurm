#!/bin/bash -l
#SBATCH --job-name=sweep1_pe_er_N25k_400k_deep
#SBATCH --partition=gpu
#SBATCH --nodelist=tujestpolin
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export DATA_PATH=data/difusco_benchmark/datasets/er_700_800/train
export EVAL_DATA_PATH=data/difusco_benchmark/datasets/er_700_800/test
export RUN_NAME=sweep1_pe_er_N25k_400k_deep
export NUM_GRAPHS=25000
export ITERATIONS=400000
export BATCH_SIZE=4
export EVAL_EVERY=5000
export EVAL_MAX_STEPS=12
export EVAL_BATCH_SIZE=2
export POS_WEIGHT=5.0
export FEAS_W=2.0
export BENCHMARK_DATASET="ER-700"
export EXTRA_ARGS="--decoder_baseline degree --num_layers 4 --L_cycles 12"

bash slurm/overfit/overfit_sl_pe_template.slurm
