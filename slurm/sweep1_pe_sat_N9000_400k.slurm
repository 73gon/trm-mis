#!/bin/bash -l
#SBATCH --job-name=sweep1_pe_sat_N9000_400k
#SBATCH --partition=gpu
#SBATCH --nodelist=tujestpolin
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export DATA_PATH=data/difusco_benchmark/datasets/satlib/train
export EVAL_DATA_PATH=data/difusco_benchmark/datasets/satlib/test
export RUN_NAME=sweep1_pe_sat_N9000_400k
export NUM_GRAPHS=9000
export ITERATIONS=400000
export BATCH_SIZE=16
export EVAL_EVERY=5000
export EVAL_MAX_STEPS=12
export EVAL_BATCH_SIZE=16
export POS_WEIGHT=2.0
export FEAS_W=2.0
export BENCHMARK_DATASET="SAT"
export EXTRA_ARGS="--decoder_baseline degree --log_every 20"

bash slurm/overfit/overfit_sl_pe_template.slurm
