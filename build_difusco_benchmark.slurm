#!/bin/bash
#SBATCH --job-name=build_difusco
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/build_difusco_%j.log
#SBATCH --error=logs/build_difusco_%j.err

# Activate your virtual environment
source /home/mmardan/trm/.venv/bin/activate

# Run your command
python dataset/build_difusco_benchmark.py --mode er --num_train 163840 --num_workers 20 --kamis_time_limit 60
