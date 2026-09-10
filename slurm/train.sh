#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=00:10:00
#SBATCH --nodes=1

singularity exec $SIF uv run python src/main.py