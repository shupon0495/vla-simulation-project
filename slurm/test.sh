#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:10:00
#SBATCH --nodes=1

singularity exec $SIF uv run python main.py