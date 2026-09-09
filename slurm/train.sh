#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=00:10:00
#SBATCH --nodes=1

singularity exec ubuntu24.04.sif uv run __init__.py