#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --output=$LOG/slurm
#SBATCH --error=$LOG/slurm

singularity exec ubuntu24.04.sif uv run __init__.py