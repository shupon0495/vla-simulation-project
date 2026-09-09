#!/bin/bash
#SBATCH --partition=NVGPU_HPC
#SBATCH --job-name=preprocess
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --output=$LOG/slurm
#SBATCH --error=$LOG/slurm

singularity exec ubuntu24.04.sif uv run __init__.py