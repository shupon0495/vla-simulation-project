#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:10:00
#SBATCH --nodes=1

singularity exec ubuntu24.04.sif uv run __init__.py