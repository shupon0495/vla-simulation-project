#!/bin/bash
#SBATCH --partition=NVGPU_HPC
#SBATCH --job-name=train
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --output=$LOG/slurm
#SBATCH --error=$LOG/slurm