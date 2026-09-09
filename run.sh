#!/bin/bash
#SBATCH --partition=NVGPU_HPC
#SBATCH --job-name=VLA_simulation
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --output=VLA_simulation_%j.out