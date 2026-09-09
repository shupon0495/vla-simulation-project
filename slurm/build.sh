#!/bin/bash
#SBATCH --partition=ng-dgx-m2
#SBATCH --job-name=build
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --output=$LOG/slurm
#SBATCH --error=$LOG/slurm
singularity build --fakeroot --force "$SIF" "$DEF"