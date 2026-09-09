#!/bin/bash
#SBATCH --job-name=build
#SBATCH --time=00:10:00
#SBATCH --nodes=1
singularity build --fakeroot --force "$SIF" "$DEF"