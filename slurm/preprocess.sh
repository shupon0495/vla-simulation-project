#!/bin/bash
#SBATCH --partition=NVGPU_HPC
#SBATCH --job-name=preprocess
#SBATCH --time=00:30:00
#SBATCH --nodes=1

set -euo pipefail
cd "${PROJECT:?PROJECT is not set}"
mkdir -p "${DATA_DIR:?DATA_DIR is not set}" "${LOG:?LOG is not set}"
PYTHON_FILE="$PROJECT/src/vla_simulation_source/preprocess.py"
exec "${RUN_STAGE:?RUN_STAGE is not set}" "$PYTHON_FILE"
