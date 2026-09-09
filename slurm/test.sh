#!/bin/bash
set -euo pipefail
#SBATCH --partition=NVGPU_HPC
#SBATCH --job-name=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
cd "${PROJECT:?PROJECT is not set}"
mkdir -p "${DATA_DIR:?DATA_DIR is not set}" "${LOG:?LOG is not set}"
PYTHON_FILE="$PROJECT/src/vla_simulation_source/test.py"
exec "${RUN_STAGE:?RUN_STAGE is not set}" "$PYTHON_FILE"
