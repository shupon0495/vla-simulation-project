#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:10:00
#SBATCH --nodes=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT=${PROJECT:-$(cd "$SCRIPT_DIR/.." && pwd)}
source "$PROJECT/slurm/config.sh"

cd "$PROJECT"
singularity exec --bind "$PROJECT:$PROJECT" --pwd "$PROJECT" "$SIF" \
    uv run --frozen --offline python -m vla_simulation_project.main test
