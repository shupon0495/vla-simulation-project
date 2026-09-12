#!/bin/bash
#SBATCH --job-name=build
#SBATCH --time=01:00:00
#SBATCH --nodes=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT=${PROJECT:-$(cd "$SCRIPT_DIR/.." && pwd)}
source "$PROJECT/slurm/config.sh"

singularity build --fakeroot --force "$SIF" "$DEF"
