#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT=${PROJECT:-$(cd "$SCRIPT_DIR/.." && pwd)}
source "$PROJECT/slurm/config.sh"

TARGET_ARCH=$(uname -m)
COMPUTE_VENV="$PROJECT/.venv-compute-$TARGET_ARCH"

cd "$PROJECT"
singularity exec --nv \
    --bind "$PROJECT:$PROJECT" \
    --bind /usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d:ro \
    --env __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    --env "UV_PROJECT_ENVIRONMENT=$COMPUTE_VENV" \
    --env "UV_CACHE_DIR=$UV_CACHE_DIR" \
    --pwd "$PROJECT" "$SIF" \
    uv run --frozen --offline --no-sync python -m vla_simulation_project.main test
