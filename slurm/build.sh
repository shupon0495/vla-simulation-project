#!/bin/bash
#SBATCH --job-name=build
#SBATCH --time=01:00:00
#SBATCH --nodes=1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT=${PROJECT:-$(cd "$SCRIPT_DIR/.." && pwd)}
source "$PROJECT/slurm/config.sh"

ARCH_FILE="${SIF}.arch"
TARGET_ARCH=$(uname -m)

case "$TARGET_ARCH" in
    x86_64|aarch64) ;;
    *)
        echo "build: unsupported compute architecture: $TARGET_ARCH" >&2
        exit 1
        ;;
esac

COMPUTE_VENV="$PROJECT/.venv-compute-$TARGET_ARCH"

sync_compute_environment() {
    singularity exec --nv \
        --bind "$PROJECT:$PROJECT" \
        --pwd "$PROJECT" \
        --env "UV_PROJECT_ENVIRONMENT=$COMPUTE_VENV" \
        --env "UV_CACHE_DIR=$UV_CACHE_DIR" \
        "$SIF" \
        uv sync --frozen --offline "$@"
}

verify_torch_extension() {
    singularity exec --nv \
        --bind "$PROJECT:$PROJECT" \
        --pwd "$PROJECT" \
        "$SIF" \
        "$COMPUTE_VENV/bin/python" -c 'import torch; print(f"torch={torch.__version__} torch._C={torch._C.__file__}")'
}

# A SIF is architecture-specific.  This check intentionally runs inside the
# Slurm allocation so it reflects the architecture that will execute the
# pipeline, not that of the login node.
if [ ! -f "$SIF" ] || [ ! -f "$ARCH_FILE" ] || [ "$DEF" -nt "$SIF" ] || [ "$(<"$ARCH_FILE")" != "$TARGET_ARCH" ]; then
    echo "Building compute image for architecture: $TARGET_ARCH"
    singularity build --fakeroot --force "$SIF" "$DEF"
    printf '%s\n' "$TARGET_ARCH" > "$ARCH_FILE"
else
    echo "Compute image is up to date for architecture: $TARGET_ARCH"
fi

# A compute stage is strictly offline and uses --no-sync.  Build the matching
# environment here, from the login-node-populated local uv cache, and reject
# an incomplete native torch installation before dependent jobs are submitted.
if ! sync_compute_environment; then
    echo "build: unable to create $COMPUTE_VENV from the offline uv cache at $UV_CACHE_DIR" >&2
    echo "build: stage compatible wheels on the Internet-connected login node, then resubmit." >&2
    exit 1
fi
if ! verify_torch_extension; then
    echo "build: torch native extension validation failed; reinstalling locked torch from the offline cache." >&2
    sync_compute_environment --reinstall-package torch
    verify_torch_extension
fi
