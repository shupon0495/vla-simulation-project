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
