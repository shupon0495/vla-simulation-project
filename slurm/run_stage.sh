#!/usr/bin/env bash
set -euo pipefail
python_file="${1:?python file is required}"
[[ -f "$python_file" ]] || { echo "Python file was not found: $python_file" >&2; exit 1; }
cd "${PROJECT:?PROJECT is not set}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/.uv-cache}"
if [[ -z "${CONTAINER:-}" ]]; then
  echo "CONTAINER is not set" >&2
  exit 1
fi
exec singularity exec --fakeroot --bind "$PROJECT:$PROJECT" "$CONTAINER" \
  uv run --project "$PROJECT" python "$python_file" \
  --data-dir "${DATA_DIR:?DATA_DIR is not set}"
