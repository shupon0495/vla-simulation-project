#!/usr/bin/env bash
set -euo pipefail
python_file="${1:?python file is required}"
[[ -f "$python_file" ]] || { echo "Python file was not found: $python_file" >&2; exit 1; }
cd "${PROJECT:?PROJECT is not set}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/.uv-cache}"
container="${CONTAINER:?CONTAINER is not set}"
data_dir="${DATA_DIR:?DATA_DIR is not set}"
mkdir -p "$data_dir" "$UV_CACHE_DIR"

# PROJECT 配下のパスは最初の bind で見える。外部パスだけを追加で bind する。
bind_args=(--bind "$PROJECT:$PROJECT")
for path in "$data_dir" "$UV_CACHE_DIR"; do
  if [[ "$path" != "$PROJECT" && "$path" != "$PROJECT/"* ]]; then
    bind_args+=(--bind "$path:$path")
  fi
done

exec singularity exec --fakeroot "${bind_args[@]}" "$container" \
  uv run --project "$PROJECT" python "$python_file" \
  --data-dir "$data_dir"
