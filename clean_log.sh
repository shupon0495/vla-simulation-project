#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET_DIR="$SCRIPT_DIR/log"

if [[ ! -d "$TARGET_DIR" ]]; then
    printf '削除対象のディレクトリが存在しません: %s\n' "$TARGET_DIR"
    exit 0
fi

while true; do
    read -r -p "$TARGET_DIR 内のファイルを削除しますか？ (y/n): " yn
    case "$yn" in
        [Yy])
            find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
            printf '削除しました: %s\n' "$TARGET_DIR"
            break
            ;;
        [Nn])
            printf '処理を中止しました。\n'
            exit 0
            ;;
        *)
            printf 'y または n で入力してください。\n'
            ;;
    esac
done
