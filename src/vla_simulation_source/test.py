"""Evaluation/test stage template."""
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"test stage template: data_dir={args.data_dir}")
    # TODO: 評価・テスト処理を実装する

if __name__ == "__main__":
    main()
