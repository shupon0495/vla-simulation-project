"""Stage runner used by the Slurm pipeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preprocess", "train", "test"))
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "preprocess":
        payload = {"status": "preprocessed"}
        output = args.data_dir / "preprocessed.json"
    elif args.stage == "train":
        if not (args.data_dir / "preprocessed.json").exists():
            raise FileNotFoundError("preprocessed.json is missing")
        payload, output = {"status": "trained"}, args.data_dir / "model.json"
    else:
        if not (args.data_dir / "model.json").exists():
            raise FileNotFoundError("model.json is missing")
        payload, output = {"status": "passed"}, args.data_dir / "test.json"
    output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"completed stage: {args.stage}")

if __name__ == "__main__":
    main()
