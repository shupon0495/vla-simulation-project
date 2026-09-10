"""CLI dispatcher for isolated pipeline stages."""
from __future__ import annotations
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare-assets", "preprocess", "train", "test"))
    stage = parser.parse_args().stage
    if stage == "prepare-assets":
        from .prepare_assets import prepare_assets

        prepare_assets()
    elif stage == "preprocess":
        from .preprocess import preprocess

        preprocess()
    elif stage == "train":
        from .train import train

        train()
    else:
        from .evaluate import evaluate

        evaluate()

if __name__ == "__main__":
    main()
