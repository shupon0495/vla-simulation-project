from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project: Path

    @classmethod
    def from_environment(cls):
        return cls(Path(os.environ.get("PROJECT", Path.cwd())).resolve())

    @property
    def data(self):
        return self.project / "data"

    def run_id(self):
        if not (v := os.environ.get("RUN_ID")):
            raise RuntimeError("RUN_ID is required; use submit_pipeline.sh")
        return v

    def run_dir(self):
        if not (v := os.environ.get("RUN_DIR")):
            raise RuntimeError("RUN_DIR is required; use submit_pipeline.sh")
        p = Path(v).resolve()
        root = (self.data / "outputs").resolve()
        if root not in p.parents:
            raise RuntimeError(f"RUN_DIR must be under {root}")
        return p


def ensure_run_layout(paths):
    run = paths.run_dir()
    for p in (run, run / "manifests", run / "model", run / "eval"):
        p.mkdir(parents=True, exist_ok=True)
    return run
