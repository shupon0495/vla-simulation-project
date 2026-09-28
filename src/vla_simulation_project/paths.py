from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    project: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, 'project', self.project.resolve())

    @classmethod
    def from_environment(cls) -> 'ProjectPaths':
        return cls(Path(os.environ.get('PROJECT', Path.cwd())))

    @property
    def data(self) -> Path:
        return self.project / 'data'

    def run_id(self) -> str:
        value = os.environ.get('RUN_ID', '')
        if not value or not value.replace('-', '').isalnum():
            raise RuntimeError('RUN_ID is missing or invalid; use submit_pipeline.sh')
        return value

    def run_dir(self) -> Path:
        value = os.environ.get('RUN_DIR')
        if not value:
            raise RuntimeError('RUN_DIR is required; use submit_pipeline.sh')
        path, root = (Path(value).resolve(), (self.data / 'outputs').resolve())
        if path == root or root not in path.parents:
            raise RuntimeError(f'RUN_DIR must be a child of {root}')
        return path

def ensure_run_layout(paths: ProjectPaths) -> Path:
    run = paths.run_dir()
    for directory in (run, run / 'manifests', run / 'model', run / 'eval'):
        directory.mkdir(parents=True, exist_ok=True)
    return run
