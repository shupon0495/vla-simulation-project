from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

def run_command(stage: str, command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> None:
    try:
        subprocess.run(list(command), check=True, cwd=cwd, env=dict(env))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'{stage} failed while running: {' '.join(command)} (cwd={cwd})') from exc
