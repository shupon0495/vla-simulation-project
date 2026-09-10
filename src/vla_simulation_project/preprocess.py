from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path
from .artifacts import write_json
from .config import DATASET_REPO, DATASET_REVISION, SPATIAL_TASK_NAMES
from .paths import ProjectPaths, ensure_run_layout
from .prepare_assets import validate_assets

def normalize_task_name(value: str) -> str:
    value = value.lower().replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", value)).strip()

def task_name_from_cell(value: object) -> str:
    if isinstance(value, str): return value
    try: return str(value[0]) if len(value) else str(value)  # type: ignore[arg-type,index]
    except TypeError: return str(value)

def choose_evenly_spaced(episode_indices: list[int], count: int) -> list[int]:
    if count < 1 or len(episode_indices) < count: raise ValueError(f"task has {len(episode_indices)} episodes but {count} are required")
    if count == 1: return [episode_indices[0]]
    return [episode_indices[round(index * (len(episode_indices) - 1) / (count - 1))] for index in range(count)]

def select_spatial_episodes(tasks: list[object], per_task: int = 5) -> tuple[list[int], dict[str, list[int]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cell in enumerate(tasks): grouped[task_name_from_cell(cell)].append(index)
    available = {normalize_task_name(name): name for name in grouped}; selected = {}
    for expected in SPATIAL_TASK_NAMES:
        actual = available.get(normalize_task_name(expected))
        if actual is None: raise RuntimeError(f"Spatial task not found in local dataset metadata: {expected}")
        selected[actual] = choose_evenly_spaced(grouped[actual], per_task)
    indices = sorted(index for values in selected.values() for index in values)
    if len(indices) != len(SPATIAL_TASK_NAMES) * per_task: raise RuntimeError("episode selection produced an unexpected count")
    return indices, selected

def load_episode_tasks(dataset: Path) -> list[object]:
    jsonl = dataset / "meta/episodes.jsonl"
    if jsonl.is_file():
        tasks = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); tasks.append(row.get("tasks", row.get("task")))
        return tasks
    parquet = sorted((dataset / "meta/episodes").rglob("*.parquet"))
    if parquet:
        import pandas as pd
        return [value for file in parquet for value in pd.read_parquet(file)["tasks"].tolist()]
    raise FileNotFoundError(f"preprocess: local episode metadata missing under {dataset}; rerun prepare-assets on the login node")

def preprocess() -> None:
    paths = ProjectPaths.from_environment(); run = ensure_run_layout(paths); lock = validate_assets(paths)
    dataset = paths.project / lock["dataset"]["local_path"]
    indices, selected = select_spatial_episodes(load_episode_tasks(dataset))
    write_json(run / "manifests/preprocess.json", {"run_id": paths.run_id(), "stage": "preprocess",
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"), "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION, "selected_episode_indices": indices, "selected_episode_count": len(indices),
        "selected_by_task": selected, "base_model_path": lock["base_model"]["local_path"],
        "dataset_path": lock["dataset"]["local_path"], "vlm_path": lock["vlm"]["local_path"],
        "resolved_vlm_revision": lock["vlm"]["revision"], "libero_assets_path": lock["libero_assets"]["local_path"]})
