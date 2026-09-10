from __future__ import annotations

import re
from collections import defaultdict

from .artifacts import write_json
from .config import DATASET_REPO, DATASET_REVISION, SPATIAL_TASK_NAMES
from .paths import ProjectPaths, ensure_run_layout
from .prepare_assets import validate_assets


def normalize_task_name(v):
    return re.sub(
        r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(v).lower().replace("_", " "))
    ).strip()


def task_name_from_cell(v):
    if isinstance(v, str):
        return v
    try:
        return str(v[0]) if len(v) else str(v)
    except TypeError:
        return str(v)


def choose_evenly_spaced(ids, count):
    if count < 1 or len(ids) < count:
        raise ValueError(f"Task has {len(ids)} episodes, needs {count}")
    return [ids[0]] if count == 1 else [
        ids[round(i * (len(ids) - 1) / (count - 1))] for i in range(count)
    ]


def select_spatial_episodes(tasks, per_task=5):
    grouped = defaultdict(list)
    for i, x in enumerate(tasks):
        grouped[task_name_from_cell(x)].append(i)
    names = {normalize_task_name(k): k for k in grouped}
    selected = {}
    for expected in SPATIAL_TASK_NAMES:
        if not (actual := names.get(normalize_task_name(expected))):
            raise RuntimeError(f"Spatial task not found: {expected}")
        selected[actual] = choose_evenly_spaced(grouped[actual], per_task)
    ids = sorted(x for values in selected.values() for x in values)
    if len(ids) != 50:
        raise RuntimeError("Expected exactly 50 selected episodes")
    return ids, selected


def preprocess():
    paths = ProjectPaths.from_environment()
    run = ensure_run_layout(paths)
    lock = validate_assets(paths)
    try:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        md = LeRobotDatasetMetadata(
            DATASET_REPO,
            root=paths.project / lock["dataset"]["local_path"],
            revision=DATASET_REVISION,
        )
    except Exception as e:
        raise RuntimeError(
            "preprocess cannot load local dataset metadata; run login-node asset preparation"
        ) from e
    ids, selected = select_spatial_episodes(list(md.episodes["tasks"]))
    write_json(
        run / "manifests/preprocess.json",
        dict(
            run_id=paths.run_id(),
            dataset_repo=DATASET_REPO,
            dataset_revision=DATASET_REVISION,
            selected_episode_indices=ids,
            selected_episode_count=len(ids),
            selected_by_task=selected,
            base_model_path=lock["base_model"]["local_path"],
            dataset_path=lock["dataset"]["local_path"],
            vlm_path=lock["vlm"]["local_path"],
            libero_assets_path=lock["libero_assets"]["local_path"],
        ),
    )
