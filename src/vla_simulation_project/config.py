from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
BASE_MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"
DATASET_REPO = "lerobot/libero_plus"
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SPATIAL_TASK_NAMES = (
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
    "pick up the black bowl on the wooden cabinet shelf and place it on the plate",
    "pick up the black bowl on the wooden cabinet top and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
)


@dataclass(frozen=True)
class ExperimentConfig:
    steps: int
    batch_size: int
    learning_rate: float
    final_learning_rate: float
    warmup_steps: int
    lora_r: int
    lora_alpha: int
    log_freq: int
    seed: int
    task_ids: tuple[int, ...]
    episodes_per_task: int
    evaluation_seed: int
    video_task_id: int


def load_config(project: Path) -> ExperimentConfig:
    raw = tomllib.loads((project / "config/experiment.toml").read_text())
    t, e = raw["training"], raw["evaluation"]
    c = ExperimentConfig(
        **{
            k: t[k]
            for k in (
                "steps",
                "batch_size",
                "learning_rate",
                "final_learning_rate",
                "warmup_steps",
                "lora_r",
                "lora_alpha",
                "log_freq",
                "seed",
            )
        },
        task_ids=tuple(e["task_ids"]),
        episodes_per_task=e["episodes_per_task"],
        evaluation_seed=e["seed"],
        video_task_id=e["video"]["task_id"],
    )
    if (
        min(c.steps, c.batch_size, c.episodes_per_task) < 1
        or not c.task_ids
        or any(x not in range(10) for x in c.task_ids)
    ):
        raise ValueError("invalid experiment configuration")
    return c
