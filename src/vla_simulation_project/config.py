from __future__ import annotations
import tomllib
from dataclasses import dataclass
from pathlib import Path

BASE_MODEL_REPO = 'lerobot/smolvla_libero_plus'
BASE_MODEL_REVISION = '7bb70aa5bc92b82c9239142775d3a173103567ff'
DATASET_REPO = 'lerobot/libero_plus'
DATASET_REVISION = 'f3f49f426d75030177b18778374005bc12ccd588'
VLM_REPO = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
LIBERO_ASSETS_REPO = 'Sylvest/LIBERO-plus'
LIBERO_SOURCE_REPO = 'https://github.com/sylvestf/LIBERO-plus.git'
# This is the full commit recorded for the ``4976dc3`` revision in uv.lock.
LIBERO_SOURCE_REVISION = '4976dc30028e805ff8094b55501d532c48fec182'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
SPATIAL_TASK_NAMES = (
    'pick up the black bowl from table center and place it on the plate',
    'pick up the black bowl next to the cookie box and place it on the plate',
    'pick up the black bowl next to the plate and place it on the plate',
    'pick up the black bowl next to the ramekin and place it on the plate',
    'pick up the black bowl on the cookie box and place it on the plate',
    'pick up the black bowl on the ramekin and place it on the plate',
    'pick up the black bowl on the stove and place it on the plate',
    'pick up the black bowl on the wooden cabinet and place it on the plate',
    'pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate',
    'pick up the black bowl between the plate and the ramekin and place it on the plate',
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
    chunk_size: int
    n_action_steps: int
    task_ids: tuple[int, ...]
    episodes_per_task: int
    evaluation_seed: int
    video_task_id: int
    video_keep_all: bool
    auto_select_enabled: bool
    auto_select_n_tasks: int


def load_config(project: Path, run: Path | None = None) -> ExperimentConfig:
    if run is None:
        source = project / 'config/experiment.toml'
    else:
        source = run / 'experiment.toml'
        if not source.is_file():
            raise FileNotFoundError(
                f'load_config: pinned experiment config snapshot missing at {source}; '
                'submit jobs through submit_pipeline.sh so the config is pinned at submission time'
            )
    raw = tomllib.loads(source.read_text(encoding='utf-8'))
    t, e = (raw['training'], raw['evaluation'])
    policy = t.get('policy', {})
    auto_select = e.get('auto_select', {})
    cfg = ExperimentConfig(
        steps=t['steps'],
        batch_size=t['batch_size'],
        learning_rate=t['learning_rate'],
        final_learning_rate=t['final_learning_rate'],
        warmup_steps=t['warmup_steps'],
        lora_r=t['lora_r'],
        lora_alpha=t['lora_alpha'],
        log_freq=t['log_freq'],
        seed=t['seed'],
        chunk_size=policy.get('chunk_size', 50),
        n_action_steps=policy.get('n_action_steps', 50),
        task_ids=tuple(e['task_ids']),
        episodes_per_task=e['episodes_per_task'],
        evaluation_seed=e['seed'],
        video_task_id=e['video']['task_id'],
        video_keep_all=e['video'].get('keep_all_videos', False),
        auto_select_enabled=auto_select.get('enabled', False),
        auto_select_n_tasks=auto_select.get('n_tasks', 100),
    )
    positive = (
        cfg.steps,
        cfg.batch_size,
        cfg.warmup_steps,
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.log_freq,
        cfg.episodes_per_task,
    )
    if any(v < 1 for v in positive):
        raise ValueError('invalid experiment configuration')
    if not cfg.auto_select_enabled:
        if not cfg.task_ids or any(v not in range(10) for v in cfg.task_ids):
            raise ValueError('invalid experiment configuration')
    if cfg.video_task_id not in cfg.task_ids:
        raise ValueError('evaluation.video.task_id must be included in evaluation.task_ids')
    return cfg
