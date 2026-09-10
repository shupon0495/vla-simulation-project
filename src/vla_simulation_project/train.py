from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import archive_model, read_json, write_json, write_parameter_csv
from .config import DATASET_REPO, DATASET_REVISION, VLM_REPO, load_config
from .paths import ProjectPaths, ensure_run_layout
from .prepare_assets import validate_assets


def build_train_command(c, pre, run):
    episodes = json.dumps(pre["selected_episode_indices"], separators=(",", ":"))
    output = run / "training"
    return [
        "lerobot-train",
        f"--policy.path={pre['base_model_path']}",
        f"--policy.vlm_model_name={pre['vlm_path']}",
        "--policy.push_to_hub=false",
        "--policy.repo_id=null",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        f"--policy.optimizer_lr={c.learning_rate}",
        f"--policy.scheduler_decay_lr={c.final_learning_rate}",
        f"--policy.scheduler_warmup_steps={c.warmup_steps}",
        f"--policy.scheduler_decay_steps={c.steps}",
        f"--dataset.repo_id={DATASET_REPO}",
        f"--dataset.root={pre['dataset_path']}",
        f"--dataset.revision={DATASET_REVISION}",
        f"--dataset.episodes={episodes}",
        "--dataset.use_imagenet_stats=false",
        "--dataset.video_backend=torchcodec",
        f"--output_dir={output}",
        "--job_name=smolvla_libero_plus_spatial_lora",
        f"--steps={c.steps}",
        f"--batch_size={c.batch_size}",
        "--num_workers=0",
        "--persistent_workers=false",
        "--env_eval_freq=0",
        "--eval_steps=0",
        f"--seed={c.seed}",
        "--save_checkpoint=true",
        f"--save_freq={c.steps}",
        "--save_checkpoint_to_hub=false",
        f"--log_freq={c.log_freq}",
        "--wandb.enable=false",
        "--peft.method_type=LORA",
        f"--peft.r={c.lora_r}",
        f"--peft.lora_alpha={c.lora_alpha}",
    ]


def train():
    paths = ProjectPaths.from_environment()
    run = ensure_run_layout(paths)
    validate_assets(paths)
    pre = read_json(run / "manifests/preprocess.json")
    c = load_config(paths.project)
    cmd = build_train_command(c, pre, run)
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(paths.data / "hf_cache"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = datetime.now(timezone.utc).isoformat()
    import subprocess

    subprocess.run(cmd, check=True, cwd=paths.project, env=env)
    finished = datetime.now(timezone.utc).isoformat()
    checkpoint = run / "training" / "checkpoints" / f"{c.steps:06d}" / "pretrained_model"
    from .merge import merge_checkpoint

    model = run / "model" / f"{paths.run_id()}_smolvla"
    merge_checkpoint(
        paths.project / pre["base_model_path"], checkpoint, model, pre["vlm_path"]
    )
    archive = run / f"{paths.run_id()}_model.tar.gz"
    archive_model(model, archive)
    params = run / f"{paths.run_id()}_parameters.csv"
    write_parameter_csv(
        params,
        paths.run_id(),
        {
            **c.__dict__,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "base_model_repo": "lerobot/smolvla_libero_plus",
            "vlm_repo": VLM_REPO,
            "training_episode_count": pre["selected_episode_count"],
        },
    )
    write_json(
        run / "manifests/train.json",
        dict(
            run_id=paths.run_id(),
            training_started_at=started,
            training_finished_at=finished,
            training_exit_code=0,
            checkpoint_path=str(checkpoint),
            merged_model_path=str(model),
            model_archive_path=str(archive),
            parameters_csv_path=str(params),
            command=cmd,
        ),
    )
