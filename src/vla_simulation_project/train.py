from __future__ import annotations
import json, os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from .artifacts import archive_model, read_json, write_json, write_parameter_csv
from .config import BASE_MODEL_REPO, BASE_MODEL_REVISION, DATASET_REPO, DATASET_REVISION, VLM_REPO, ExperimentConfig, load_config
from .paths import ProjectPaths, ensure_run_layout
from .prepare_assets import validate_assets
from .subprocess_utils import run_command

def build_train_command(config: ExperimentConfig, preprocess_manifest: dict, run: Path) -> list[str]:
    episodes = json.dumps(preprocess_manifest["selected_episode_indices"], separators=(",", ":"))
    return ["lerobot-train", f"--policy.path={preprocess_manifest['base_model_path']}",
        f"--policy.vlm_model_name={preprocess_manifest['vlm_path']}", "--policy.push_to_hub=false", "--policy.repo_id=null",
        "--policy.input_features=null", "--policy.output_features=null", "--policy.empty_cameras=0",
        "--policy.freeze_vision_encoder=true", "--policy.train_expert_only=true", f"--policy.optimizer_lr={config.learning_rate}",
        f"--policy.scheduler_decay_lr={config.final_learning_rate}", f"--policy.scheduler_warmup_steps={config.warmup_steps}",
        f"--policy.scheduler_decay_steps={config.steps}", f"--dataset.repo_id={DATASET_REPO}",
        f"--dataset.root={preprocess_manifest['dataset_path']}", f"--dataset.revision={DATASET_REVISION}", f"--dataset.episodes={episodes}",
        "--dataset.use_imagenet_stats=false", "--dataset.video_backend=torchcodec", f"--output_dir={run / 'training'}",
        "--job_name=smolvla_libero_plus_spatial_lora", f"--steps={config.steps}", f"--batch_size={config.batch_size}",
        "--num_workers=0", "--persistent_workers=false", "--env_eval_freq=0", "--eval_steps=0", f"--seed={config.seed}",
        "--save_checkpoint=true", f"--save_freq={config.steps}", "--save_checkpoint_to_hub=false", f"--log_freq={config.log_freq}",
        "--wandb.enable=false", "--peft.method_type=LORA", f"--peft.r={config.lora_r}", f"--peft.lora_alpha={config.lora_alpha}"]

def offline_environment(paths: ProjectPaths) -> dict[str, str]:
    env = os.environ.copy(); env.update({"HF_HOME": str(paths.data / "hf_cache"), "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1"})
    return env

def train() -> None:
    paths = ProjectPaths.from_environment(); run = ensure_run_layout(paths)
    # Repair stale upstream processor metadata before invoking the strictly
    # offline trainer.  --policy.vlm_model_name does not override the
    # tokenizer_name serialized in policy_preprocessor.json.
    validate_assets(paths, repair_tokenizer=True)
    pre = read_json(run / "manifests/preprocess.json")
    if pre.get("run_id") != paths.run_id(): raise ValueError("train: preprocess manifest RUN_ID mismatch")
    config = load_config(paths.project); command = build_train_command(config, pre, run)
    started = datetime.now(timezone.utc).isoformat(); run_command("train", command, cwd=paths.project, env=offline_environment(paths))
    checkpoint = run / "training/checkpoints" / f"{config.steps:06d}" / "pretrained_model"
    from .merge import merge_checkpoint
    model = run / "model" / f"{paths.run_id()}_smolvla"
    merge_checkpoint(paths.project / pre["base_model_path"], checkpoint, model, VLM_REPO)
    archive = run / f"{paths.run_id()}_model.tar.gz"; archive_model(model, archive)
    parameters = run / f"{paths.run_id()}_parameters.csv"
    values = {**asdict(config), "dataset_repo": DATASET_REPO, "dataset_revision": DATASET_REVISION,
        "base_model_repo": BASE_MODEL_REPO, "base_model_revision": BASE_MODEL_REVISION, "vlm_repo": VLM_REPO,
        "resolved_vlm_revision": pre["resolved_vlm_revision"], "training_episode_count": pre["selected_episode_count"]}
    write_parameter_csv(parameters, paths.run_id(), values)
    write_json(run / "manifests/train.json", {"run_id": paths.run_id(), "stage": "train", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "training_started_at": started, "training_finished_at": datetime.now(timezone.utc).isoformat(), "training_exit_code": 0,
        "checkpoint_path": str(checkpoint), "merged_model_path": str(model), "model_archive_path": str(archive),
        "parameters_csv_path": str(parameters), "command": command, "experiment_config": values})
