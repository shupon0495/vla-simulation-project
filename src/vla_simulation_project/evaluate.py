from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

from .artifacts import read_json, result_rows, write_json, write_results_csv
from .config import SUITES, load_config
from .paths import ProjectPaths, ensure_run_layout
from .prepare_assets import validate_assets


CAMERAS = {"agentview_image": "front", "robot0_eye_in_hand_image": "wrist"}


def build_eval_command(policy, output, suite, c):
    return [
        "lerobot-eval",
        f"--policy.path={policy}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        "--env.is_libero_plus=true",
        f"--env.task={suite}",
        "--env.task_ids=" + json.dumps(list(c.task_ids), separators=(",", ":")),
        "--env.camera_name_mapping=" + json.dumps(CAMERAS, separators=(",", ":")),
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.control_mode=relative",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        f"--eval.n_episodes={c.episodes_per_task}",
        "--eval.use_async_envs=false",
        "--eval.recording=false",
        f"--seed={c.evaluation_seed}",
        f"--output_dir={output}",
    ]


def evaluate():
    paths = ProjectPaths.from_environment()
    run = ensure_run_layout(paths)
    validate_assets(paths)
    c = load_config(paths.project)
    policy = run / "model" / f"{paths.run_id()}_smolvla"
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(paths.data / "hf_cache"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MUJOCO_GL": "egl",
        }
    )
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    videos = {}
    commands = {}
    for suite in SUITES:
        out = run / "eval" / suite
        cmd = build_eval_command(policy, out, suite, c)
        commands[suite] = cmd
        subprocess.run(cmd, check=True, cwd=paths.project, env=env)
        rows.extend(result_rows(paths.run_id(), suite, read_json(out / "eval_info.json")))
        candidates = sorted(out.rglob("*.mp4"))
        if not candidates:
            raise FileNotFoundError(
                f"Evaluation did not produce a rollout video for {suite}, task 0"
            )
        name = {
            "libero_spatial": "spatial",
            "libero_object": "object",
            "libero_goal": "goal",
            "libero_10": "libero10",
        }[suite]
        target = run / f"{paths.run_id()}_{name}.mp4"
        shutil.copy2(candidates[0], target)
        videos[suite] = str(target)
    result = run / f"{paths.run_id()}_results.csv"
    write_results_csv(result, rows)
    write_json(
        run / "manifests/test.json",
        dict(
            run_id=paths.run_id(),
            evaluation_started_at=started,
            evaluation_finished_at=datetime.now(timezone.utc).isoformat(),
            suites=list(SUITES),
            episodes_per_task=c.episodes_per_task,
            results_csv_path=str(result),
            video_paths=videos,
            commands=commands,
        ),
    )
