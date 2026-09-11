from __future__ import annotations
import json, os
from pathlib import Path
from types import SimpleNamespace
import pytest
from vla_simulation_project.artifacts import result_rows, validate_final_artifacts, write_json, write_parameter_csv, write_results_csv
from vla_simulation_project.evaluate import build_eval_command
from vla_simulation_project.paths import ProjectPaths
from vla_simulation_project.prepare_assets import (_extract_assets, _localize_tokenizer_config,
                                                   _snapshot, tqdm, validate_assets)
from vla_simulation_project.preprocess import choose_evenly_spaced, normalize_task_name, select_spatial_episodes
from vla_simulation_project.train import build_train_command
from vla_simulation_project.config import SPATIAL_TASK_NAMES

def config():
    return SimpleNamespace(steps=3000, batch_size=1, learning_rate=.0003, final_learning_rate=.00003,
        warmup_steps=100, lora_r=16, lora_alpha=16, log_freq=100, seed=42,
        task_ids=(0, 4, 8), episodes_per_task=1, evaluation_seed=2026, video_task_id=0)

def test_run_path_and_run_id(monkeypatch, tmp_path):
    run = tmp_path / "data/outputs/20260910T000000Z-ab12"
    monkeypatch.setenv("RUN_ID", "ab12"); monkeypatch.setenv("RUN_DIR", str(run))
    paths = ProjectPaths(tmp_path)
    assert paths.run_id() == "ab12" and paths.run_dir() == run
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(RuntimeError, match="child"): paths.run_dir()

def test_normalization_and_deterministic_selection():
    assert normalize_task_name("Pick_up,  Bowl!") == "pick up bowl"
    assert choose_evenly_spaced(list(range(9)), 5) == [0, 2, 4, 6, 8]
    tasks = [name for name in SPATIAL_TASK_NAMES for _ in range(9)]
    first = select_spatial_episodes(tasks); second = select_spatial_episodes(tasks)
    assert first == second and len(first[0]) == 50

def test_asset_manifest_validation_and_missing_failure(tmp_path):
    paths = ProjectPaths(tmp_path)
    with pytest.raises(FileNotFoundError, match="login node"): validate_assets(paths)
    base = tmp_path / "data/models/base"; dataset = tmp_path / "data/datasets/set"; vlm = tmp_path / "data/models/vlm"; assets = tmp_path / "data/assets/libero/assets"
    for directory in (base, dataset / "meta", dataset / "data", dataset / "videos", vlm, assets): directory.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "policy_preprocessor_stats.safetensors", "policy_postprocessor.json", "policy_postprocessor_stats.safetensors"): (base / name).write_text("x")
    (base / "policy_preprocessor.json").write_text(json.dumps({"steps": [{"config": {"tokenizer_name": "data/models/vlm"}}]}))
    (dataset / "meta/info.json").write_text("x"); (dataset / "data/a.parquet").write_text("x"); (dataset / "videos/a.mp4").write_text("x")
    for name in ("config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json"): (vlm / name).write_text("x")
    (assets / "arena.xml").write_text("x")
    from vla_simulation_project.config import BASE_MODEL_REPO, BASE_MODEL_REVISION, DATASET_REPO, DATASET_REVISION, VLM_REPO
    write_json(tmp_path / "data/manifests/assets.lock.json", {"base_model": {"repo": BASE_MODEL_REPO, "revision": BASE_MODEL_REVISION, "local_path": "data/models/base"},
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "local_path": "data/datasets/set"},
        "vlm": {"repo": VLM_REPO, "revision": "abc", "local_path": "data/models/vlm"}, "libero_assets": {"local_path": "data/assets/libero/assets"}})
    assert validate_assets(paths)["vlm"]["revision"] == "abc"

def test_localize_tokenizer_config_uses_staged_vlm_path(tmp_path):
    base = tmp_path / "base"; base.mkdir()
    config_path = base / "policy_preprocessor.json"
    config_path.write_text(json.dumps({"steps": [{"registry_name": "tokenizer_processor", "config": {
        "tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "max_length": 48}}]}))

    _localize_tokenizer_config(base, "data/models/smolvlm2_500m")

    localized = json.loads(config_path.read_text())
    assert localized["steps"][0]["config"]["tokenizer_name"] == "data/models/smolvlm2_500m"

def test_train_and_eval_commands_are_local_and_semantic(tmp_path):
    pre = {"base_model_path": "data/models/base", "vlm_path": "data/models/vlm", "dataset_path": "data/datasets/set", "selected_episode_indices": [1, 3]}
    train = build_train_command(config(), pre, tmp_path)
    assert "--dataset.root=data/datasets/set" in train and "--dataset.episodes=[1,3]" in train
    assert "--policy.freeze_vision_encoder=true" in train and "--wandb.enable=false" in train
    assert "--policy.vlm_model_name=data/models/vlm" in train
    evaluate = build_eval_command(Path("model"), Path("out"), "libero_goal", config())
    assert "--env.task=libero_goal" in evaluate and "--env.task_ids=[0,4,8]" in evaluate and "--eval.n_episodes=1" in evaluate

def test_results_use_actual_episode_success_values(tmp_path):
    info = {"per_task": [{"task_id": 0, "metrics": {"successes": [True, False]}}, {"task_id": 4, "metrics": {"successes": [True]}}]}
    rows = result_rows("run1", "libero_spatial", info)
    assert rows[-1]["n_trials"] == 3 and rows[-1]["n_success"] == 2 and rows[-1]["success_rate"] == 2 / 3

def _complete_policy(path):
    path.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "policy_preprocessor.json", "policy_preprocessor_stats.safetensors", "policy_postprocessor.json", "policy_postprocessor_stats.safetensors"): (path / name).write_text("x")

def test_final_artifact_contract(tmp_path):
    run, run_id = tmp_path, "r1"; _complete_policy(run / "model/r1_smolvla")
    for name in ("r1_model.tar.gz", "r1_spatial.mp4", "r1_object.mp4", "r1_goal.mp4", "r1_libero10.mp4"): (run / name).write_text("x")
    write_parameter_csv(run / "r1_parameters.csv", run_id, {"steps": 3000})
    for name in ("run.json", "preprocess.json", "train.json", "test.json"): write_json(run / "manifests" / name, {"run_id": run_id})
    rows = []
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"): rows += result_rows(run_id, suite, {"per_task": [{"task_id": 0, "metrics": {"successes": [True]}}]})
    write_results_csv(run / "r1_results.csv", rows)
    validate_final_artifacts(run, run_id, ("libero_spatial", "libero_object", "libero_goal", "libero_10"))

def test_dependency_files_present_and_not_part_of_worktree_diff():
    root = Path(__file__).parents[1]
    for name in ("pyproject.toml", "uv.lock", "singularity/ubuntu24.04.def", "final_homework_Advanced.ipynb"): assert (root / name).is_file()

def test_asset_preparation_uses_dedicated_python312_login_image():
    root = Path(__file__).parents[1]
    submit = (root / "submit_pipeline.sh").read_text()
    login_def = (root / "singularity/login.def").read_text()
    assert '"$LOGIN_SIF"' in submit
    assert '"$PROJECT/.venv/bin/python" -m vla_simulation_project.main prepare-assets' in submit
    assert "python3 -m vla_simulation_project.main prepare-assets" not in submit
    assert "From: ubuntu:24.04" in login_def
    assert "python3.12 -c 'import tomllib'" in login_def

def test_slurm_log_paths_use_submission_timestamp():
    submit = (Path(__file__).parents[1] / "submit_pipeline.sh").read_text()
    for stage in ("build", "preprocess", "train", "test"):
        assert f'{stage}-${{TIMESTAMP}}-%j.out' in submit
        assert f'{stage}-${{TIMESTAMP}}-%j.err' in submit
    assert "%Y-%m-%d" not in submit

def test_snapshot_download_enables_progress_and_uses_resolved_revision(monkeypatch, tmp_path, capsys):
    calls = {}
    monkeypatch.setattr("vla_simulation_project.prepare_assets.HfApi.repo_info",
        lambda *args, **kwargs: SimpleNamespace(sha="resolved-sha"))
    monkeypatch.setattr("vla_simulation_project.prepare_assets.enable_progress_bars",
        lambda: calls.setdefault("progress_enabled", True))
    monkeypatch.setattr("vla_simulation_project.prepare_assets.snapshot_download",
        lambda **kwargs: calls.update(kwargs) or str(tmp_path))

    assert _snapshot("owner/data", "dataset", "fixed", tmp_path, allow=("*.parquet",)) == "resolved-sha"
    assert calls["progress_enabled"] is True
    assert calls["revision"] == "resolved-sha"
    assert calls["local_dir"] == tmp_path
    assert calls["allow_patterns"] == ["*.parquet"]
    assert calls["tqdm_class"] is tqdm
    assert "Preparing Hugging Face snapshot" in capsys.readouterr().out

def test_asset_extraction_shows_byte_progress(tmp_path, capsys):
    import zipfile
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("bundle/assets/arena.xml", "x" * 32)

    target = tmp_path / "libero/assets"
    _extract_assets(archive, target)

    assert (target / "arena.xml").read_text() == "x" * 32
    assert "Extracting LIBERO assets" in capsys.readouterr().err
