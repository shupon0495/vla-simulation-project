import json
from pathlib import Path
from types import SimpleNamespace

from vla_simulation_project.artifacts import (
    result_rows,
    validate_policy_directory,
    write_results_csv,
)
from vla_simulation_project.evaluate import build_eval_command
from vla_simulation_project.paths import ProjectPaths
from vla_simulation_project.prepare_assets import validate_assets
from vla_simulation_project.preprocess import choose_evenly_spaced, normalize_task_name
from vla_simulation_project.train import build_train_command


def test_normalization_and_even_selection():
    assert normalize_task_name("Pick_up, Bowl!") == "pick up bowl"
    assert choose_evenly_spaced(list(range(9)), 5) == [0, 2, 4, 6, 8]


def test_train_command_is_offline_local(tmp_path):
    c = SimpleNamespace(learning_rate=.0003, final_learning_rate=.00003, warmup_steps=100, steps=3000, batch_size=1, seed=42, log_freq=100, lora_r=16, lora_alpha=16)
    cmd = build_train_command(c, {"base_model_path": "data/models/base", "vlm_path": "data/models/vlm", "dataset_path": "data/datasets/data", "selected_episode_indices": [1, 3]}, tmp_path)
    assert "--dataset.episodes=[1,3]" in cmd and "--policy.freeze_vision_encoder=true" in cmd and "--dataset.root=data/datasets/data" in cmd


def test_eval_command():
    c = SimpleNamespace(task_ids=(0, 4, 8), episodes_per_task=1, evaluation_seed=2026)
    cmd = build_eval_command(Path("model"), Path("out"), "libero_goal", c)
    assert "--env.task=libero_goal" in cmd and "--env.task_ids=[0,4,8]" in cmd and "--eval.n_episodes=1" in cmd


def test_results_aggregate_actual_successes(tmp_path):
    rows = result_rows("run", "libero_spatial", {"per_task": [{"task_id": 0, "metrics": {"successes": [True, False]}}, {"task_id": 4, "metrics": {"successes": [True]}}]})
    assert rows[-1]["n_trials"] == 3 and rows[-1]["n_success"] == 2 and rows[-1]["success_rate"] == 2 / 3
    out = tmp_path / "results.csv"
    write_results_csv(out, rows)
    assert out.read_text().count("libero_spatial") == 3


def test_dependency_files_are_unmodified():
    root = Path(__file__).parents[1]
    # Presence is checked here; git diff is checked by the validation command in CI/local review.
    assert all((root / name).is_file() for name in ("pyproject.toml", "uv.lock", "singularity/ubuntu24.04.def", "final_homework_Advanced.ipynb"))


def test_run_path_requires_managed_output(monkeypatch, tmp_path):
    monkeypatch.setenv("RUN_ID", "abc")
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "outside"))
    try:
        ProjectPaths(tmp_path).run_dir()
    except RuntimeError:
        pass
    else:
        assert False


def test_missing_asset_lock_fails(tmp_path):
    try:
        validate_assets(ProjectPaths(tmp_path))
    except FileNotFoundError as e:
        assert "login node" in str(e)
    else:
        assert False


def test_policy_contract_requires_processor_stats(tmp_path):
    for name in ("model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        (tmp_path / name).write_text("x")
    try:
        validate_policy_directory(tmp_path)
    except FileNotFoundError:
        pass
    else:
        assert False
    (tmp_path / "policy_preprocessor_stats.safetensors").write_text("x")
    (tmp_path / "policy_postprocessor_stats.safetensors").write_text("x")
    validate_policy_directory(tmp_path)
