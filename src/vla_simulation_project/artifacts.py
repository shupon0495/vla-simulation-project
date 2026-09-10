from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest missing: {path}")
    return json.loads(path.read_text())


def validate_policy_directory(p: Path):
    missing = [
        x
        for x in (
            "model.safetensors",
            "config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        if not (p / x).is_file() or not (p / x).stat().st_size
    ]
    missing += [
        x
        for x in (
            "policy_preprocessor*.safetensors",
            "policy_postprocessor*.safetensors",
        )
        if not any(q.is_file() and q.stat().st_size for q in p.glob(x))
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete policy artifact {p}: {missing}")


def archive_model(model, archive):
    validate_policy_directory(model)
    with tarfile.open(archive, "w:gz") as f:
        f.add(model, arcname=model.name)


def result_rows(run_id, suite, info):
    rows = []
    trials = success = 0
    for task in info.get("per_task", []):
        outcomes = [bool(x) for x in task["metrics"]["successes"]]
        n = len(outcomes)
        if not n:
            raise ValueError(f"No trials in {suite}")
        s = sum(outcomes)
        trials += n
        success += s
        rows.append(
            dict(
                run_id=run_id,
                suite=suite,
                task_id=int(task["task_id"]),
                level="task",
                n_trials=n,
                n_success=s,
                success_rate=s / n,
            )
        )
    if not rows:
        raise ValueError(f"No task results in {suite}")
    return rows + [
        dict(
            run_id=run_id,
            suite=suite,
            task_id="",
            level="suite",
            n_trials=trials,
            n_success=success,
            success_rate=success / trials,
        )
    ]


def write_results_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=(
                "run_id",
                "suite",
                "task_id",
                "level",
                "n_trials",
                "n_success",
                "success_rate",
            ),
        )
        w.writeheader()
        w.writerows(rows)


def write_parameter_csv(path, run_id, values):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("run_id", "parameter", "value"))
        w.writeheader()
        w.writerows(
            {"run_id": run_id, "parameter": k, "value": v}
            for k, v in sorted(values.items())
        )
