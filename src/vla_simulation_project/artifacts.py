from __future__ import annotations
import csv, json, tarfile
from pathlib import Path

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)

def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'required manifest missing: {path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'manifest must contain an object: {path}')
    return value

def validate_policy_directory(path: Path) -> None:
    required = ('model.safetensors', 'config.json', 'policy_preprocessor.json', 'policy_postprocessor.json')
    missing = [name for name in required if not (path / name).is_file() or not (path / name).stat().st_size]
    for pattern in ('policy_preprocessor*.safetensors', 'policy_postprocessor*.safetensors'):
        if not any((p.is_file() and p.stat().st_size for p in path.glob(pattern))):
            missing.append(pattern)
    if missing:
        raise FileNotFoundError(f'incomplete merged policy at {path}; missing/non-empty required files: {missing}')

def archive_model(model: Path, archive: Path) -> None:
    validate_policy_directory(model)
    with tarfile.open(archive, 'w:gz') as output:
        output.add(model, arcname=model.name)

def result_rows(run_id: str, suite: str, info: dict) -> list[dict]:
    rows, total_trials, total_success = ([], 0, 0)
    for task in info.get('per_task', []):
        successes = [bool(value) for value in task.get('metrics', {}).get('successes', [])]
        if not successes:
            raise ValueError(f'no episode success values for {suite} task {task.get('task_id')}')
        count, succeeded = (len(successes), sum(successes))
        total_trials += count
        total_success += succeeded
        rows.append({'run_id': run_id, 'suite': suite, 'task_id': int(task['task_id']), 'level': 'task', 'n_trials': count, 'n_success': succeeded, 'success_rate': succeeded / count})
    if not rows:
        raise ValueError(f'no task results in {suite} eval_info.json')
    rows.append({'run_id': run_id, 'suite': suite, 'task_id': '', 'level': 'suite', 'n_trials': total_trials, 'n_success': total_success, 'success_rate': total_success / total_trials})
    return rows
FIELDS = ('run_id', 'suite', 'task_id', 'level', 'n_trials', 'n_success', 'success_rate')

def write_results_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def write_parameter_csv(path: Path, run_id: str, values: dict) -> None:
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=('run_id', 'parameter', 'value'))
        writer.writeheader()
        writer.writerows(({'run_id': run_id, 'parameter': key, 'value': value} for key, value in sorted(values.items())))

def validate_final_artifacts(run: Path, run_id: str, suites: tuple[str, ...]) -> None:
    required = [run / f'{run_id}_model.tar.gz', run / f'{run_id}_parameters.csv', run / f'{run_id}_results.csv', *(run / f'{run_id}_{name}.mp4' for name in ('spatial', 'object', 'goal', 'libero10')), *(run / 'manifests' / name for name in ('run.json', 'preprocess.json', 'train.json', 'test.json'))]
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError(f'run artifact contract is incomplete: {missing}')
    validate_policy_directory(run / 'model' / f'{run_id}_smolvla')
    for name in ('run.json', 'preprocess.json', 'train.json', 'test.json'):
        if read_json(run / 'manifests' / name).get('run_id') != run_id:
            raise ValueError(f'RUN_ID mismatch in manifests/{name}')
    with (run / f'{run_id}_results.csv').open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    if any((row['run_id'] != run_id for row in rows)):
        raise ValueError('RUN_ID mismatch in results CSV')
    present = {row['suite'] for row in rows if row['level'] == 'suite'}
    if present != set(suites):
        raise ValueError(f'results CSV suite rows mismatch: {sorted(present)}')
    for row in rows:
        trials, successes, rate = (int(row['n_trials']), int(row['n_success']), float(row['success_rate']))
        if trials < 1 or not 0 <= successes <= trials or abs(rate - successes / trials) > 1e-12:
            raise ValueError(f'invalid results row: {row}')
    with (run / f'{run_id}_parameters.csv').open(newline='', encoding='utf-8') as source:
        parameters = list(csv.DictReader(source))
    if not parameters or any((row.get('run_id') != run_id for row in parameters)):
        raise ValueError('RUN_ID mismatch or no rows in parameters CSV')
