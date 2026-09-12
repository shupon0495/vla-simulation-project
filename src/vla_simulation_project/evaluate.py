from __future__ import annotations
import json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
from .artifacts import read_json, result_rows, validate_final_artifacts, validate_policy_directory, write_json, write_results_csv
from .config import SUITES, ExperimentConfig, load_config
from .paths import ProjectPaths, ensure_run_layout
from .prepare_assets import _resource_root, validate_assets
from .subprocess_utils import run_command
from .train import offline_environment
CAMERAS = {'agentview_image': 'front', 'robot0_eye_in_hand_image': 'wrist'}


def add_libero_source_to_pythonpath(env: dict[str, str], source: Path) -> None:
    """Expose the prepared LIBERO-plus namespace package to lerobot-eval.

    The locked LIBERO-plus revision does not package its outer ``libero``
    namespace into a wheel.  Its validated login-node checkout is therefore
    the offline import source used by the evaluation subprocess.
    """
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(source) if not existing else f'{source}{os.pathsep}{existing}'


def build_eval_command(policy: Path, output: Path, suite: str, config: ExperimentConfig, vlm_path: Path) -> list[str]:
    return ['lerobot-eval', f'--policy.path={policy}', f'--policy.vlm_model_name={vlm_path}', '--policy.device=cuda', '--policy.use_amp=false', '--env.type=libero', '--env.is_libero_plus=true', f'--env.task={suite}', '--env.task_ids=' + json.dumps(list(config.task_ids), separators=(',', ':')), '--env.camera_name_mapping=' + json.dumps(CAMERAS, separators=(',', ':')), '--env.observation_height=256', '--env.observation_width=256', '--env.control_mode=relative', '--env.max_parallel_tasks=1', '--eval.batch_size=1', f'--eval.n_episodes={config.episodes_per_task}', '--eval.use_async_envs=false', '--eval.recording=false', f'--seed={config.evaluation_seed}', f'--output_dir={output}']

def create_libero_config(run: Path, source: Path, assets: Path) -> Path:
    """Configure uv-managed LIBERO Python to use login-prepared benchmark data."""
    resources = _resource_root(source)
    if not (resources / 'bddl_files').is_dir() or not (resources / 'init_files').is_dir():
        raise FileNotFoundError(f'test: prepared LIBERO-plus benchmark resources are incomplete under {resources}')
    if not assets.is_dir():
        raise FileNotFoundError(f'test: prepared LIBERO-plus assets are missing at {assets}')
    config_dir = run / 'libero_config'
    config_dir.mkdir(parents=True, exist_ok=True)
    content = '\n'.join((f'benchmark_root: {resources}', f'assets: {assets}', f'bddl_files: {resources / 'bddl_files'}', f'datasets: {resources.parent / 'datasets'}', f'init_states: {resources / 'init_files'}')) + '\n'
    (config_dir / 'config.yaml').write_text(content, encoding='utf-8')
    return config_dir

def _task_video(info: dict, task_id: int, output: Path) -> Path:
    for task in info.get('per_task', []):
        if int(task.get('task_id', -1)) == task_id:
            paths = task.get('metrics', {}).get('video_paths', [])
            if paths:
                candidate = Path(paths[0])
                for possible in (candidate, output / candidate):
                    if possible.is_file() and possible.stat().st_size:
                        return possible
    directory = output / 'videos' / f'{info.get('task_group', '')}_{task_id}'
    candidates = sorted(directory.rglob('*.mp4')) if directory.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f'test: evaluation did not produce a video for task {task_id} under {output}')
    return candidates[0]

def evaluate() -> None:
    paths = ProjectPaths.from_environment()
    run = ensure_run_layout(paths)
    lock = validate_assets(paths)
    train_manifest = read_json(run / 'manifests/train.json')
    if train_manifest.get('run_id') != paths.run_id():
        raise ValueError('test: train manifest RUN_ID mismatch')
    policy = run / 'model' / f'{paths.run_id()}_smolvla'
    validate_policy_directory(policy)
    config = load_config(paths.project)
    env = offline_environment(paths)
    env['MUJOCO_GL'] = 'egl'
    libero_source = paths.project / lock['libero_source']['local_path']
    vlm_path = paths.project / lock['vlm']['local_path']
    env['LIBERO_CONFIG_PATH'] = str(create_libero_config(run, libero_source, paths.project / lock['libero_assets']['local_path']))
    add_libero_source_to_pythonpath(env, libero_source)
    started = datetime.now(timezone.utc).isoformat()
    rows, videos, commands = ([], {}, {})
    names = {'libero_spatial': 'spatial', 'libero_object': 'object', 'libero_goal': 'goal', 'libero_10': 'libero10'}
    for suite in SUITES:
        output = run / 'eval' / suite
        command = build_eval_command(policy, output, suite, config, vlm_path)
        commands[suite] = command
        run_command('test', command, cwd=paths.project, env=env)
        info = read_json(output / 'eval_info.json')
        rows.extend(result_rows(paths.run_id(), suite, info))
        target = run / f'{paths.run_id()}_{names[suite]}.mp4'
        shutil.copy2(_task_video(info, config.video_task_id, output), target)
        videos[suite] = str(target)
    results = run / f'{paths.run_id()}_results.csv'
    write_results_csv(results, rows)
    write_json(run / 'manifests/test.json', {'run_id': paths.run_id(), 'stage': 'test', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'evaluation_started_at': started, 'evaluation_finished_at': datetime.now(timezone.utc).isoformat(), 'suites': list(SUITES), 'task_ids': list(config.task_ids), 'episodes_per_task': config.episodes_per_task, 'results_csv_path': str(results), 'video_paths': videos, 'commands': commands})
    validate_final_artifacts(run, paths.run_id(), SUITES)
