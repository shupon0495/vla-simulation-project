from __future__ import annotations
import json
from pathlib import Path

TASK_CLASSIFICATION_FILENAME = 'task_classification.json'


def load_task_classification(path: Path) -> dict:
    """Load task_classification.json from the LIBERO-plus benchmark directory."""
    if not path.is_file():
        raise FileNotFoundError(
            f'task classification file missing: {path}; '
            'ensure LIBERO-plus source checkout includes libero/libero/benchmark/'
        )
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid task classification JSON: {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError(f'task classification must be an object: {path}')
    return data


def select_tasks(tasks: list[dict], n_tasks: int) -> list[dict]:
    """Select n_tasks entries preserving category ratios.

    Returns a list of dicts with keys: task_id, category, difficulty_level.
    task_id is 0-based (LeRobot convention).
    """
    total = len(tasks)
    if total == 0:
        raise ValueError('empty task list in classification data')

    # group by category, preserving original order (sorted by id)
    by_category: dict[str, list[dict]] = {}
    for task in sorted(tasks, key=lambda t: t['id']):
        by_category.setdefault(task['category'], []).append(task)

    cats = sorted(by_category.keys())
    counts = _category_counts(by_category, n_tasks, total)

    selected = []
    for cat in cats:
        indices = _evenly_spaced_indices(len(by_category[cat]), counts[cat])
        for idx in indices:
            task = by_category[cat][idx]
            selected.append({
                'task_id': task['id'] - 1,
                'category': cat,
                'difficulty_level': task['difficulty_level'],
            })

    return sorted(selected, key=lambda x: x['task_id'])


def _category_counts(by_category: dict, n_tasks: int, total: int) -> dict:
    cats = sorted(by_category.keys())
    counts = {}
    remaining = n_tasks
    for i, cat in enumerate(cats):
        if i < len(cats) - 1:
            n = round(n_tasks * len(by_category[cat]) / total)
            counts[cat] = n
            remaining -= n
        else:
            counts[cat] = remaining
    return counts


def _evenly_spaced_indices(total: int, n: int) -> list[int]:
    """Return n evenly spaced unique indices from [0, total)."""
    if n <= 0:
        return []
    if n == 1:
        indices = [0]
    else:
        step = (total - 1) / max(1, n - 1)
        indices = sorted({round(i * step) for i in range(n)})
    return indices
