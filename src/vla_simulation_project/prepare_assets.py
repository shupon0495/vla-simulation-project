"""Login-node-only Hugging Face preparation and compute-safe validation."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .artifacts import write_json
from .config import (
    BASE_MODEL_REPO,
    BASE_MODEL_REVISION,
    DATASET_REPO,
    DATASET_REVISION,
    VLM_REPO,
)
from .paths import ProjectPaths


LOCK = "assets.lock.json"


def _relative(paths, p):
    return str(p.resolve().relative_to(paths.project))


def _nonempty(p):
    return p.is_dir() and any(p.rglob("*"))


def _has_file(p, patterns):
    return any(any(p.rglob(pattern)) for pattern in patterns)


def validate_assets(paths: ProjectPaths):
    lock = paths.data / "manifests" / LOCK
    if not lock.is_file():
        raise FileNotFoundError("Offline asset lock missing; run submit_pipeline.sh on the login node before Slurm submission")
    data = json.loads(lock.read_text())
    expected = (("base_model", BASE_MODEL_REPO, BASE_MODEL_REVISION), ("dataset", DATASET_REPO, DATASET_REVISION), ("vlm", VLM_REPO, None))
    for key, repo, rev in expected:
        item = data.get(key, {})
        path = paths.project / item.get("local_path", "")
        valid = _nonempty(path)
        if key == "base_model":
            valid = valid and _has_file(path, ("config.json", "model.safetensors", "model*.safetensors"))
        if key == "dataset":
            valid = valid and _has_file(path, ("info.json", "meta/info.json", "episodes.jsonl", "*.parquet"))
        if item.get("repo") != repo or (rev and item.get("revision") != rev) or not valid:
            raise FileNotFoundError(f"Missing/invalid offline {key} at {path}; prepare {repo} on the login node")
    assets = paths.project / data.get("libero_assets", {}).get("local_path", "")
    if not _nonempty(assets):
        raise FileNotFoundError(f"Missing LIBERO-plus assets at {assets}; run login-node asset preparation")
    return data


def _snapshot(repo, revision, target):
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(repo_id=repo, repo_type="dataset" if repo == DATASET_REPO else "model", revision=revision, local_dir=target))


def prepare_assets():
    paths = ProjectPaths.from_environment()
    paths.data.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    base = paths.data / "models" / "smolvla_libero_plus"
    dataset = paths.data / "datasets" / "libero_plus"
    vlm = paths.data / "models" / "smolvlm2_500m"
    try:
        existing = validate_assets(paths)
    except FileNotFoundError:
        existing = {}
    if not _nonempty(base):
        _snapshot(BASE_MODEL_REPO, BASE_MODEL_REVISION, base)
    if not _nonempty(dataset):
        _snapshot(DATASET_REPO, DATASET_REVISION, dataset)
    vlm_revision = existing.get("vlm", {}).get("revision")
    if not _nonempty(vlm):
        from huggingface_hub import snapshot_download
        from huggingface_hub import HfApi, snapshot_download

        vlm_revision = HfApi().model_info(VLM_REPO, revision=vlm_revision).sha
        snapshot_download(repo_id=VLM_REPO, revision=vlm_revision, local_dir=vlm)
    if not vlm_revision:
        vlm_revision = existing.get("vlm", {}).get("revision")
        if not vlm_revision:
            raise RuntimeError("Existing VLM snapshot has no recorded resolved revision; remove only its managed model directory and prepare again")
    assets = paths.data / "assets" / "libero_plus" / "assets"
    if not _nonempty(assets):
        from huggingface_hub import hf_hub_download

        archive = Path(hf_hub_download(repo_id="Sylvest/LIBERO-plus", repo_type="dataset", filename="assets.zip", local_dir=assets.parent))
        shutil.unpack_archive(archive, assets.parent)
        candidates = [p for p in assets.parent.rglob("assets") if p.is_dir() and _nonempty(p)]
        if not candidates:
            raise FileNotFoundError("LIBERO-plus assets.zip did not contain an assets directory")
        if candidates[0].resolve() != assets.resolve():
            shutil.move(str(candidates[0]), str(assets))
    lock = {"base_model": {"repo": BASE_MODEL_REPO, "revision": BASE_MODEL_REVISION, "local_path": _relative(paths, base)}, "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "local_path": _relative(paths, dataset)}, "vlm": {"repo": VLM_REPO, "revision": vlm_revision, "local_path": _relative(paths, vlm)}, "libero_assets": {"local_path": _relative(paths, assets)}}
    write_json(paths.data / "manifests" / LOCK, lock)
    validate_assets(paths)
