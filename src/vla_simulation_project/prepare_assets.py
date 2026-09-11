"""Download, localize, and validate assets on the Internet-connected login node."""
from __future__ import annotations
import json
import shutil, zipfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import enable_progress_bars
from tqdm.auto import tqdm

from .artifacts import read_json, write_json
from .config import (BASE_MODEL_REPO, BASE_MODEL_REVISION, DATASET_REPO, DATASET_REVISION,
                     LIBERO_ASSETS_REPO, VLM_REPO)
from .paths import ProjectPaths

LOCK_NAME = "assets.lock.json"

def _relative(paths: ProjectPaths, path: Path) -> str: return str(path.resolve().relative_to(paths.project))
def _files(path: Path, pattern: str) -> list[Path]: return [p for p in path.rglob(pattern) if p.is_file() and p.stat().st_size]
def _valid_base(path: Path) -> bool:
    return bool(_files(path, "config.json") and _files(path, "model*.safetensors") and _files(path, "policy_preprocessor.json")
                and _files(path, "policy_postprocessor.json") and _files(path, "policy_preprocessor*.safetensors")
                and _files(path, "policy_postprocessor*.safetensors"))
def _valid_vlm(path: Path) -> bool:
    return bool(
        _files(path, "config.json")
        and _files(path, "*.safetensors")
        and _files(path, "tokenizer_config.json")
        and _files(path, "tokenizer.json")
    )
def _valid_dataset(path: Path) -> bool:
    return bool((_files(path, "info.json") or _files(path, "episodes.jsonl") or _files(path, "*.parquet")) and (_files(path, "*.parquet")) and (_files(path, "*.mp4")))
def _valid_assets(path: Path) -> bool: return path.is_dir() and bool(_files(path, "*.xml") or _files(path, "*.bddl"))

def _localize_tokenizer_config(base: Path, vlm_path: str) -> None:
    """Point serialized tokenizer processor steps at the staged local VLM."""
    config_path = base / "policy_preprocessor.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid policy preprocessor config at {config_path}") from exc

    tokenizer_entries: list[dict] = []
    pending = [config]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if "tokenizer_name" in value:
                tokenizer_entries.append(value)
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if not tokenizer_entries:
        raise ValueError(f"no tokenizer_name found in policy preprocessor config at {config_path}")
    for entry in tokenizer_entries:
        entry["tokenizer_name"] = vlm_path
    write_json(config_path, config)

def _tokenizer_config_is_local(base: Path, vlm_path: str) -> bool:
    try:
        config = json.loads((base / "policy_preprocessor.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    values: list[str] = []
    pending = [config]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if "tokenizer_name" in value: values.append(value["tokenizer_name"])
            pending.extend(value.values())
        elif isinstance(value, list): pending.extend(value)
    return bool(values) and all(value == vlm_path for value in values)

def validate_assets(paths: ProjectPaths, *, repair_tokenizer: bool = False) -> dict:
    lock_path = paths.data / "manifests" / LOCK_NAME
    if not lock_path.is_file(): raise FileNotFoundError(f"offline asset lock missing at {lock_path}; run prepare-assets on the Internet-connected login node")
    lock = read_json(lock_path)
    checks = (("base_model", BASE_MODEL_REPO, BASE_MODEL_REVISION, _valid_base),
              ("dataset", DATASET_REPO, DATASET_REVISION, _valid_dataset),
              ("vlm", VLM_REPO, None, _valid_vlm))
    for key, repo, revision, validator in checks:
        item = lock.get(key, {}); relative = item.get("local_path"); local = paths.project / str(relative or "")
        if not relative or item.get("repo") != repo or (revision and item.get("revision") != revision) or not item.get("revision") or not validator(local):
            raise FileNotFoundError(f"missing/invalid offline {key} for {repo}@{revision or item.get('revision')} at {local}; rerun prepare-assets on the login node")
    base_path = paths.project / lock["base_model"]["local_path"]
    vlm_path = lock["vlm"]["local_path"]
    if repair_tokenizer and not _tokenizer_config_is_local(base_path, vlm_path):
        managed_models = (paths.data / "models").resolve()
        if base_path.resolve() != managed_models and managed_models not in base_path.resolve().parents:
            raise ValueError(f"refusing to modify tokenizer config outside {managed_models}: {base_path}")
        _localize_tokenizer_config(base_path, vlm_path)
    if not _tokenizer_config_is_local(base_path, vlm_path):
        raise FileNotFoundError(
            f"offline tokenizer path is not localized to {vlm_path} in {base_path / 'policy_preprocessor.json'}; "
            "rerun prepare-assets on the login node"
        )
    item = lock.get("libero_assets", {}); local = paths.project / str(item.get("local_path", ""))
    if not _valid_assets(local): raise FileNotFoundError(f"missing/invalid LIBERO-plus assets at {local}; rerun prepare-assets on the login node")
    return lock

def _snapshot(repo: str, repo_type: str, revision: str | None, target: Path, *, allow=None, ignore=None) -> str:
    """Download one immutable Hub snapshot with visible file/byte progress bars."""
    resolved = HfApi().repo_info(repo_id=repo, repo_type=repo_type, revision=revision).sha
    print(f"Preparing Hugging Face snapshot: {repo}@{resolved} -> {target}", flush=True)
    enable_progress_bars()
    snapshot_download(
        repo_id=repo,
        repo_type=repo_type,
        revision=resolved,
        local_dir=target,
        allow_patterns=list(allow) if allow else None,
        ignore_patterns=list(ignore) if ignore else None,
        tqdm_class=tqdm,
    )
    print(f"Hugging Face snapshot ready: {repo}@{resolved}", flush=True)
    return resolved

def _extract_assets(archive: Path, target: Path) -> None:
    staging = target.parent / ".libero-assets-extract"
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        root = staging.resolve()
        members = bundle.infolist()
        for member in members:
            destination = (staging / member.filename).resolve()
            if destination != root and root not in destination.parents: raise ValueError(f"unsafe path in LIBERO assets archive: {member.filename}")
        with tqdm(total=sum(member.file_size for member in members), desc="Extracting LIBERO assets",
                  unit="B", unit_scale=True, unit_divisor=1024) as progress:
            for member in members:
                destination = staging / member.filename
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))
    candidates = [p for p in staging.rglob("assets") if _valid_assets(p)]
    if not candidates: raise FileNotFoundError(f"downloaded archive has no valid LIBERO assets directory: {archive}")
    if target.exists(): shutil.rmtree(target)
    shutil.move(str(candidates[0]), target); shutil.rmtree(staging)

def prepare_assets() -> None:
    paths = ProjectPaths.from_environment(); (paths.data / "manifests").mkdir(parents=True, exist_ok=True)
    base, dataset, vlm = paths.data / "models/smolvla_libero_plus", paths.data / "datasets/libero_plus", paths.data / "models/smolvlm2_500m"
    lock_path = paths.data / "manifests" / LOCK_NAME
    existing = read_json(lock_path) if lock_path.is_file() else {}
    if not _valid_base(base): _snapshot(BASE_MODEL_REPO, "model", BASE_MODEL_REVISION, base, allow=("config.json", "model*.safetensors", "train_config.json", "policy_preprocessor*", "policy_postprocessor*"), ignore=("README.md", "eval/**"))
    if not _valid_dataset(dataset): _snapshot(DATASET_REPO, "dataset", DATASET_REVISION, dataset)
    vlm_revision = existing.get("vlm", {}).get("revision")
    if not _valid_vlm(vlm): vlm_revision = _snapshot(VLM_REPO, "model", vlm_revision, vlm)
    if not vlm_revision: raise RuntimeError("valid VLM files exist but assets.lock.json has no resolved revision; move the managed VLM directory aside and rerun preparation")
    _localize_tokenizer_config(base, _relative(paths, vlm))
    assets = paths.data / "assets/libero_plus/assets"
    assets_revision = existing.get("libero_assets", {}).get("revision")
    if not _valid_assets(assets):
        archive_root = paths.data / "assets/libero_plus"
        archive = archive_root / "assets.zip"
        assets_revision = _snapshot(
            LIBERO_ASSETS_REPO, "dataset", assets_revision, archive_root, allow=("assets.zip",)
        )
        _extract_assets(archive, assets)
    elif not assets_revision:
        assets_revision = HfApi().repo_info(repo_id=LIBERO_ASSETS_REPO, repo_type="dataset").sha
    lock = {"base_model": {"repo": BASE_MODEL_REPO, "revision": BASE_MODEL_REVISION, "local_path": _relative(paths, base)},
            "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "local_path": _relative(paths, dataset)},
            "vlm": {"repo": VLM_REPO, "revision": vlm_revision, "local_path": _relative(paths, vlm)},
            "libero_assets": {"repo": LIBERO_ASSETS_REPO, "revision": assets_revision, "local_path": _relative(paths, assets)}}
    write_json(lock_path, lock); validate_assets(paths, repair_tokenizer=True)
