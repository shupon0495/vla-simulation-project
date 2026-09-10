"""Login-node preparation implemented with the Python standard library only."""
from __future__ import annotations
import json, os, shutil, urllib.parse, urllib.request, zipfile
from pathlib import Path
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
def _valid_vlm(path: Path) -> bool: return bool(_files(path, "config.json") and _files(path, "*.safetensors"))
def _valid_dataset(path: Path) -> bool:
    return bool((_files(path, "info.json") or _files(path, "episodes.jsonl") or _files(path, "*.parquet")) and (_files(path, "*.parquet")) and (_files(path, "*.mp4")))
def _valid_assets(path: Path) -> bool: return path.is_dir() and bool(_files(path, "*.xml") or _files(path, "*.bddl"))

def validate_assets(paths: ProjectPaths) -> dict:
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
    item = lock.get("libero_assets", {}); local = paths.project / str(item.get("local_path", ""))
    if not _valid_assets(local): raise FileNotFoundError(f"missing/invalid LIBERO-plus assets at {local}; rerun prepare-assets on the login node")
    return lock

def _request_json(url: str) -> dict:
    headers = {"User-Agent": "vla-simulation-project/asset-preparer"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token: headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response: return json.load(response)

def _repo_info(repo: str, repo_type: str, revision: str | None) -> dict:
    kind = "datasets" if repo_type == "dataset" else "models"
    suffix = f"/revision/{urllib.parse.quote(revision, safe='')}" if revision else ""
    return _request_json(f"https://huggingface.co/api/{kind}/{repo}{suffix}")

def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True); temporary = target.with_suffix(target.suffix + ".partial")
    headers = {"User-Agent": "vla-simulation-project/asset-preparer"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token: headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as source, temporary.open("wb") as output: shutil.copyfileobj(source, output)
        temporary.replace(target)
    finally:
        if temporary.exists(): temporary.unlink()

def _snapshot(repo: str, repo_type: str, revision: str | None, target: Path, *, allow=None, ignore=None) -> str:
    info = _repo_info(repo, repo_type, revision); resolved = info["sha"]
    for sibling in info.get("siblings", []):
        name = sibling["rfilename"]
        if allow and not any(Path(name).match(pattern) for pattern in allow): continue
        if ignore and any(Path(name).match(pattern) for pattern in ignore): continue
        destination = target / name
        expected_size = sibling.get("size") or sibling.get("lfs", {}).get("size")
        if destination.is_file() and destination.stat().st_size and (not expected_size or destination.stat().st_size == expected_size): continue
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in name.split("/"))
        _download(f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo}/resolve/{resolved}/{encoded}?download=true", destination)
    return resolved

def _extract_assets(archive: Path, target: Path) -> None:
    staging = target.parent / ".libero-assets-extract"
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        root = staging.resolve()
        for member in bundle.infolist():
            destination = (staging / member.filename).resolve()
            if destination != root and root not in destination.parents: raise ValueError(f"unsafe path in LIBERO assets archive: {member.filename}")
        bundle.extractall(staging)
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
    assets = paths.data / "assets/libero_plus/assets"
    assets_revision = existing.get("libero_assets", {}).get("revision")
    if not assets_revision: assets_revision = _repo_info(LIBERO_ASSETS_REPO, "dataset", None)["sha"]
    if not _valid_assets(assets):
        archive = paths.data / "assets/libero_plus/assets.zip"
        info = _repo_info(LIBERO_ASSETS_REPO, "dataset", assets_revision); assets_revision = info["sha"]
        if not archive.is_file() or not archive.stat().st_size:
            _download(f"https://huggingface.co/datasets/{LIBERO_ASSETS_REPO}/resolve/{assets_revision}/assets.zip?download=true", archive)
        _extract_assets(archive, assets)
    lock = {"base_model": {"repo": BASE_MODEL_REPO, "revision": BASE_MODEL_REVISION, "local_path": _relative(paths, base)},
            "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "local_path": _relative(paths, dataset)},
            "vlm": {"repo": VLM_REPO, "revision": vlm_revision, "local_path": _relative(paths, vlm)},
            "libero_assets": {"repo": LIBERO_ASSETS_REPO, "revision": assets_revision, "local_path": _relative(paths, assets)}}
    write_json(lock_path, lock); validate_assets(paths)
