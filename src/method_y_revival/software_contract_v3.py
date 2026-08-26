from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .contracts import sha256_mapping


class SoftwareContractV3Error(RuntimeError):
    """Raised when executable source/runtime provenance cannot be bound."""


SOURCE_PATHS = (
    "scripts/run_method_y_image_training_v3.py",
    "scripts/verify_method_y_image_training_v3.py",
    "src/method_y_revival/image_modeling_v3.py",
    "src/method_y_revival/image_survival_v3.py",
    "src/method_y_revival/image_training_runtime_v3.py",
    "src/method_y_revival/image_training_orchestrator_v3.py",
    "src/method_y_revival/image_outer_eval_v3.py",
    "src/method_y_revival/outer_evaluation_control.py",
    "src/method_y_revival/image_reporting_v3.py",
    "src/method_y_revival/image_robustness_v3.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_contract(
    project_root: str | Path, source_paths: Iterable[str] = SOURCE_PATHS
) -> dict[str, Any]:
    root = Path(project_root)
    files: dict[str, str] = {}
    for relative in sorted(set(str(value).replace("\\", "/") for value in source_paths)):
        path = root / relative
        if not path.is_file():
            raise SoftwareContractV3Error(f"TRAINING_SOURCE_MISSING:{relative}")
        files[relative] = _sha256_file(path)
    return {"files": files, "bundle_sha256": sha256_mapping(files)}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _git_value(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def build_software_runtime_contract(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root)
    source = source_bundle_contract(root)
    try:
        commit = _git_value(root, "rev-parse", "HEAD")
        status = _git_value(root, "status", "--porcelain=v1", "--", *SOURCE_PATHS)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SoftwareContractV3Error("GIT_PROVENANCE_UNAVAILABLE") from exc
    runtime_policy = copy.deepcopy(config.get("runtime", {}))
    model_policy = {
        key: config.get("model", {}).get(key)
        for key in (
            "batch_size",
            "gradient_accumulation_steps",
            "num_workers",
            "pin_memory",
            "amp",
            "checkpoint_every_epochs",
        )
    }
    payload = {
        "training_source": source,
        "git_commit": commit,
        "scoped_dirty_digest": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "cuda_runtime": _cuda_runtime_version(),
        "scikit_survival": _package_version("scikit-survival"),
        "runtime_policy": runtime_policy,
        "loader_and_math_policy": model_policy,
    }
    return {**payload, "contract_sha256": sha256_mapping(payload)}


def _cuda_runtime_version() -> str:
    try:
        import torch

        return str(torch.version.cuda or "CPU_ONLY")
    except Exception:
        return "TORCH_IMPORT_FAILED"


def bind_software_runtime_contract(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    bound = copy.deepcopy(config)
    contract = build_software_runtime_contract(project_root, bound)
    bound["training_source_sha256"] = contract["training_source"]["bundle_sha256"]
    bound["software_runtime_contract"] = contract
    return bound
