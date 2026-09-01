from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any

from .contracts import sha256_mapping


class SoftwareContractV4Error(RuntimeError):
    pass


SOURCE_PATHS = (
    "scripts/build_method_y_pet_augmentation_qc_v4.py",
    "scripts/run_method_y_image_factorial_v4.py",
    "scripts/verify_method_y_image_factorial_v4.py",
    "src/method_y_revival/atomic_io.py",
    "src/method_y_revival/contracts.py",
    "src/method_y_revival/image_factorial_v4.py",
    "src/method_y_revival/image_factorial_orchestrator_v4.py",
    "src/method_y_revival/image_factorial_reporting_v4.py",
    "src/method_y_revival/image_modeling_v3.py",
    "src/method_y_revival/image_outer_eval_v3.py",
    "src/method_y_revival/image_outer_eval_v4.py",
    "src/method_y_revival/image_reporting_v3.py",
    "src/method_y_revival/image_robustness_v3.py",
    "src/method_y_revival/image_survival_v3.py",
    "src/method_y_revival/image_training_orchestrator_v3.py",
    "src/method_y_revival/image_training_runtime_v3.py",
    "src/method_y_revival/image_training_runtime_v4.py",
    "src/method_y_revival/oof_reporting.py",
    "src/method_y_revival/outer_evaluation_control.py",
    "src/method_y_revival/pet_augmentation_qc_v4.py",
    "src/method_y_revival/pet_augmentation_v4.py",
    "src/method_y_revival/process_guard.py",
    "src/method_y_revival/science_gates.py",
    "src/method_y_revival/survival_models.py",
    "src/method_y_revival/software_contract_v4.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def build_software_runtime_contract_v4(
    project_root: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path(project_root)
    files: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise SoftwareContractV4Error(f"TRAINING_SOURCE_MISSING:{relative}")
        files[relative] = _sha256_file(path)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        scoped_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--", *SOURCE_PATHS],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SoftwareContractV4Error("GIT_PROVENANCE_UNAVAILABLE") from exc
    payload = {
        "training_source": {
            "files": files,
            "bundle_sha256": sha256_mapping(files),
        },
        "git_commit": commit,
        "scoped_dirty_digest": hashlib.sha256(scoped_status.encode("utf-8")).hexdigest(),
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "cuda_runtime": _cuda_runtime_version(),
        "cuda_hardware": _cuda_hardware_contract(),
        "scikit_survival": _package_version("scikit-survival"),
        "monai": _package_version("monai"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "h5py": _package_version("h5py"),
        "nibabel": _package_version("nibabel"),
        "scipy": _package_version("scipy"),
        "runtime_policy": copy.deepcopy(config.get("runtime", {})),
        "model_policy": copy.deepcopy(config.get("model", {})),
        "factorial_arms": copy.deepcopy(config.get("factorial_arms", {})),
        "augmentation_policy": copy.deepcopy(config.get("augmentation", {})),
    }
    return {**payload, "contract_sha256": sha256_mapping(payload)}


def _cuda_runtime_version() -> str:
    try:
        import torch

        return str(torch.version.cuda or "CPU_ONLY")
    except Exception:
        return "TORCH_IMPORT_FAILED"


def _cuda_hardware_contract() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        properties = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "device_name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "cudnn_version": int(torch.backends.cudnn.version() or 0),
        }
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}


def bind_software_runtime_contract_v4(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    bound = copy.deepcopy(config)
    contract = build_software_runtime_contract_v4(project_root, bound)
    bound["training_source_sha256"] = contract["training_source"]["bundle_sha256"]
    bound["software_runtime_contract"] = contract
    return bound
