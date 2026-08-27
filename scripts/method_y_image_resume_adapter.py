#!/usr/bin/env python3
"""Prepare a provenance-safe resume environment for Method Y image training.

The training contract deliberately binds the Git commit.  A later commit that
does not alter any contract-bound source byte would otherwise make a partially
completed run impossible to resume.  This adapter permits exactly that one
case by presenting Git metadata pinned to the saved commit while continuing to
execute the byte-identical files in the live worktree.  Every other contract
difference remains fail-closed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.software_contract_v3 import bind_software_runtime_contract


class ResumeContractAdapterError(RuntimeError):
    """Raised when a saved run cannot be resumed without contract drift."""


def _run_git(
    cwd: Path,
    *arguments: str,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diff_paths(saved: Any, current: Any, prefix: str = "") -> list[str]:
    if isinstance(saved, dict) and isinstance(current, dict):
        differences: list[str] = []
        for key in sorted(set(saved) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved or key not in current:
                differences.append(path)
            else:
                differences.extend(_diff_paths(saved[key], current[key], path))
        return differences
    return [] if saved == current else [prefix]


def classify_contract_compatibility(
    saved: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Classify exact, commit-only, and unsafe software-contract drift."""
    differences = _diff_paths(saved, current)
    if not differences:
        return {"mode": "DIRECT", "diff_paths": []}

    saved_semantic = copy.deepcopy(saved)
    current_semantic = copy.deepcopy(current)
    for payload in (saved_semantic, current_semantic):
        payload.pop("contract_sha256", None)
        payload.pop("git_commit", None)

    allowed = {"contract_sha256", "git_commit"}
    if set(differences) <= allowed and saved_semantic == current_semantic:
        return {
            "mode": "PIN_SAVED_COMMIT",
            "diff_paths": differences,
        }
    return {"mode": "BLOCKED", "diff_paths": differences}


def prepare_pinned_git_environment(
    *, project_root: str | Path, saved_commit: str, runtime_root: str | Path
) -> dict[str, Any]:
    """Create isolated Git metadata at ``saved_commit`` without editing files."""
    root = Path(project_root).resolve()
    runtime = Path(runtime_root).resolve()
    if root not in runtime.parents:
        raise ResumeContractAdapterError("PIN_RUNTIME_MUST_BE_INSIDE_PROJECT_ROOT")
    if len(saved_commit) != 40 or any(ch not in "0123456789abcdef" for ch in saved_commit.lower()):
        raise ResumeContractAdapterError("SAVED_GIT_COMMIT_INVALID")
    try:
        _run_git(root, "cat-file", "-e", f"{saved_commit}^{{commit}}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ResumeContractAdapterError("SAVED_GIT_COMMIT_UNAVAILABLE") from exc

    runtime.mkdir(parents=True, exist_ok=True)
    snapshot = runtime / f"git_{saved_commit[:12]}"
    if snapshot.exists():
        try:
            observed = _run_git(snapshot, "rev-parse", "HEAD")
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ResumeContractAdapterError("EXISTING_PINNED_GIT_SNAPSHOT_INVALID") from exc
        if observed != saved_commit:
            raise ResumeContractAdapterError("EXISTING_PINNED_GIT_SNAPSHOT_COMMIT_MISMATCH")
    else:
        temporary = Path(tempfile.mkdtemp(prefix="git_pin_tmp_", dir=runtime))
        # git clone requires a non-existing destination.
        shutil.rmtree(temporary)
        try:
            _run_git(
                root,
                "clone",
                "--no-checkout",
                "--shared",
                "--local",
                str(root),
                str(temporary),
            )
            _run_git(temporary, "reset", "--mixed", saved_commit)
            if _run_git(temporary, "rev-parse", "HEAD") != saved_commit:
                raise ResumeContractAdapterError("PINNED_GIT_SNAPSHOT_VERIFY_FAILED")
            os.replace(temporary, snapshot)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    git_dir = snapshot / ".git"
    if not git_dir.is_dir():
        raise ResumeContractAdapterError("PINNED_GIT_DIR_MISSING")
    environment = {
        "GIT_DIR": str(git_dir),
        "GIT_WORK_TREE": str(root),
        "GIT_OPTIONAL_LOCKS": "0",
    }
    verify_env = os.environ.copy()
    verify_env.update(environment)
    if _run_git(root, "rev-parse", "HEAD", env=verify_env) != saved_commit:
        raise ResumeContractAdapterError("PINNED_GIT_ENVIRONMENT_VERIFY_FAILED")
    return {
        "snapshot_root": str(snapshot),
        "environment": environment,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_resume_environment(
    *, project_root: Path, config_path: Path
) -> dict[str, Any]:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime_config = raw_config.get("runtime", {})
    if runtime_config.get("deterministic_cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["IMAGE_V3_PERF"] = "0"
    current_config = bind_software_runtime_contract(raw_config, project_root)
    output_value = Path(current_config["output_root"])
    output_root = output_value if output_value.is_absolute() else project_root / output_value
    run_contract_path = output_root / "run_contract.json"
    if not run_contract_path.is_file():
        raise ResumeContractAdapterError("RUN_CONTRACT_MISSING")
    run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
    payload = run_contract.get("payload")
    if not isinstance(payload, dict) or run_contract.get("digest") != sha256_mapping(payload):
        raise ResumeContractAdapterError("RUN_CONTRACT_SELF_HASH_MISMATCH")
    saved_contract = payload.get("software_runtime_contract")
    current_contract = current_config.get("software_runtime_contract")
    if not isinstance(saved_contract, dict) or not isinstance(current_contract, dict):
        raise ResumeContractAdapterError("SOFTWARE_RUNTIME_CONTRACT_MISSING")

    classification = classify_contract_compatibility(saved_contract, current_contract)
    if classification["mode"] == "BLOCKED":
        raise ResumeContractAdapterError(
            "SOFTWARE_RUNTIME_CONTRACT_MISMATCH:"
            + ",".join(classification["diff_paths"])
        )

    environment: dict[str, str] = {}
    snapshot_root: str | None = None
    if classification["mode"] == "PIN_SAVED_COMMIT":
        prepared = prepare_pinned_git_environment(
            project_root=project_root,
            saved_commit=str(saved_contract["git_commit"]),
            runtime_root=project_root / ".runtime" / "method_y_image_v3" / "contract_git",
        )
        environment = prepared["environment"]
        snapshot_root = prepared["snapshot_root"]
        verify_env = os.environ.copy()
        verify_env.update(environment)
        old_values = {key: os.environ.get(key) for key in environment}
        try:
            os.environ.update(environment)
            rebound = bind_software_runtime_contract(raw_config, project_root)[
                "software_runtime_contract"
            ]
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if rebound != saved_contract:
            diff = _diff_paths(saved_contract, rebound)
            raise ResumeContractAdapterError(
                "PINNED_SOFTWARE_CONTRACT_VERIFY_FAILED:" + ",".join(diff)
            )

    receipt = {
        "schema_version": "1.0",
        "status": "RESUME_ENVIRONMENT_READY",
        "mode": classification["mode"],
        "diff_paths": classification["diff_paths"],
        "saved_git_commit": saved_contract.get("git_commit"),
        "live_git_commit": current_contract.get("git_commit"),
        "saved_software_contract_sha256": saved_contract.get("contract_sha256"),
        "live_software_contract_sha256": current_contract.get("contract_sha256"),
        "training_source_bundle_sha256": saved_contract.get("training_source", {}).get(
            "bundle_sha256"
        ),
        "run_contract_sha256": run_contract.get("digest"),
        "adapter_sha256": _sha256_file(Path(__file__)),
        "snapshot_root": snapshot_root,
        "scientific_contract_changed": False,
    }
    receipt_path = output_root / "provenance" / "resume_contract_adapter.json"
    _atomic_write_json(receipt_path, receipt)
    return {
        "status": "READY",
        "mode": classification["mode"],
        "environment": environment,
        "receipt": str(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Method Y image-training contract-compatible resume"
    )
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "method_y_image_training_v3_2.yaml")
    )
    args = parser.parse_args()
    try:
        result = prepare_resume_environment(
            project_root=Path(args.project_root).resolve(),
            config_path=Path(args.config).resolve(),
        )
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
