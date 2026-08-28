from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess

import pytest

from src.method_y_revival.contracts import sha256_mapping


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _contract(*, commit: str, raw_source: bytes) -> dict:
    files = {"bound.py": hashlib.sha256(raw_source).hexdigest()}
    payload = {
        "git_commit": commit,
        "scoped_dirty_digest": hashlib.sha256(b"").hexdigest(),
        "python": "3.11.15",
        "torch": "2.5.1+cu121",
        "cuda_runtime": "12.1",
        "scikit_survival": "0.27.0",
        "runtime_policy": {"deterministic_cuda": True},
        "loader_and_math_policy": {"batch_size": 4},
        "training_source": {
            "files": files,
            "bundle_sha256": sha256_mapping(files),
        },
    }
    return {**payload, "contract_sha256": sha256_mapping(payload)}


def _repo_with_provenance_only_commit(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Method Y Test")
    (root / "bound.py").write_bytes(b"VALUE = 1\n")
    _git(root, "add", "bound.py")
    _git(root, "commit", "-m", "locked training source")
    saved_commit = _git(root, "rev-parse", "HEAD")
    (root / "unrelated.md").write_text("provenance only\n", encoding="utf-8")
    _git(root, "add", "unrelated.md")
    _git(root, "commit", "-m", "unrelated provenance")
    return root, saved_commit, _git(root, "rev-parse", "HEAD")


def test_clean_git_canonical_source_allows_raw_eol_hash_drift(tmp_path: Path) -> None:
    """Break caught: a raw CRLF/LF rewrite must not invalidate identical code."""
    from scripts.verify_method_y_image_training_v3_compat import (
        verify_canonical_source_compatibility,
    )

    root, saved_commit, live_commit = _repo_with_provenance_only_commit(tmp_path)
    saved = _contract(commit=saved_commit, raw_source=b"VALUE = 1\r\n")
    current = _contract(commit=live_commit, raw_source=b"VALUE = 1\n")
    current["scoped_dirty_digest"] = "f" * 64
    current_body = {key: value for key, value in current.items() if key != "contract_sha256"}
    current["contract_sha256"] = sha256_mapping(current_body)

    result = verify_canonical_source_compatibility(
        project_root=root,
        saved_contract=saved,
        current_contract=current,
        source_paths=("bound.py",),
    )

    assert result["status"] == "PASS"
    assert result["mode"] == "GIT_CANONICAL_SOURCE_EQUIVALENCE"
    assert result["saved_commit"] == saved_commit
    assert result["live_commit"] == live_commit


def test_semantic_source_drift_is_rejected_even_when_runtime_matches(tmp_path: Path) -> None:
    """Break caught: compatibility must never hide an executable code change."""
    from scripts.verify_method_y_image_training_v3_compat import (
        CanonicalContractCompatibilityError,
        verify_canonical_source_compatibility,
    )

    root, saved_commit, live_commit = _repo_with_provenance_only_commit(tmp_path)
    saved = _contract(commit=saved_commit, raw_source=b"VALUE = 1\r\n")
    (root / "bound.py").write_bytes(b"VALUE = 2\n")
    current = _contract(commit=live_commit, raw_source=b"VALUE = 2\n")

    with pytest.raises(CanonicalContractCompatibilityError, match="CANONICAL_SOURCE_DRIFT"):
        verify_canonical_source_compatibility(
            project_root=root,
            saved_contract=saved,
            current_contract=current,
            source_paths=("bound.py",),
        )


def test_runtime_policy_drift_is_rejected_before_contract_override(tmp_path: Path) -> None:
    """Break caught: EOL compatibility must not relax runtime or math policy."""
    from scripts.verify_method_y_image_training_v3_compat import (
        CanonicalContractCompatibilityError,
        verify_canonical_source_compatibility,
    )

    root, saved_commit, live_commit = _repo_with_provenance_only_commit(tmp_path)
    saved = _contract(commit=saved_commit, raw_source=b"VALUE = 1\r\n")
    current = copy.deepcopy(_contract(commit=live_commit, raw_source=b"VALUE = 1\n"))
    current["runtime_policy"]["deterministic_cuda"] = False
    current_body = {key: value for key, value in current.items() if key != "contract_sha256"}
    current["contract_sha256"] = sha256_mapping(current_body)

    with pytest.raises(CanonicalContractCompatibilityError, match="RUNTIME_OR_MATH_POLICY_DRIFT"):
        verify_canonical_source_compatibility(
            project_root=root,
            saved_contract=saved,
            current_contract=current,
            source_paths=("bound.py",),
        )
