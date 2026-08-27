from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def test_commit_only_drift_is_pinned_but_source_drift_is_blocked() -> None:
    from scripts.method_y_image_resume_adapter import classify_contract_compatibility

    saved = {
        "contract_sha256": "1" * 64,
        "git_commit": "a" * 40,
        "training_source": {"bundle_sha256": "b" * 64},
        "runtime_policy": {"deterministic_cuda": True},
    }
    commit_only = {
        **saved,
        "contract_sha256": "2" * 64,
        "git_commit": "c" * 40,
    }
    source_changed = {
        **commit_only,
        "training_source": {"bundle_sha256": "d" * 64},
    }

    compatible = classify_contract_compatibility(saved, commit_only)
    assert compatible["mode"] == "PIN_SAVED_COMMIT"
    assert compatible["diff_paths"] == ["contract_sha256", "git_commit"]

    blocked = classify_contract_compatibility(saved, source_changed)
    assert blocked["mode"] == "BLOCKED"
    assert "training_source.bundle_sha256" in blocked["diff_paths"]


def test_pinned_git_environment_reports_saved_commit_without_touching_worktree(
    tmp_path: Path,
) -> None:
    from scripts.method_y_image_resume_adapter import prepare_pinned_git_environment

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Method Y Test")
    bound = repo / "bound.py"
    bound.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "bound.py")
    _git(repo, "commit", "-m", "bound source")
    saved_commit = _git(repo, "rev-parse", "HEAD")

    (repo / "unrelated.md").write_text("new provenance only\n", encoding="utf-8")
    _git(repo, "add", "unrelated.md")
    _git(repo, "commit", "-m", "unrelated change")
    live_commit = _git(repo, "rev-parse", "HEAD")
    assert live_commit != saved_commit

    prepared = prepare_pinned_git_environment(
        project_root=repo,
        saved_commit=saved_commit,
        runtime_root=repo / ".runtime" / "contract_pin",
    )
    # A terminal interruption leaves the reusable Git snapshot on disk.  The
    # next invocation must validate and reuse it instead of treating it as a
    # contract mismatch or trying to rebuild it.
    prepared_after_interruption = prepare_pinned_git_environment(
        project_root=repo,
        saved_commit=saved_commit,
        runtime_root=repo / ".runtime" / "contract_pin",
    )
    env = os.environ.copy()
    env.update(prepared["environment"])

    assert prepared_after_interruption == prepared
    assert _git(repo, "rev-parse", "HEAD", env=env) == saved_commit
    assert _git(repo, "status", "--porcelain=v1", "--", "bound.py", env=env) == ""
    assert bound.read_text(encoding="utf-8") == "VALUE = 1\n"
