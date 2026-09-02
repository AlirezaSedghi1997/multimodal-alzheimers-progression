from __future__ import annotations

"""Contract-binding semantics for the v4 factorial launcher.

The v4 run contract must bind to the bytes of the actually-imported source
files, NOT to the live git HEAD. A post-run commit that does not touch any
bound source (e.g. docs-only commit) must never break resume. A real byte
change in any bound source MUST change the digest (fail-closed).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.method_y_revival import software_contract_v4 as sc4  # noqa: E402


def _fake_repo(tmp_path: Path) -> Path:
    # Only a couple of bound sources are needed; the rest are absent and
    # SOURCE_PATHS is patched in each test to just these.
    for rel in ("scripts/run_method_y_image_factorial_v4.py", "src/method_y_revival/atomic_io.py"):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# v4\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def scoped_sources(monkeypatch):
    subset = (
        "scripts/run_method_y_image_factorial_v4.py",
        "src/method_y_revival/atomic_io.py",
    )
    monkeypatch.setattr(sc4, "SOURCE_PATHS", subset)
    return subset


def _patch_head(monkeypatch, commit: str, dirty: str = ""):
    class _Result:
        def __init__(self, out: str):
            self.stdout = out

    def fake_run(args, **kwargs):
        if "rev-parse" in args:
            return _Result(commit)
        if "status" in args:
            return _Result(dirty)
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(sc4.subprocess, "run", fake_run)


def test_head_move_without_source_change_keeps_bundle_digest(tmp_path, monkeypatch, scoped_sources):
    repo = _fake_repo(tmp_path)
    d1 = d2 = None
    c1 = c2 = None
    for commit in ("a" * 40, "b" * 40):
        _patch_head(monkeypatch, commit)
        c = sc4.build_software_runtime_contract_v4(repo, {})
        if d1 is None:
            d1 = c["training_source"]["bundle_sha256"]
            c1 = c["contract_sha256"]
        else:
            d2 = c["training_source"]["bundle_sha256"]
            c2 = c["contract_sha256"]
    assert d1 == d2, "HEAD move with identical bound sources must not change bundle digest"
    assert c1 == c2, "HEAD move must not change contract_sha256 (it is the resume-binding digest)"


def test_dirty_status_change_keeps_contract_digest(tmp_path, monkeypatch, scoped_sources):
    repo = _fake_repo(tmp_path)
    _patch_head(monkeypatch, "a" * 40, dirty="")
    c1 = sc4.build_software_runtime_contract_v4(repo, {})["contract_sha256"]
    _patch_head(monkeypatch, "a" * 40, dirty=" M scripts/run_method_y_image_factorial_v4.py")
    c2 = sc4.build_software_runtime_contract_v4(repo, {})["contract_sha256"]
    assert c1 == c2, "dirty-status digest is provenance metadata, never a resume-binding input"


def test_head_and_dirty_recorded_as_provenance(tmp_path, monkeypatch, scoped_sources):
    repo = _fake_repo(tmp_path)
    _patch_head(monkeypatch, "a" * 40, dirty=" M scripts/x.py")
    c = sc4.build_software_runtime_contract_v4(repo, {})
    assert c.get("git_commit") == "a" * 40
    assert c.get("git_provenance_status") == " M scripts/x.py" or c.get("scoped_dirty_digest")


def test_real_byte_change_changes_digest(tmp_path, monkeypatch, scoped_sources):
    repo = _fake_repo(tmp_path)
    _patch_head(monkeypatch, "a" * 40)
    d1 = sc4.build_software_runtime_contract_v4(repo, {})["training_source"]["bundle_sha256"]
    (repo / "scripts" / "run_method_y_image_factorial_v4.py").write_text("# v4 changed\n", encoding="utf-8")
    d2 = sc4.build_software_runtime_contract_v4(repo, {})["training_source"]["bundle_sha256"]
    assert d1 != d2


def test_live_v4_bundle_matches_stored_run_contract():
    """Live bundle digest must equal the digest recorded in the stored v4
    run_contract.json, because no bound source changed since (only HEAD)."""
    root = ROOT / "results" / "method_y_revival_v4_image_factorial_v1"
    stored_path = root / "run_contract.json"
    if not stored_path.is_file():
        pytest.skip("v4 output root absent")
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    stored_bundle = stored["software_runtime_contract"]["training_source"]["bundle_sha256"]
    live = sc4.build_software_runtime_contract_v4(ROOT, {})
    assert live["training_source"]["bundle_sha256"] == stored_bundle
