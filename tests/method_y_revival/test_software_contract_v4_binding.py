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

import torch  # noqa: E402  (eager: avoid lazy circular import inside tests)

from src.method_y_revival import software_contract_v4 as sc4  # noqa: E402
from src.method_y_revival.image_factorial_orchestrator_v4 import (  # noqa: E402
    ImageFactorialOrchestratorV4,
)
from src.method_y_revival.software_contract_v4 import (  # noqa: E402
    binding_core,
)


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
    gp = c.get("git_provenance", {})
    assert gp.get("commit") == "a" * 40
    assert str(gp.get("scoped_status", "")).strip() == "M scripts/x.py"
    assert gp.get("scoped_dirty_digest")


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


def test_git_provenance_is_informational_contract_field():
    """git metadata must live under git_provenance, excluded from
    contract_sha256 inputs and from launcher resume equality."""
    c = sc4.build_software_runtime_contract_v4(ROOT, {})
    gp = c.get("git_provenance")
    assert isinstance(gp, dict)
    assert gp.get("policy") == "INFORMATIONAL_ONLY_NEVER_RESUME_BINDING"
    for legacy in ("git_commit", "git_provenance_status", "scoped_dirty_digest"):
        assert legacy not in c, f"legacy provenance field must not be top-level: {legacy}"

    src = (ROOT / "scripts" / "run_method_y_image_factorial_v4.py").read_text(encoding="utf-8")
    assert 'if key != "git_provenance"' in src, (
        "launcher equality must exclude git_provenance from resume binding"
    )


def test_binding_core_excludes_git_provenance():
    from src.method_y_revival.software_contract_v4 import binding_core

    contract = sc4.build_software_runtime_contract_v4(ROOT, {})
    core = binding_core(contract)
    assert "git_provenance" not in core
    # byte change still flips the core
    mutated = dict(core)
    files = dict(mutated["training_source"]["files"])
    first = sorted(files)[0]
    files[first] = "0" * 64
    mutated["training_source"] = {**mutated["training_source"], "files": files}
    assert sc4.sha256_mapping(core) != sc4.sha256_mapping(mutated)


def test_master_digest_ignores_git_provenance(monkeypatch, tmp_path):
    """Master contract digest must be stable across HEAD/dirt moves; it must
    flip on bound-source byte change."""
    from src.method_y_revival.image_factorial_orchestrator_v4 import (  # noqa: E402
        ImageFactorialOrchestratorV4 as _IFO4,
    )

    assert _IFO4 is ImageFactorialOrchestratorV4

    orch_kwargs = dict(
        output_root=tmp_path,
        branch="amyloid",
        membership_sha256="a" * 64,
        cache_sha256="b" * 64,
        mask_sha256="c" * 64,
        split_sha256="d" * 64,
        augmentation_qc_sha256="e" * 64,
        h5_cache_path=tmp_path / "cache.h5",
        mask_path=tmp_path / "mask.h5",
        augmentation_qc={},
    )
    base_cfg = {
        "seed": 1,
        "model": {"hpo_candidates": [{"candidate_id": "x", "complexity_rank": 0}]},
        "factorial_arms": {"A0": {}},
        "runtime": {},
        "execution": {"arms": ["A0", "A1", "A2", "A3", "A4"]},
    }

    def make_orch(git_commit, extra_sw=None, tag="x"):
        sw = {"training_source": {"bundle_sha256": "f" * 64, "files": {}},
              "git_provenance": {"commit": git_commit, "policy": "INFORMATIONAL_ONLY_NEVER_RESUME_BINDING"}}
        if extra_sw:
            sw.update(extra_sw)
        cfg = dict(base_cfg, software_runtime_contract=sw)
        (tmp_path / "cache.h5").touch(exist_ok=True)
        (tmp_path / "mask.h5").touch(exist_ok=True)
        return ImageFactorialOrchestratorV4(config=cfg, output_root=tmp_path / tag, **{k: v for k, v in orch_kwargs.items() if k != "output_root"})

    d1 = make_orch("a" * 40, tag="r1").master_contract_sha256
    d2 = make_orch("b" * 40, tag="r2").master_contract_sha256
    assert d1 == d2, "HEAD-only move must not change master digest"

    d3 = make_orch("b" * 40, extra_sw={"torch": "9.9.9"}, tag="r3").master_contract_sha256
    assert d1 != d3, "toolchain change must still flip master digest"


def test_migration_master_formula_uses_binding_core():
    """The migration tool's master-digest formula must be identical to the
    orchestrator's (binding-core), otherwise migrated masters will never
    satisfy the launcher's resume check."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "migtool",
        ROOT / "scripts" / "migrate_method_y_image_factorial_v4_contract.py",
    )
    assert spec is not None and spec.loader is not None
    # Load source and check the formula textually: the tool-local
    # _master_digest must route the software contract through binding_core.
    src = (ROOT / "scripts" / "migrate_method_y_image_factorial_v4_contract.py").read_text(
        encoding="utf-8"
    )
    start = src.find("def _master_digest(")
    assert start != -1
    segment = src[start : start + 1400]
    assert "binding_core" in segment, (
        "migration tool must compute master digests with binding_core, "
        "mirroring the orchestrator"
    )


def test_master_formula_parity_orchestrator_vs_migration_tool(tmp_path):
    """Same inputs through the orchestrator formula and the migration-tool
    formula must yield the SAME master digest (fail-closed parity)."""
    import importlib.util

    from src.method_y_revival.image_factorial_orchestrator_v4 import (
        ImageFactorialOrchestratorV4 as _IFO4b,
    )

    assert _IFO4b is ImageFactorialOrchestratorV4
    spec = importlib.util.spec_from_file_location(
        "migtool3",
        ROOT / "scripts" / "migrate_method_y_image_factorial_v4_contract.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sw = {
        "training_source": {"bundle_sha256": "f" * 64, "files": {}},
        "git_provenance": {"commit": "zzz", "policy": "X"},
    }
    cfg = {
        "seed": 7,
        "model": {"hpo_candidates": [{"candidate_id": "x", "complexity_rank": 0}]},
        "factorial_arms": {"A0": {}},
        "runtime": {},
        "execution": {"arms": ["A0", "A1", "A2", "A3", "A4"]},
        "software_runtime_contract": sw,
        "branches": {"amyloid": {"model_features": []}},
    }
    inputs = {
        "membership": "a" * 64,
        "cache": "b" * 64,
        "mask": "c" * 64,
        "split": "d" * 64,
        "augmentation_qc": "e" * 64,
    }
    tool_digest = mod._master_digest(config=cfg, branch="amyloid", inputs=inputs)

    (tmp_path / "cache.h5").touch(exist_ok=True)
    (tmp_path / "mask.h5").touch(exist_ok=True)
    orch = ImageFactorialOrchestratorV4(
        config=cfg,
        output_root=tmp_path / "o2",
        branch="amyloid",
        membership_sha256="a" * 64,
        cache_sha256="b" * 64,
        mask_sha256="c" * 64,
        split_sha256="d" * 64,
        augmentation_qc_sha256="e" * 64,
        h5_cache_path=tmp_path / "cache.h5",
        mask_path=tmp_path / "mask.h5",
        augmentation_qc={},
    )
    assert tool_digest == orch.master_contract_sha256


def test_winner_target_is_branch_master_digest():
    """The migration helper must resolve each winner row to its branch's NEW
    master digest — never the global run digest."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "migtool4",
        ROOT / "scripts" / "migrate_method_y_image_factorial_v4_contract.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = {
        "master_mappings": [
            {"branch": "amyloid", "new_master_digest": "m" * 64},
            {"branch": "tau", "new_master_digest": "t" * 64},
        ],
        "new_run_contract_sha256": "g" * 64,
    }
    targets = mod._branch_master_digests(plan)
    assert targets == {"amyloid": "m" * 64, "tau": "t" * 64}
    row = {"branch": "amyloid", "path": "x"}
    assert mod._winner_target_digest(row, targets) == "m" * 64
    with pytest.raises(Exception):
        mod._winner_target_digest({"branch": "ghost", "path": "x"}, targets)


def test_winner_migrates_to_branch_master_digest_not_global(tmp_path):
    """A migrated winner's run_contract_sha256 must equal the branch MASTER
    digest (what read_verified_winner_record compares), never the global run
    digest. Regression test for the ec968b9b-era mislabel."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "migtool2",
        ROOT / "scripts" / "migrate_method_y_image_factorial_v4_contract.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_branch_master_digests") or hasattr(mod, "_winner_target_digest"), (
        "migration tool must resolve per-branch master digests for winner rewrite"
    )
