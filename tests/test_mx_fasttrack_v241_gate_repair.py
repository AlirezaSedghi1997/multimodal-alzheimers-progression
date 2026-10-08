"""Adversarial tests for the additive, in-memory v2.4.1 verifier repair."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.mx_fasttrack_v2 import integrity
from scripts.mx_fasttrack_v2.run import sha
from scripts.mx_fasttrack_v2_4_1 import compatible_lock as repair
from scripts.mx_fasttrack_v2_4_1.supervise_v241 import _no_outer_evidence
from scripts.mx_fasttrack_v2_4_1.supervise_v241 import Supervisor as V241Supervisor
from scripts.mx_fasttrack_v2_4.supervise_v24 import Supervisor as V24Supervisor


def _fixture(tmp_path: Path, monkeypatch):
    out = tmp_path / "run"
    directory = out / "inner_tasks" / "tiny"
    directory.mkdir(parents=True)
    metric = directory / "three_d_metrics.json"
    contract = directory / "run_contract.json"
    metric.write_text(json.dumps({"aggregate": {"status": "COMPLETED"}}), encoding="utf-8")
    contract.write_text(json.dumps({"run_contract_digest": "a" * 64,
                                    "run_contract": {"digest": "a" * 64}}), encoding="utf-8")
    lock_path = out / "reports" / "fasttrack_inner_lock.json"
    lock_path.parent.mkdir()
    lock = {"tasks": {"tiny": {"run_contract_digest": "a" * 64,
                                "metrics_sha256": sha(metric),
                                "run_contract_sha256": sha(contract)}}}
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(repair, "TASKS", (("tiny", "hybrid", "baseline"),))
    return out, metric, contract, lock


def test_only_missing_aggregate_digest_is_projected_in_memory(tmp_path, monkeypatch):
    out, metric, _, lock = _fixture(tmp_path, monkeypatch)
    original = metric.read_bytes()

    def original_verifier(root, output, *, hash_cache=True):
        payload = integrity.read(metric)
        assert payload["aggregate"]["run_contract_digest"] == "a" * 64
        assert metric.read_bytes() == original
        return lock

    monkeypatch.setattr(repair, "ORIGINAL_CHECK_LOCK", original_verifier)
    prior = integrity.read
    assert repair.check_lock_compatible(tmp_path, out, hash_cache=False) == lock
    assert integrity.read is prior
    assert metric.read_bytes() == original


def test_contradictory_recorded_digest_is_not_replaced(tmp_path, monkeypatch):
    out, metric, _, lock = _fixture(tmp_path, monkeypatch)
    metric.write_text(json.dumps({"aggregate": {"run_contract_digest": "b" * 64}}),
                      encoding="utf-8")
    lock["tasks"]["tiny"]["metrics_sha256"] = sha(metric)
    (out / "reports/fasttrack_inner_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(RuntimeError, match="RECORDED_AGGREGATE_DIGEST_CONTRADICTION"):
        repair.check_lock_compatible(tmp_path, out, hash_cache=False)


def test_metric_byte_drift_blocks_before_original_verifier(tmp_path, monkeypatch):
    out, metric, _, _ = _fixture(tmp_path, monkeypatch)
    metric.write_text(json.dumps({"aggregate": {"changed": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="HASHED_METRIC_OR_CONTRACT_INVALID"):
        repair.check_lock_compatible(tmp_path, out, hash_cache=False)


def test_original_verifier_failure_propagates_and_patch_restores(tmp_path, monkeypatch):
    out, _, _, _ = _fixture(tmp_path, monkeypatch)
    prior = integrity.read
    monkeypatch.setattr(repair, "ORIGINAL_CHECK_LOCK",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ORIGINAL_BLOCKER")))
    with pytest.raises(RuntimeError, match="ORIGINAL_BLOCKER"):
        repair.check_lock_compatible(tmp_path, out, hash_cache=False)
    assert integrity.read is prior


def test_outer_receipt_must_be_absent_before_amendment(tmp_path):
    out = tmp_path / "run"
    reports = out / "reports"
    reports.mkdir(parents=True)
    _no_outer_evidence(out)
    (reports / "evaluation_outer_hybrid_pair_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="OUTER_EVIDENCE_ALREADY_PRESENT"):
        _no_outer_evidence(out)


def test_versioned_supervisor_keeps_separate_journal_and_routes_outer(tmp_path, monkeypatch):
    captured = []

    def fake_init(self, root, output_relative, package_root):
        self.root = root
        self.output_relative = output_relative
        self.out = root / output_relative
        self.reports = self.out / "reports"
        self.reports.mkdir(parents=True)

    def fake_run_step(self, name, command, *, allowed=None):
        captured.append((name, command, allowed))
        return 0

    monkeypatch.setattr(V24Supervisor, "__init__", fake_init)
    monkeypatch.setattr(V24Supervisor, "run_step", fake_run_step)
    supervisor = V241Supervisor(tmp_path, Path("results/versioned"), tmp_path)
    assert supervisor.state_path.name == "fasttrack_v241_supervisor_state.json"
    supervisor.run_step("mx5_outer_hybrid_pair", ["old-command"])
    assert captured[0][0] == "mx5_outer_hybrid_pair"
    assert captured[0][1][2] == "hybrid-pair"
    supervisor.run_step("mx6_v22_xai_crossfit", ["old-command"], allowed={0, 3})
    assert captured[1][1][2] == "xai-run"
    assert captured[1][2] == {0, 3}
