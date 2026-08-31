from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.research.reconcile_brainiac_plan017_legacy_smoke_v1 import (
    LEGACY_SMOKE_PAYLOAD,
    reconcile_legacy_smoke,
)
from scripts.research import prepare_brainiac_raw_t1_v2 as runner


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_exact_legacy_smoke_is_archived_append_only_and_idempotent(
    tmp_path: Path,
) -> None:
    smoke = tmp_path / "registry" / "brainiac_preprocessing_v2_smoke.json"
    archive_root = tmp_path / "registry" / "superseded" / "legacy_smoke_reports"
    report = tmp_path / "registry" / "BRAINIAC_PLAN017_LEGACY_SMOKE_RECONCILIATION_V1.json"
    _write_json(smoke, dict(LEGACY_SMOKE_PAYLOAD))

    expected_current = {
        "schema_version": 3,
        "artifact": "BRAINIAC_REAL_12_SAMPLE_SMOKE_CROSS_BIND",
        "status": "PASS",
        "rows": 12,
        "source_role": "EQUIVALENCE_SEQUENTIAL_LEG",
        "root_relative_path": "data/derived/example",
        "root_tree_sha256": "1" * 64,
        "selection_sha256": "2" * 64,
        "contract_sha256": "3" * 64,
        "execution_profile_integrity_sha256": "4" * 64,
        "physical_qc_validated_from_live_receipts": True,
        "label_blind": True,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    first = reconcile_legacy_smoke(
        smoke_path=smoke,
        archive_root=archive_root,
        report_path=report,
        expected_current_scientific=expected_current,
        evidence={"primary_valid_n": 0, "calibration_valid_n": 32},
    )

    assert first["status"] == "PASS"
    assert first["action"] == "ARCHIVED_EXACT_LEGACY_SCHEMA2_SMOKE"
    assert first["preprocessing_reexecuted"] is False
    assert first["primary_valid_n"] == 0
    assert not smoke.exists()
    archived = archive_root / str(first["archive_leaf"]) / smoke.name
    assert archived.is_file()
    assert json.loads(archived.read_text(encoding="utf-8")) == LEGACY_SMOKE_PAYLOAD
    assert (archived.parent / "reason.json").is_file()

    second = reconcile_legacy_smoke(
        smoke_path=smoke,
        archive_root=archive_root,
        report_path=report,
        expected_current_scientific=expected_current,
        evidence={"primary_valid_n": 0, "calibration_valid_n": 32},
    )
    assert second == first


def test_unknown_or_mutated_smoke_never_uses_legacy_exception(tmp_path: Path) -> None:
    smoke = tmp_path / "brainiac_preprocessing_v2_smoke.json"
    mutated = dict(LEGACY_SMOKE_PAYLOAD)
    mutated["rows"] = 11
    _write_json(smoke, mutated)

    with pytest.raises(RuntimeError, match="LEGACY_SMOKE_NOT_EXACT_ALLOWLIST"):
        reconcile_legacy_smoke(
            smoke_path=smoke,
            archive_root=tmp_path / "archive",
            report_path=tmp_path / "report.json",
            expected_current_scientific={"schema_version": 3},
            evidence={"primary_valid_n": 0, "calibration_valid_n": 32},
        )


def test_reconciliation_refuses_nonzero_primary_or_incomplete_calibration(
    tmp_path: Path,
) -> None:
    smoke = tmp_path / "brainiac_preprocessing_v2_smoke.json"
    _write_json(smoke, dict(LEGACY_SMOKE_PAYLOAD))

    with pytest.raises(RuntimeError, match="LEGACY_SMOKE_SAFETY_GATE_BLOCKED"):
        reconcile_legacy_smoke(
            smoke_path=smoke,
            archive_root=tmp_path / "archive",
            report_path=tmp_path / "report.json",
            expected_current_scientific={"schema_version": 3},
            evidence={"primary_valid_n": 1, "calibration_valid_n": 32},
        )


def test_equivalence_binding_stays_precalibration_after_threshold_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_threshold = "a" * 64
    base = {
        "policy_sha256": "1" * 64,
        "qc_threshold_manifest_sha256": actual_threshold,
    }
    monkeypatch.setattr(runner, "_base_bindings", lambda: dict(base))

    bindings = runner._equivalence_receipt_bindings()

    assert bindings["qc_threshold_manifest_sha256"] == hashlib.sha256(
        b"NOT_YET_CALIBRATED_PLAN_003"
    ).hexdigest()
    assert bindings["qc_threshold_manifest_sha256"] != actual_threshold
    assert base["qc_threshold_manifest_sha256"] == actual_threshold


def test_parallel_unlock_validates_profile_with_precalibration_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path = tmp_path / "profile.json"
    _write_json(profile_path, {"execution_mode": "PARALLEL_SAFE"})
    monkeypatch.setattr(runner, "EXECUTION_PROFILE_PATH", profile_path)
    monkeypatch.setattr(
        runner, "_current_equivalence_contract_sha256", lambda: "c" * 64
    )
    expected = {"qc_threshold_manifest_sha256": "p" * 64}
    monkeypatch.setattr(
        runner, "_equivalence_receipt_bindings", lambda: dict(expected)
    )
    observed: dict[str, object] = {}

    def _validate(profile, *, expected_contract_sha256, expected_receipt_bindings):
        observed["profile"] = profile
        observed["contract"] = expected_contract_sha256
        observed["bindings"] = expected_receipt_bindings
        return {"execution_mode": "PARALLEL_SAFE"}

    monkeypatch.setattr(runner, "validate_execution_profile", _validate)

    assert runner._parallel_unlock_state() == (False, True)
    assert observed["contract"] == "c" * 64
    assert observed["bindings"] == expected
