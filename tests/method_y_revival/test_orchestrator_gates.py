from __future__ import annotations

import json
import os

import pytest


def test_active_writer_lock_blocks_a_second_writer(tmp_path):
    from src.method_y_revival.process_guard import ActiveWriterError, WriterLock

    lock_path = tmp_path / ".writer.lock"
    lock_path.write_text(json.dumps({"pid": os.getpid(), "output_root": str(tmp_path)}), encoding="utf-8")

    with pytest.raises(ActiveWriterError, match="ACTIVE_WRITER"):
        WriterLock(lock_path, output_root=tmp_path).acquire()


def test_stale_writer_lock_is_replaced_and_released(tmp_path):
    from src.method_y_revival.process_guard import WriterLock

    lock_path = tmp_path / ".writer.lock"
    lock_path.write_text(json.dumps({"pid": 99999999, "output_root": str(tmp_path)}), encoding="utf-8")
    lock = WriterLock(lock_path, output_root=tmp_path)
    lock.acquire()

    assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    lock.release()
    assert not lock_path.exists()


def test_orchestrator_initialization_is_resumable_and_config_mismatch_fails(tmp_path):
    from src.method_y_revival.contracts import RunContractMismatch
    from src.method_y_revival.orchestrator import initialize_run

    config = {
        "schema_version": 1,
        "seed": 20260714,
        "validation": {"outer_folds": 5, "inner_folds": 3},
        "paths": {"raw_root": r"H:\\A4 Learn"},
    }
    root = tmp_path / "run"
    first = initialize_run(config=config, output_root=root, repo_snapshot={"git_commit": "abc"})
    repeated = initialize_run(config=config, output_root=root, repo_snapshot={"git_commit": "abc"})

    assert first["run_contract_sha256"] == repeated["run_contract_sha256"]
    assert json.loads((root / "orchestration_state.json").read_text(encoding="utf-8"))["stage"] == "PREFLIGHT_DONE"
    assert json.loads((root / "provenance" / "legacy_artifact_registry.json").read_text(encoding="utf-8"))[0]["claim_status"] == "TECHNICAL_ONLY"
    assert json.loads((root / "external_validation" / "status.json").read_text(encoding="utf-8"))["status"] == "EXTERNAL_PENDING"

    changed = {**config, "seed": 7}
    with pytest.raises(RunContractMismatch, match="RUN_CONTRACT_MISMATCH"):
        initialize_run(config=changed, output_root=root, repo_snapshot={"git_commit": "abc"})


def test_state_machine_rejects_skipped_required_stages():
    from src.method_y_revival.orchestrator import OrchestrationError, next_state

    assert next_state("INIT", "PREFLIGHT_DONE") == "PREFLIGHT_DONE"
    with pytest.raises(OrchestrationError, match="INVALID_STAGE_TRANSITION"):
        next_state("INIT", "QC_PASSED")


def test_incremental_value_gate_requires_all_fold_science_conditions():
    from src.method_y_revival.science_gates import incremental_value_gate

    passed = incremental_value_gate(
        delta_ibs_ci=(-0.03, -0.01),
        delta_cindex_ci=(0.01, 0.04),
        outer_folds_completed=5,
        expected_outer_folds=5,
        scanner_sensitivity_passed=True,
    )
    failed = incremental_value_gate(
        delta_ibs_ci=(-0.03, 0.01),
        delta_cindex_ci=(0.01, 0.04),
        outer_folds_completed=5,
        expected_outer_folds=5,
        scanner_sensitivity_passed=True,
    )

    assert passed["science_gate_status"] == "PASS"
    assert passed["claim_status"] == "INTERNAL_ONLY"
    assert failed["science_gate_status"] == "FAIL"
    assert failed["claim_status"] == "EXPLORATORY"


def test_external_policy_forbids_refit_and_only_promotes_a_passed_internal_model():
    from src.method_y_revival.external_validation import ExternalValidationError, external_claim_status

    with pytest.raises(ExternalValidationError, match="EXTERNAL_REFIT_FORBIDDEN"):
        external_claim_status(internal_gate_passed=True, external_gate_passed=True, refit_used=True)
    assert external_claim_status(
        internal_gate_passed=True, external_gate_passed=True, refit_used=False
    ) == "EXTERNALLY_VALIDATED"
    assert external_claim_status(
        internal_gate_passed=False, external_gate_passed=True, refit_used=False
    ) == "EXPLORATORY"


def test_public_bundle_scanner_detects_private_paths_and_checkpoints(tmp_path):
    from src.method_y_revival.privacy import scan_public_bundle

    (tmp_path / "report.md").write_text(r"source H:\\A4 Learn and model.ckpt", encoding="utf-8")
    findings = scan_public_bundle(tmp_path)

    assert {finding["rule"] for finding in findings} >= {"RAW_DRIVE_PATH", "PRIVATE_CHECKPOINT_REFERENCE"}
