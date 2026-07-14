from __future__ import annotations

import json

import pytest


def test_completed_artifact_does_not_imply_a_scientific_claim():
    from src.method_y_revival.contracts import ArtifactRecord

    record = ArtifactRecord(
        artifact_id="legacy-amyloid",
        stage="legacy_registration",
        branch="amyloid",
        artifact_status="COMPLETED",
        science_gate_status="FAIL",
        claim_status="TECHNICAL_ONLY",
        terminal_reason="SAME_MODALITY_DERIVED_TARGET_AND_COHORT_CONFOUNDING",
    )

    assert record.to_dict()["claim_status"] == "TECHNICAL_ONLY"


def test_external_claim_requires_a_passed_science_gate():
    from src.method_y_revival.contracts import ArtifactRecord, ContractError

    with pytest.raises(ContractError, match="EXTERNAL_CLAIM_REQUIRES_SCIENCE_GATE_PASS"):
        ArtifactRecord(
            artifact_id="invalid",
            stage="external_validation",
            branch="tau",
            artifact_status="COMPLETED",
            science_gate_status="FAIL",
            claim_status="EXTERNALLY_VALIDATED",
        )


def test_run_contract_digest_is_canonical_and_mismatch_fails_closed():
    from src.method_y_revival.contracts import RunContract, RunContractMismatch

    first = RunContract.from_mapping({"seed": 20260714, "folds": 5, "nested": {"inner": 3}})
    reordered = RunContract.from_mapping({"nested": {"inner": 3}, "folds": 5, "seed": 20260714})
    changed = RunContract.from_mapping({"seed": 7, "folds": 5, "nested": {"inner": 3}})

    assert first.digest == reordered.digest
    with pytest.raises(RunContractMismatch, match="RUN_CONTRACT_MISMATCH"):
        first.require_match(changed)


def test_atomic_json_replaces_the_target_without_leaving_temp_files(tmp_path):
    from src.method_y_revival.atomic_io import atomic_write_json

    target = tmp_path / "state.json"
    atomic_write_json(target, {"stage": "INIT"})
    atomic_write_json(target, {"stage": "PREFLIGHT_DONE"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"stage": "PREFLIGHT_DONE"}
    assert list(tmp_path.glob("*.tmp")) == []
