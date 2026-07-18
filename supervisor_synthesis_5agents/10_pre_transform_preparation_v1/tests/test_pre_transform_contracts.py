import json
from pathlib import Path

import pytest

from preflight import (
    build_endpoint_contract,
    build_model_contract,
    compare_transform_status,
    evaluate_contracts,
)
from post_transform_gate import evaluate_post_transform_gate


def test_endpoint_contract_rejects_same_pet_derived_target():
    contract = build_endpoint_contract()
    assert contract["target_derivation"] == "future_clinical_only"
    assert contract["forbidden_targets"]
    assert "same_pet_derived_label" in contract["forbidden_targets"]


def test_tracer_branches_are_disjoint():
    contract = build_endpoint_contract()
    assert contract["branches"]["amyloid"]["tracer"] == "FBP"
    assert contract["branches"]["tau"]["tracer"] == "FTP"
    assert contract["branches"]["amyloid"]["tracer"] != contract["branches"]["tau"]["tracer"]


def test_model_contract_keeps_outer_test_unavailable_during_hpo():
    contract = build_model_contract()
    assert contract["outer_folds"] == 5
    assert contract["inner_folds"] == 3
    assert contract["outer_test_access"] == "after_inner_selection_and_lock_only"
    assert contract["selection_metric"] == "integrated_brier_score"


def test_contract_evaluation_fails_closed_for_mutated_rules():
    endpoint = build_endpoint_contract()
    endpoint["branches"]["amyloid"]["tracer"] = "FTP"
    result = evaluate_contracts(endpoint, build_model_contract())
    assert result["status"] == "FAIL"
    assert any("tracer" in error for error in result["errors"])


def test_transform_status_mismatch_is_flagged_as_stale_not_silently_accepted():
    result = compare_transform_status(
        {"row_count": 1043, "status_counts": {"PASS": 1041, "FAIL": 2}},
        {"transform": {"source_eligible_count": 1317, "validated_count": 169}, "status": "RUNNING"},
        active_writer=True,
    )
    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert "STALE_VALIDATED_COUNT" in result["flags"]


def test_post_transform_gate_is_fail_closed_while_writer_is_active():
    result = evaluate_post_transform_gate(
        {"active_writer": True, "row_count": 1043, "status_counts": {"PASS": 1041, "FAIL": 2}},
        {"status": "RUNNING", "transform": {"source_eligible_count": 1317, "validated_count": 169}},
    )
    assert result["status"] == "BLOCKED_ACTIVE_WRITER"
    assert result["training_allowed"] is False
