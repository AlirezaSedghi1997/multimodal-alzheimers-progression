from __future__ import annotations

import json
from pathlib import Path
import pytest


def test_public_report_contains_no_patient_hash_path_embedding_or_prediction(tmp_path):
    from src.method_y_revival.image_reporting_v3 import render_private_and_public, privacy_scan_reports

    preds = [{"patient_hash": "hash123", "event": 1, "time_days": 300}]
    out = tmp_path / "out"
    res = render_private_and_public(
        private_oof_predictions=preds,
        private_embeddings=[[0.1, 0.2]],
        output_root=out,
        branch="tau",
        variant="scalar_tabular",
    )
    scan = privacy_scan_reports(out / "reports")
    assert scan["status"] == "PASS"
    # public should not contain private tokens
    text = Path(res["public_path"]).read_text(encoding="utf-8")
    assert "hash123" not in text
    assert "patient_hash" not in text


def test_claim_gate_with_quantitative_external_limitations():
    from src.method_y_revival.image_reporting_v3 import claim_consistency_state

    # Simulate positive internal delta but scanner not proven
    state = claim_consistency_state(
        training_execution_allowed=True,
        incremental_value_pass=True,
        quantitative_equivalence_proven=False,
        external_validation_pass=False,
        scanner_robustness_passed=False,
    )
    # Internal incremental should be False because scanner not passed
    assert state["internal_incremental_pass"] is False
    assert state["quantitative_standard_claim_allowed"] is False

    # Now scanner passed but quant still false
    state2 = claim_consistency_state(
        training_execution_allowed=True,
        incremental_value_pass=True,
        quantitative_equivalence_proven=False,
        external_validation_pass=False,
        scanner_robustness_passed=True,
    )
    assert state2["internal_incremental_pass"] is True
    assert state2["quantitative_standard_claim_allowed"] is False
    assert state2["external_claim_allowed"] is False


def test_scanner_support_is_not_robustness_proof_in_readiness(tmp_path):
    # Read actual readiness to ensure scanner_support vs robustness decoupling
    import json
    from pathlib import Path

    p = Path("results/method_y_revival_v3_image_v1/readiness/amyloid_image_readiness.json")
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["scanner_support_available_for_prespecified_analysis"] is True
        assert data["scanner_robustness_proven"] is False
        # The limitation must survive
        assert "SCANNER_ROBUSTNESS_NOT_PROVEN" in data["policy"]["claim_limitations"]

    p2 = Path("results/method_y_revival_v3_image_v1/readiness/tau_image_readiness.json")
    if p2.exists():
        data2 = json.loads(p2.read_text(encoding="utf-8"))
        assert data2["scanner_robustness_proven"] is False


def test_training_execution_allowed_vs_positive_claim_allowed():
    from src.method_y_revival.image_training_policy import assess_image_training_policy

    integrity = {
        "branch": "amyloid",
        "tracer": "FBP",
        "endpoint_locked": True,
        "endpoint_is_future": True,
        "split_locked": True,
        "outer_fold_count": 5,
        "subject_overlap_count": 0,
        "eligible_subject_count": 10,
        "eligible_event_count": 5,
        "invalid_images_included": 0,
        "minimum_outer_test_events": 2,
        "minimum_inner_validation_events": 2,
        "all_images_finite": True,
        "all_images_geometry_valid": True,
    }
    policy = assess_image_training_policy(
        integrity=integrity,
        quantitative_equivalence_proven=False,
        scanner_robustness_proven=False,
        external_validation_available=False,
        incremental_value_demonstrated=False,
    )
    assert policy["training_execution_allowed"] is True
    assert policy["positive_claim_allowed"] is False
    assert "QUANTITATIVE_EQUIVALENCE_NOT_PROVEN" in policy["claim_limitations"]


def test_privacy_scan_fails_on_leak(tmp_path):
    from src.method_y_revival.image_reporting_v3 import privacy_scan_reports

    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    # good report
    (reports / "good.json").write_text(json.dumps({"aggregate": {"c_index": 0.7}}), encoding="utf-8")
    scan = privacy_scan_reports(reports)
    assert scan["status"] == "PASS"
    # leak
    (reports / "bad.json").write_text(json.dumps({"aggregate": "H:\\New project\\secret"}), encoding="utf-8")
    scan2 = privacy_scan_reports(reports)
    assert scan2["status"] == "FAIL"
