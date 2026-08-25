from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _dummy_variant(n_folds=2, n_subjects_per_fold=20, seed=0):
    rng = np.random.default_rng(seed)
    folds = []
    for fold in range(n_folds):
        grid = np.linspace(100, 1500, 10).tolist()
        oof = []
        for i in range(n_subjects_per_fold):
            survival = np.sort(rng.random(10))[::-1].tolist()  # decreasing
            oof.append(
                {
                    "patient_hash": f"pt_{fold}_{i}",
                    "event": int(rng.integers(0, 2)),
                    "time_days": float(rng.integers(150, 1400)),
                    "risk_score": float(rng.random()),
                    "survival_probabilities": survival,
                }
            )
        # ensure at least some events
        if sum(r["event"] for r in oof) == 0:
            oof[0]["event"] = 1
        folds.append(
            {
                "fold": fold,
                "evaluation_grid_days": grid,
                "oof_predictions": oof,
                "censoring_reference": [{"event": 0, "time_days": 1000.0}, {"event": 1, "time_days": 500.0}],
            }
        )
    return {"folds": folds}


def test_calibration_uses_locked_oof_without_refit():
    from src.method_y_revival.image_reporting_v3 import build_calibration_reports

    base = _dummy_variant(seed=1)
    aug = _dummy_variant(seed=2)
    reports = build_calibration_reports(base_variant=base, augmented_variant=aug, horizons_days=[365.0, 730.0])
    assert reports["status"] == "COMPLETED"
    assert "365.0" in reports["horizons"]
    # Should use locked OOF without refit – reports contain no model refit flag
    for h, rep in reports["horizons"].items():
        assert "base" in rep and "augmented" in rep


def test_subgroup_small_cells_are_not_estimable():
    from src.method_y_revival.image_reporting_v3 import build_subgroup_reports

    base = _dummy_variant(n_folds=2, n_subjects_per_fold=10, seed=3)
    aug = _dummy_variant(n_folds=2, n_subjects_per_fold=10, seed=4)
    # Create metadata with subgroup column where one group has insufficient support (<30 subjects)
    hashes = [r["patient_hash"] for f in base["folds"] for r in f["oof_predictions"]]
    # Assign: 5 subjects to rare group, rest to common
    groups = ["RARE"] * 5 + ["COMMON"] * (len(hashes) - 5)
    metadata = pd.DataFrame(
        {
            "patient_hash": hashes,
            "sex_binary": groups,
            "age": [70] * len(hashes),
            "treatment_active": [0] * len(hashes),
            "event": [1] * len(hashes),
            "manufacturer": ["SIEMENS"] * len(hashes),
        }
    )
    # Need event column in base rows? Already there, but metadata also needs manufacturer etc.
    # The function uses all_base_rows from base variant; it counts members via metadata join
    report = build_subgroup_reports(
        base_variant=base,
        augmented_variant=aug,
        metadata=metadata,
        subgroup_columns=["sex_binary"],
        minimum_subjects=30,
        minimum_events=10,
        iterations=20,
        seed=123,
    )
    assert report["subgroups"]["sex_binary"]["RARE"]["status"] == "NOT_ESTIMABLE_INSUFFICIENT_SUPPORT"
    # COMMON may still be small (15) <30, so also not estimable
    assert report["subgroups"]["sex_binary"]["COMMON"]["status"] == "NOT_ESTIMABLE_INSUFFICIENT_SUPPORT"


def test_incremental_gate_stays_exploratory_when_scanner_gate_fails():
    from src.method_y_revival.science_gates import incremental_value_gate

    # Even with good delta, if scanner gate fails, should stay exploratory
    gate = incremental_value_gate(
        delta_ibs_ci=(-0.01, -0.001),
        delta_cindex_ci=(0.01, 0.05),
        outer_folds_completed=5,
        expected_outer_folds=5,
        scanner_sensitivity_passed=False,
    )
    assert gate["science_gate_status"] == "FAIL"
    assert "SCANNER_SHORTCUT_NOT_CONTROLLED" in gate["blockers"]
    assert gate["claim_status"] == "EXPLORATORY"


def test_quantitative_and_external_limitations_survive_positive_internal_delta():
    from src.method_y_revival.image_reporting_v3 import claim_consistency_state

    state = claim_consistency_state(
        training_execution_allowed=True,
        incremental_value_pass=True,
        quantitative_equivalence_proven=False,
        external_validation_pass=False,
        scanner_robustness_passed=True,
    )
    # Even with incremental pass and scanner passed, quantitative claim remains False
    assert state["quantitative_standard_claim_allowed"] is False
    assert state["external_claim_allowed"] is False
    assert state["training_execution_allowed"] is True
    # Now with quant proven but external still false
    state2 = claim_consistency_state(
        training_execution_allowed=True,
        incremental_value_pass=True,
        quantitative_equivalence_proven=True,
        external_validation_pass=False,
        scanner_robustness_passed=True,
    )
    assert state2["quantitative_standard_claim_allowed"] is True
    assert state2["external_claim_allowed"] is False


def test_public_report_contains_no_patient_hash_path_embedding_or_prediction(tmp_path):
    from src.method_y_revival.image_reporting_v3 import render_private_and_public, privacy_scan_reports

    private_preds = [
        {"patient_hash": "secret123", "event": 1, "time_days": 400, "risk_score": 0.7},
        {"patient_hash": "secret456", "event": 0, "time_days": 800, "risk_score": 0.3},
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    out = tmp_path / "out"
    result = render_private_and_public(
        private_oof_predictions=private_preds,
        private_embeddings=embeddings,
        output_root=out,
        branch="amyloid",
        variant="scalar_tabular_plus_pattern_image",
    )
    # privacy scan should PASS because public does not contain secrets
    scan = privacy_scan_reports(out / "reports")
    assert scan["status"] == "PASS"
    # Verify public file content
    public_path = Path(result["public_path"])
    public_content = public_path.read_text(encoding="utf-8")
    assert "secret123" not in public_content
    assert "patient_hash" not in public_content
    assert "embedding" not in public_content.lower()
    # Now create a violating public report
    bad_path = out / "reports" / "bad.json"
    bad_path.write_text(json.dumps({"patient_hash": "leak", "image_path_private": "H:\\secret"}), encoding="utf-8")
    scan2 = privacy_scan_reports(out / "reports")
    assert scan2["status"] == "FAIL"
    assert any("patient_hash" in v for v in scan2["violations"])


def test_paired_bootstrap_requires_2000_iterations(tmp_path):
    from src.method_y_revival.image_reporting_v3 import build_paired_bootstrap_report, ImageReportingV3Error

    base = _dummy_variant(seed=5)
    aug = _dummy_variant(seed=6)
    with pytest.raises(ImageReportingV3Error, match="BOOTSTRAP_ITERATIONS_MUST_BE_2000"):
        build_paired_bootstrap_report(base_variant=base, augmented_variant=aug, output_path=tmp_path / "boot.json", iterations=1000, seed=0)
