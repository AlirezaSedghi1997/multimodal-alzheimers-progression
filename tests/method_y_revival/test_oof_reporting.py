from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _variant(*, improved: bool) -> dict:
    folds = []
    for fold in range(2):
        predictions = []
        for index in range(24):
            event = int(index % 3 == 0)
            time = 150.0 if event else 350.0
            base_risk = 0.25 + 0.04 * event + 0.002 * index
            risk = base_risk + (0.20 * event if improved else 0.0)
            survival = [0.98, 1.0 - risk, max(0.05, 1.0 - risk - 0.1)]
            predictions.append(
                {
                    "patient_hash": f"pt_{fold}_{index:02d}",
                    "event": event,
                    "time_days": time,
                    "original_event": event,
                    "original_time_days": time,
                    "risk_score": risk,
                    "survival_probabilities": survival,
                }
            )
        folds.append(
            {
                "fold": fold,
                "evaluation_grid_days": [100.0, 200.0, 300.0],
                "oof_predictions": predictions,
                "censoring_reference": [
                    {"event": int(index % 3 == 0), "time_days": 150.0 if index % 3 == 0 else 400.0}
                    for index in range(50)
                ],
            }
        )
    return {"folds": folds, "outer_folds_completed": 2}


def test_horizon_calibration_uses_locked_predictions_without_recalibration():
    from src.method_y_revival.oof_reporting import horizon_calibration_from_oof

    report = horizon_calibration_from_oof(
        _variant(improved=True), horizon_days=200.0, minimum_effective_rows=20
    )

    assert report["status"] == "ESTIMABLE"
    assert report["recalibration_applied"] is False
    assert report["effective_row_count"] == 48
    assert math.isfinite(report["calibration_intercept"])
    assert math.isfinite(report["calibration_slope"])
    assert len(report["calibration_slope_95ci"]) == 2


def test_prespecified_subgroups_are_joined_without_imputing_unknown_scanner():
    from src.method_y_revival.oof_reporting import build_subgroup_metadata

    endpoint = pd.DataFrame(
        {
            "patient_hash": ["a", "b"],
            "age": [69.0, 78.0],
            "sex_binary": [0.0, 1.0],
            "treatment_active": [1.0, 0.0],
            "event": [0, 1],
        }
    )
    scanner = pd.DataFrame(
        {"patient_hash": ["a"], "manufacturer": ["Siemens"]}
    )

    metadata = build_subgroup_metadata(endpoint=endpoint, scanner_linkage=scanner)

    assert metadata.set_index("patient_hash").loc["a", "age_group"] == "LT70"
    assert metadata.set_index("patient_hash").loc["b", "age_group"] == "GE75"
    assert metadata.set_index("patient_hash").loc["b", "manufacturer"] == "UNKNOWN"
    assert metadata.attrs["unknown_scanner_imputed"] is False


def test_subgroup_incremental_report_has_paired_cis_but_never_positive_subgroup_claims():
    from src.method_y_revival.oof_reporting import subgroup_incremental_report

    base = _variant(improved=False)
    augmented = _variant(improved=True)
    metadata = pd.DataFrame(
        {
            "patient_hash": [row["patient_hash"] for fold in base["folds"] for row in fold["oof_predictions"]],
            "sex_binary": ["F"] * 48,
            "event": [row["event"] for fold in base["folds"] for row in fold["oof_predictions"]],
        }
    )

    report = subgroup_incremental_report(
        base_variant=base,
        augmented_variant=augmented,
        metadata=metadata,
        subgroup_columns=["sex_binary"],
        minimum_subjects=30,
        minimum_events=10,
        iterations=40,
        seed=9,
    )

    group = report["subgroups"]["sex_binary"]["F"]
    assert group["status"] == "ESTIMABLE"
    assert len(group["delta_cindex_95ci"]) == 2
    assert group["positive_subgroup_claim_allowed"] is False
    assert report["multiplicity_status"] == "DESCRIPTIVE_NO_POSITIVE_SUBGROUP_CLAIMS"
