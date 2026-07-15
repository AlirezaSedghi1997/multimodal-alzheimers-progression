from __future__ import annotations

import numpy as np
import pandas as pd


def _frame() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rng = np.random.default_rng(31)
    n = 90
    pet = rng.normal(size=n)
    event = np.asarray([(index % 3) != 0 for index in range(n)], dtype=int)
    frame = pd.DataFrame(
        {
            "patient_hash": [f"p-{index:03d}" for index in range(n)],
            "event": event,
            "time_days": 250.0 + np.arange(n) * 12.0 - pet * 15.0,
            "treatment_arm": ["PLACEBO" if index % 2 else "SOLANEZUMAB" for index in range(n)],
            "cohort": ["A4"] * n,
            "age": 70.0 + rng.normal(size=n),
            "baseline_pacc_raw": rng.normal(size=n),
            "baseline_amyloid_centiloid": pet,
        }
    )
    vendors = ["GE", "PHILIPS", "SIEMENS"]
    manufacturers = np.repeat(vendors, n // len(vendors)).astype(object)
    rng.shuffle(manufacturers)
    linkage = [
        {
            "patient_hash": patient,
            "manufacturer": manufacturer,
            "scanner_sensitivity_eligible": True,
        }
        for patient, manufacturer in zip(frame.patient_hash, manufacturers, strict=True)
    ]
    return frame, linkage


def test_leave_manufacturer_out_is_nested_resumable_and_never_uses_unknown(tmp_path, monkeypatch):
    import src.method_y_revival.scanner_sensitivity as sensitivity_module
    from src.method_y_revival.scanner_sensitivity import run_leave_manufacturer_out_sensitivity

    frame, linkage = _frame()
    report = run_leave_manufacturer_out_sensitivity(
        frame=frame,
        linkage_rows=linkage,
        feature_sets={
            "clinical_cognition": ["age", "baseline_pacc_raw"],
            "clinical_cognition_amyloid": [
                "age",
                "baseline_pacc_raw",
                "baseline_amyloid_centiloid",
            ],
        },
        base_variant="clinical_cognition",
        augmented_variant="clinical_cognition_amyloid",
        output_root=tmp_path,
        branch="amyloid",
        inner_folds=2,
        seed=19,
        alpha_grid=[0.1],
        requested_horizons=[400.0, 700.0],
        min_at_risk=2,
        bootstrap_iterations=20,
        minimum_subjects=20,
        minimum_events=5,
        minimum_manufacturers=2,
    )

    assert report["status"] == "COMPLETED"
    assert report["heldout_manufacturers"] == ["GE", "PHILIPS", "SIEMENS"]
    assert all(variant["outer_folds_completed"] == 3 for variant in report["variants"].values())
    assert report["paired_bootstrap"]["status"] == "COMPLETED"
    assert report["unknown_imputed"] is False

    def fail_if_hpo_repeated(*args, **kwargs):
        raise AssertionError("locked leave-manufacturer-out HPO must be reused")

    monkeypatch.setattr(sensitivity_module, "_select_alpha", fail_if_hpo_repeated)
    repeated = run_leave_manufacturer_out_sensitivity(
        frame=frame,
        linkage_rows=linkage,
        feature_sets=report["feature_sets"],
        base_variant="clinical_cognition",
        augmented_variant="clinical_cognition_amyloid",
        output_root=tmp_path,
        branch="amyloid",
        inner_folds=2,
        seed=19,
        alpha_grid=[0.1],
        requested_horizons=[400.0, 700.0],
        min_at_risk=2,
        bootstrap_iterations=20,
        minimum_subjects=20,
        minimum_events=5,
        minimum_manufacturers=2,
    )
    assert repeated["status"] == "COMPLETED"


def test_leave_manufacturer_out_blocks_before_training_when_event_support_is_insufficient(tmp_path):
    from src.method_y_revival.scanner_sensitivity import run_leave_manufacturer_out_sensitivity

    frame, linkage = _frame()
    report = run_leave_manufacturer_out_sensitivity(
        frame=frame,
        linkage_rows=linkage,
        feature_sets={"base": ["age"], "pet": ["age", "baseline_amyloid_centiloid"]},
        base_variant="base",
        augmented_variant="pet",
        output_root=tmp_path,
        branch="amyloid",
        inner_folds=2,
        seed=19,
        alpha_grid=[0.1],
        requested_horizons=[400.0],
        min_at_risk=2,
        bootstrap_iterations=20,
        minimum_subjects=20,
        minimum_events=100,
        minimum_manufacturers=2,
    )

    assert report["status"] == "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT"
    assert report["scanner_sensitivity_passed"] is False
    assert not (tmp_path / "scanner_sensitivity" / "amyloid").exists()
