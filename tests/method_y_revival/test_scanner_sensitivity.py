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


def test_scanner_gate_cannot_promote_claim_without_quantitative_feature_provenance():
    from src.method_y_revival.scanner_sensitivity import apply_scanner_gate_to_branch_report

    report = {
        "paired_bootstrap": {
            "delta_ibs_95ci": [-0.02, -0.01],
            "delta_cindex_95ci": [0.01, 0.03],
        },
        "variants": {"pet": {"outer_folds_completed": 5}},
    }
    blocked = apply_scanner_gate_to_branch_report(
        report,
        augmented_variant="pet",
        expected_outer_folds=5,
        scanner_sensitivity_passed=True,
        quantitative_feature_provenance_passed=False,
    )
    passed = apply_scanner_gate_to_branch_report(
        report,
        augmented_variant="pet",
        expected_outer_folds=5,
        scanner_sensitivity_passed=True,
        quantitative_feature_provenance_passed=True,
    )

    assert blocked["science_gate_status"] == "FAIL"
    assert "QUANTITATIVE_PET_FEATURE_PROVENANCE_NOT_FORMALLY_VALIDATED" in blocked["blockers"]
    assert blocked["claim_status"] == "EXPLORATORY"
    assert passed["science_gate_status"] == "PASS"
    assert passed["claim_status"] == "INTERNAL_ONLY"


def test_locked_scanner_wrapper_updates_claim_gates_without_promoting_unvalidated_features(
    tmp_path, monkeypatch
):
    import src.method_y_revival.scanner_sensitivity as module
    from src.method_y_revival.atomic_io import atomic_write_json
    from src.method_y_revival.endpoint_builder import write_locked_endpoint

    root = tmp_path / "run"
    project = tmp_path / "project"
    amyloid_rows = [
        {
            "patient_hash": "a1",
            "event": 1,
            "time_days": 300.0,
            "cohort": "A4",
            "treatment_arm": "PLACEBO",
            "treatment_active": 0.0,
            "age": 70.0,
            "baseline_amyloid_centiloid": 30.0,
        }
    ]
    tau_rows = [
        {
            "patient_hash": "t1",
            "event": 1,
            "time_days": 300.0,
            "cohort": "A4_placebo",
            "treatment_arm": "PLACEBO",
            "treatment_active": 0.0,
            "age": 70.0,
        }
    ]
    for branch, rows in (("amyloid", amyloid_rows), ("tau", tau_rows)):
        write_locked_endpoint(
            rows,
            output_path=root / "endpoints" / f"{branch}.parquet",
            contract_path=root / "endpoints" / f"{branch}_endpoint_contract.json",
            source_hashes={},
            endpoint_definition={"tracer": "FBP" if branch == "amyloid" else "FTP"},
        )
        atomic_write_json(
            root / "reports" / f"{branch}_regional_development.json",
            {
                "paired_bootstrap": {
                    "delta_ibs_95ci": [-0.02, -0.01],
                    "delta_cindex_95ci": [0.01, 0.03],
                },
                "variants": {
                    "clinical_cognition_pet" if branch == "tau" else "clinical_cognition_amyloid": {
                        "outer_folds_completed": 5
                    }
                },
            },
        )
    (root / "qc").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"branch": "amyloid", "patient_hash": "a1", "manufacturer": "GE", "linkage_status": "EXACT", "scanner_sensitivity_eligible": True},
            {"branch": "tau", "patient_hash": "t1", "manufacturer": "GE", "linkage_status": "EXACT", "scanner_sensitivity_eligible": True},
        ]
    ).to_csv(root / "qc" / "scanner_linkage_private.csv", index=False)
    project.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"patient_hash": "t1", "tau_persi__roi": 1.2}]).to_csv(
        project / "tau_features.csv", index=False
    )
    config = {
        "seed": 19,
        "paths": {"tau_regional_features": "tau_features.csv"},
        "validation": {
            "inner_folds": 3,
            "outer_folds": 5,
            "requested_horizons_days": [300.0],
            "minimum_at_risk_per_horizon": 2,
            "bootstrap_iterations": 20,
        },
        "hpo": {"cox": {"alpha": [0.1]}},
        "scanner_sensitivity": {
            "minimum_subjects_per_manufacturer": 30,
            "minimum_events_per_manufacturer": 10,
            "minimum_eligible_manufacturers": 2,
        },
        "science_gates": {"quantitative_feature_provenance_passed": False},
    }

    def fake_leave(*, branch, **kwargs):
        return {
            "status": "COMPLETED" if branch == "amyloid" else "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT",
            "scanner_sensitivity_passed": branch == "amyloid",
        }

    monkeypatch.setattr(module, "run_leave_manufacturer_out_sensitivity", fake_leave)
    result = module.run_locked_scanner_sensitivities(
        config=config, output_root=root, project_root=project
    )

    assert result["amyloid"]["scanner_sensitivity_passed"] is True
    amyloid_report = module.read_json(root / "reports" / "amyloid_regional_development.json")
    assert amyloid_report["claim_status"] == "EXPLORATORY"
    assert "QUANTITATIVE_PET_FEATURE_PROVENANCE_NOT_FORMALLY_VALIDATED" in amyloid_report["science_gate"]["blockers"]
