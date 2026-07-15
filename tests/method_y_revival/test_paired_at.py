from __future__ import annotations


def _config():
    return {
        "endpoint": {
            "primary_name": "sustained_cdr_global_progression",
            "pet_to_baseline_window_days": 183,
            "followup_window_days": 1640.0,
            "baseline_cdr_required": 0.0,
            "event_time_rule": "first_increase_if_confirmed_by_next_visit",
            "forbidden_label_sources": ["CDEVENT", "CDOLEEVENT"],
        },
        "validation": {
            "minimum_total_events": 1,
            "minimum_events_per_treatment_arm": 1,
        },
        "branches": {"paired_at": {"scan_window_days": 183}},
    }


def test_paired_endpoint_time_zero_is_after_both_pet_scans():
    from src.method_y_revival.endpoint_builder import build_paired_at_endpoint_rows

    key = ("A4", "1")
    amyloid = {
        key: {
            "subject_key": key,
            "viscode": "2",
            "scan_number": "1",
            "scan_date": 100.0,
            "regions": {"Composite_Summary": {"suvr_persi": 1.2}},
        }
    }
    tau = {
        key: {
            "subject_key": key,
            "viscode": "1",
            "scan_number": "1",
            "scan_date": 0.0,
            "regions": {"roi": {"suvr_persi": 1.4, "suvr_crus": 1.1}},
        }
    }
    subjects = [
        {"SUBSTUDY": "A4", "BID": "1", "TX": "PLACEBO", "AGEYR": "70", "AMYLCENT": "42"}
    ]
    cdr = [
        {"SUBSTUDY": "A4", "BID": "1", "VISCODE": "2", "CDADTC_DAYS_T0": "100", "CDGLOBAL": "0", "CDRSB": "0"},
        {"SUBSTUDY": "A4", "BID": "1", "VISCODE": "3", "CDADTC_DAYS_T0": "200", "CDGLOBAL": "0.5", "CDRSB": "1"},
        {"SUBSTUDY": "A4", "BID": "1", "VISCODE": "4", "CDADTC_DAYS_T0": "300", "CDGLOBAL": "0.5", "CDRSB": "1"},
    ]

    rows, audit, regions = build_paired_at_endpoint_rows(
        amyloid, tau, subjects, cdr, [], _config()
    )

    assert len(rows) == 1
    assert rows[0]["time_days"] == 100.0
    assert rows[0]["paired_fbp_ftp_gap_days"] == 100.0
    assert rows[0]["treatment_active"] == 0.0
    assert regions == ["tau_persi__roi"]
    assert audit["paired_within_window"] == 1
    assert audit["time_zero_after_both_pet_scans"] is True


def test_paired_endpoint_excludes_scans_outside_prespecified_window():
    from src.method_y_revival.endpoint_builder import build_paired_at_endpoint_rows

    key = ("A4", "1")
    amyloid = {key: {"subject_key": key, "viscode": "2", "scan_number": "1", "scan_date": 200.0, "regions": {}}}
    tau = {key: {"subject_key": key, "viscode": "1", "scan_number": "1", "scan_date": 0.0, "regions": {}}}

    rows, audit, _ = build_paired_at_endpoint_rows(amyloid, tau, [], [], [], _config())

    assert rows == []
    assert audit["excluded_outside_scan_window"] == 1


def test_paired_at_training_compares_clinical_then_amyloid_then_tau(tmp_path):
    import numpy as np

    from src.method_y_revival.endpoint_builder import write_locked_endpoint
    from src.method_y_revival.regional_training import (
        run_paired_at_regional_development_from_locked_endpoint,
    )

    rng = np.random.default_rng(7)
    rows = []
    for index in range(48):
        rows.append(
            {
                "patient_hash": f"p-{index:03d}",
                "event": int(index % 3 != 0),
                "time_days": 250.0 + index * 20.0,
                "cohort": "A4_placebo" if index % 2 else "A4_treatment",
                "treatment_arm": "PLACEBO" if index % 2 else "SOLANEZUMAB",
                "treatment_active": 0.0 if index % 2 else 1.0,
                "age": 70.0 + rng.normal(),
                "baseline_amyloid_centiloid": float(rng.normal()),
                "tau_persi__roi": float(rng.normal()),
            }
        )
    write_locked_endpoint(
        rows,
        output_path=tmp_path / "endpoints" / "paired.parquet",
        contract_path=tmp_path / "endpoints" / "paired_at_endpoint_contract.json",
        source_hashes={},
        endpoint_definition={"branch": "paired_at"},
    )
    config = {
        "seed": 19,
        "validation": {
            "outer_folds": 3,
            "inner_folds": 2,
            "requested_horizons_days": [400.0, 700.0],
            "minimum_at_risk_per_horizon": 2,
            "bootstrap_iterations": 20,
        },
        "hpo": {"cox": {"alpha": [0.1]}},
    }

    report = run_paired_at_regional_development_from_locked_endpoint(
        config=config, output_root=tmp_path
    )

    features = report["locked_config"]["feature_sets"]
    assert set(features) == {
        "clinical_cognition",
        "clinical_cognition_amyloid",
        "clinical_cognition_amyloid_tau",
    }
    assert set(features["clinical_cognition_amyloid"]) - set(
        features["clinical_cognition"]
    ) == {"baseline_amyloid_centiloid"}
    assert set(features["clinical_cognition_amyloid_tau"]) - set(
        features["clinical_cognition_amyloid"]
    ) == {"tau_persi__roi"}
    assert report["tau_beyond_amyloid_bootstrap"]["status"] == "COMPLETED"
    assert report["claim_status"] == "EXPLORATORY"
