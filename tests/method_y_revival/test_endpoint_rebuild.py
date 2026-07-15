from __future__ import annotations

import json

import pytest


def test_new_endpoint_rule_dates_event_at_first_increase_after_confirmation():
    from scripts.build_method_y_tau_progression import detect_sustained_event

    followups = [
        {"_date": 30.0, "_cdglobal": 0.5},
        {"_date": 40.0, "_cdglobal": 0.5},
    ]

    old_rule = detect_sustained_event(followups)
    new_rule = detect_sustained_event(
        followups, event_time_rule="first_increase_if_confirmed_by_next_visit"
    )

    assert old_rule == (True, 40.0)
    assert new_rule == (True, 30.0)


def test_locked_endpoint_writer_is_atomic_hashes_rows_and_rejects_duplicates(tmp_path):
    from src.method_y_revival.endpoint_builder import EndpointBuildError, write_locked_endpoint

    rows = [
        {"patient_hash": "pt-a", "event": 1, "time_days": 300.0},
        {"patient_hash": "pt-b", "event": 0, "time_days": 500.0},
    ]
    output = tmp_path / "tau.parquet"
    contract_path = tmp_path / "endpoint_contract.json"
    report = write_locked_endpoint(
        rows,
        output_path=output,
        contract_path=contract_path,
        source_hashes={"tau": "a" * 64},
        endpoint_definition={"rule": "future_sustained_cdr"},
    )

    stored_output = tmp_path / report["endpoint_path"]
    assert stored_output.is_file()
    assert report["row_count"] == 2
    assert len(report["endpoint_sha256"]) == 64
    assert report["storage_format"] in {"parquet", "csv_fallback"}
    if report["storage_format"] == "csv_fallback":
        assert report["parquet_status"] == "BLOCKED_PARQUET_ENGINE_UNAVAILABLE"
        assert stored_output.suffix == ".csv"
    assert json.loads(contract_path.read_text(encoding="utf-8"))["locked"] is True

    with pytest.raises(EndpointBuildError, match="DUPLICATE_ENDPOINT_SUBJECT"):
        write_locked_endpoint(
            [rows[0], rows[0]],
            output_path=tmp_path / "duplicate.parquet",
            contract_path=tmp_path / "duplicate.json",
            source_hashes={},
            endpoint_definition={},
        )


def test_endpoint_writer_rejects_raw_identifiers_and_paths(tmp_path):
    from src.method_y_revival.endpoint_builder import EndpointBuildError, write_locked_endpoint

    with pytest.raises(EndpointBuildError, match="PRIVATE_ENDPOINT_FIELD_FORBIDDEN"):
        write_locked_endpoint(
            [{"patient_hash": "pt-a", "event": 1, "time_days": 1.0, "raw_path": "private"}],
            output_path=tmp_path / "bad.parquet",
            contract_path=tmp_path / "bad.json",
            source_hashes={},
            endpoint_definition={},
        )


def test_revival_to_legacy_endpoint_adapter_contains_every_required_field():
    from src.method_y_revival.endpoint_builder import compatibility_endpoint_config

    config = {
        "endpoint": {
            "primary_name": "sustained_cdr_global_progression",
            "pet_to_baseline_window_days": 183,
            "followup_window_days": 1640.0,
            "baseline_cdr_required": 0.0,
            "event_time_rule": "first_increase_if_confirmed_by_next_visit",
            "forbidden_label_sources": ["CDEVENT", "CDOLEEVENT"],
        },
        "validation": {
            "minimum_total_events": 60,
            "minimum_events_per_treatment_arm": 20,
        }
    }
    adapted = compatibility_endpoint_config(config)

    assert adapted["endpoint"]["primary_name"] == "sustained_cdr_global_progression"
    assert adapted["endpoint"]["event_time_rule"] == "first_increase_if_confirmed_by_next_visit"
    assert adapted["validation"]["min_total_events"] == 60


def test_endpoint_contract_paths_are_branch_specific(tmp_path):
    from src.method_y_revival.endpoint_builder import endpoint_contract_path

    assert endpoint_contract_path(tmp_path, "tau").name == "tau_endpoint_contract.json"
    assert endpoint_contract_path(tmp_path, "amyloid").name == "amyloid_endpoint_contract.json"
