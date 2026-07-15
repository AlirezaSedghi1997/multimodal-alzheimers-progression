from __future__ import annotations

import csv
import json


def _config():
    return {
        "endpoint": {
            "primary_name": "sustained_cdr_global_progression",
            "baseline_cdr_required": 0.0,
            "pet_to_baseline_window_days": 183,
            "followup_window_days": 1640.0,
            "event_time_rule": "first_increase_if_confirmed_by_next_visit",
            "forbidden_label_sources": ["CDEVENT", "CDOLEEVENT"],
        }
    }


def test_amyloid_inventory_accepts_only_analyzed_florbetapir_and_selects_first_scan():
    from src.method_y_revival.endpoint_builder import load_first_amyloid_scans

    rows = [
        {"SUBSTUDY": "A4", "BID": "1", "VISCODE": "2", "scan_number": "2", "scan_date_DAYS_T0": "20", "ligand": "Florbetapir", "scan_analyzed": "Yes", "brain_region": "Composite_Summary", "suvr_persi": "1.3"},
        {"SUBSTUDY": "A4", "BID": "1", "VISCODE": "1", "scan_number": "1", "scan_date_DAYS_T0": "10", "ligand": "Florbetapir", "scan_analyzed": "Yes", "brain_region": "Composite_Summary", "suvr_persi": "1.2"},
        {"SUBSTUDY": "A4", "BID": "2", "VISCODE": "1", "scan_number": "1", "scan_date_DAYS_T0": "10", "ligand": "Flortaucipir", "scan_analyzed": "Yes", "brain_region": "Composite_Summary", "suvr_persi": "1.1"},
        {"SUBSTUDY": "A4", "BID": "3", "VISCODE": "1", "scan_number": "1", "scan_date_DAYS_T0": "10", "ligand": "Florbetapir", "scan_analyzed": "No", "brain_region": "Composite_Summary", "suvr_persi": "1.1"},
    ]

    first, audit = load_first_amyloid_scans(rows)

    assert len(first) == 1
    assert next(iter(first.values()))["scan_date"] == 10.0
    assert audit["tracer"] == "FBP"
    assert audit["first_analyzed_fbp_subjects"] == 1
    assert audit["rejected_non_fbp_rows"] == 1


def test_amyloid_endpoint_is_future_facing_and_excludes_sf():
    from src.method_y_revival.endpoint_builder import build_amyloid_endpoint_rows

    first = {
        ("A4", "1"): {"subject_key": ("A4", "1"), "viscode": "1", "scan_number": "1", "scan_date": 0.0, "regions": {"Composite_Summary": {"suvr_persi": 1.2}}},
        ("LEARN", "2"): {"subject_key": ("LEARN", "2"), "viscode": "1", "scan_number": "1", "scan_date": 0.0, "regions": {"Composite_Summary": {"suvr_persi": 1.1}}},
        ("SF", "3"): {"subject_key": ("SF", "3"), "viscode": "1", "scan_number": "1", "scan_date": 0.0, "regions": {"Composite_Summary": {"suvr_persi": 1.0}}},
    }
    subjects = [
        {"SUBSTUDY": "A4", "BID": "1", "TX": "PLACEBO", "AGEYR": "70", "AMYLCENT": "42"},
        {"SUBSTUDY": "LEARN", "BID": "2", "TX": "NA", "AGEYR": "72", "AMYLCENT": "12"},
        {"SUBSTUDY": "SF", "BID": "3", "TX": "NA", "AGEYR": "73", "AMYLCENT": "8"},
    ]
    cdr = []
    for study, bid in (("A4", "1"), ("LEARN", "2"), ("SF", "3")):
        cdr.extend(
            [
                {"SUBSTUDY": study, "BID": bid, "VISCODE": "1", "CDADTC_DAYS_T0": "0", "CDGLOBAL": "0", "CDRSB": "0"},
                {"SUBSTUDY": study, "BID": bid, "VISCODE": "2", "CDADTC_DAYS_T0": "100", "CDGLOBAL": "0.5", "CDRSB": "1"},
                {"SUBSTUDY": study, "BID": bid, "VISCODE": "3", "CDADTC_DAYS_T0": "200", "CDGLOBAL": "0.5", "CDRSB": "1"},
            ]
        )

    rows, audit = build_amyloid_endpoint_rows(first, subjects, cdr, [], _config())

    assert {row["cohort"] for row in rows} == {"A4", "LEARN"}
    assert all(row["event"] == 1 and row["time_days"] == 100.0 for row in rows)
    assert all("CDEVENT" not in row and "CDOLEEVENT" not in row for row in rows)
    assert {row["baseline_amyloid_centiloid"] for row in rows} == {42.0, 12.0}
    assert next(row for row in rows if row["cohort"] == "A4")["treatment_active"] == 0.0
    assert next(row for row in rows if row["cohort"] == "LEARN")["treatment_active"] == ""
    assert audit["excluded_cohorts"] == {"SF": 1}
    assert audit["development_cohort"] == "A4"
    assert audit["internal_transport_cohort"] == "LEARN"


def test_treatment_structure_is_encoded_without_treating_unknown_as_placebo():
    from src.method_y_revival.endpoint_builder import treatment_active_value

    assert treatment_active_value("PLACEBO") == 0.0
    assert treatment_active_value("SOLANEZUMAB") == 1.0
    assert treatment_active_value("active treatment") == 1.0
    assert treatment_active_value("NA") == ""


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_amyloid_endpoint_builder_locks_a_separate_artifact(tmp_path):
    from src.method_y_revival.endpoint_builder import build_amyloid_endpoint_from_sources

    raw = tmp_path / "raw"
    _write_csv(
        raw / "amyloid.csv",
        [{"SUBSTUDY": "A4", "BID": "1", "VISCODE": "1", "scan_number": "1", "scan_date_DAYS_T0": "0", "ligand": "Florbetapir", "scan_analyzed": "Yes", "brain_region": "Composite_Summary", "suvr_cer": "1.1", "suvr_persi": "1.2", "suvr_crus": "1.3"}],
    )
    _write_csv(
        raw / "subj.csv",
        [{"SUBSTUDY": "A4", "BID": "1", "TX": "PLACEBO", "AGEYR": "70", "AMYLCENT": "42"}],
    )
    _write_csv(
        raw / "cdr.csv",
        [
            {"SUBSTUDY": "A4", "BID": "1", "VISCODE": "1", "CDADTC_DAYS_T0": "0", "CDGLOBAL": "0", "CDRSB": "0"},
            {"SUBSTUDY": "A4", "BID": "1", "VISCODE": "2", "CDADTC_DAYS_T0": "100", "CDGLOBAL": "0", "CDRSB": "0"},
        ],
    )
    _write_csv(raw / "pacc.csv", [{"SUBSTUDY": "A4", "BID": "1", "VISCODE": "1", "PACC.raw": "0"}])
    config = _config() | {
        "paths": {"raw_root": str(raw)},
        "clinical_sources": {
            "amyloid_suvr": "amyloid.csv",
            "cdr": "cdr.csv",
            "pacc": "pacc.csv",
            "subject_info": "subj.csv",
        },
    }

    summary = build_amyloid_endpoint_from_sources(config=config, output_root=tmp_path / "out")

    contract_path = tmp_path / "out" / "endpoints" / "amyloid_endpoint_contract.json"
    assert contract_path.is_file()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["row_count"] == 1
    assert contract["endpoint_definition"]["tracer"] == "FBP"
    assert contract["endpoint_definition"]["treatment_structure"] == "active_vs_placebo_covariate"
    assert summary["status"] == "ENDPOINT_LOCKED"
    assert not (tmp_path / "out" / "endpoints" / "tau_endpoint_contract.json").exists()
