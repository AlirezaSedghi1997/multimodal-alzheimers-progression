from __future__ import annotations

import csv


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_amyloid_scanner_linkage_requires_exact_patient_visit_and_day():
    from scripts.build_method_y_tau_progression import patient_hash_for_key
    from src.method_y_revival.scanner_audit import link_amyloid_scanner_rows

    patient = patient_hash_for_key(("A4", "1"))
    first = {
        ("A4", "1"): {
            "subject_key": ("A4", "1"),
            "viscode": "2",
            "scan_date": 20.0,
        }
    }
    private = [
        {"patient_hash": patient, "visit_code": "1", "pet_day": "10", "sample_hash": "wrong"},
        {"patient_hash": patient, "visit_code": "2", "pet_day": "20", "sample_hash": "exact"},
    ]
    pet_input = [
        {"subject_id_hash": "wrong", "tracer": "FBP", "scanner": "GE"},
        {"subject_id_hash": "exact", "tracer": "FBP", "scanner": "Siemens:Biograph"},
    ]

    rows, audit = link_amyloid_scanner_rows(first, private, pet_input, {patient})

    assert rows == [
        {
            "branch": "amyloid",
            "patient_hash": patient,
            "manufacturer": "SIEMENS",
            "linkage_status": "EXACT",
            "scanner_sensitivity_eligible": True,
        }
    ]
    assert audit["exact"] == 1


def test_amyloid_scanner_linkage_does_not_guess_from_another_visit():
    from scripts.build_method_y_tau_progression import patient_hash_for_key
    from src.method_y_revival.scanner_audit import link_amyloid_scanner_rows

    patient = patient_hash_for_key(("A4", "1"))
    first = {("A4", "1"): {"subject_key": ("A4", "1"), "viscode": "2", "scan_date": 20.0}}
    private = [{"patient_hash": patient, "visit_code": "1", "pet_day": "10", "sample_hash": "other"}]
    pet_input = [{"subject_id_hash": "other", "tracer": "FBP", "scanner": "GE"}]

    rows, audit = link_amyloid_scanner_rows(first, private, pet_input, {patient})

    assert rows[0]["manufacturer"] == "UNKNOWN"
    assert rows[0]["linkage_status"] == "MISSING_EXACT_SCAN_LINK"
    assert rows[0]["scanner_sensitivity_eligible"] is False
    assert audit["missing_exact_scan_link"] == 1


def test_amyloid_scanner_linkage_accepts_exact_subject_visit_tracer_when_day_origins_differ():
    from scripts.build_method_y_tau_progression import patient_hash_for_key
    from src.method_y_revival.scanner_audit import link_amyloid_scanner_rows

    patient = patient_hash_for_key(("A4", "1"))
    first = {("A4", "1"): {"subject_key": ("A4", "1"), "viscode": "2", "scan_date": 20.0}}
    private = [{"patient_hash": patient, "visit_code": "2", "pet_day": "1481", "sample_hash": "same-session"}]
    pet_input = [{"subject_id_hash": "same-session", "tracer": "FBP", "scanner": "Philips"}]

    rows, audit = link_amyloid_scanner_rows(first, private, pet_input, {patient})

    assert rows[0]["manufacturer"] == "PHILIPS"
    assert rows[0]["linkage_status"] == "EXACT_SUBJECT_VISIT_TRACER"
    assert rows[0]["scanner_sensitivity_eligible"] is True
    assert audit["exact_subject_visit_tracer"] == 1


def test_tau_scanner_linkage_excludes_qc_failure_and_keeps_unknown_unknown():
    from src.method_y_revival.scanner_audit import link_tau_scanner_rows

    rows, audit = link_tau_scanner_rows(
        {"p1", "p2", "p3"},
        [
            {"patient_hash": "p1", "manufacturer": "GE", "qc_status": "PASS"},
            {"patient_hash": "p2", "manufacturer": "MISSING", "qc_status": "PASS"},
            {"patient_hash": "p3", "manufacturer": "Philips", "qc_status": "SIDECAR_RADIOPHARM_CONTRADICTS_FTP"},
        ],
    )

    assert {row["patient_hash"]: row["scanner_sensitivity_eligible"] for row in rows} == {
        "p1": True,
        "p2": False,
        "p3": False,
    }
    assert audit["eligible"] == 1
    assert audit["unknown"] == 1
    assert audit["qc_excluded"] == 1


def test_scanner_audit_builder_writes_only_hashed_private_linkage(tmp_path):
    import pandas as pd

    from scripts.build_method_y_tau_progression import patient_hash_for_key
    from src.method_y_revival.endpoint_builder import write_locked_endpoint
    from src.method_y_revival.scanner_audit import build_scanner_audit_from_sources

    project = tmp_path / "project"
    raw = tmp_path / "raw"
    output = project / "results" / "revival"
    amyloid_patient = patient_hash_for_key(("A4", "1"))
    write_locked_endpoint(
        [{"patient_hash": amyloid_patient, "event": 0, "time_days": 100.0, "cohort": "A4", "treatment_arm": "PLACEBO"}],
        output_path=output / "endpoints" / "amyloid.parquet",
        contract_path=output / "endpoints" / "amyloid_endpoint_contract.json",
        source_hashes={},
        endpoint_definition={"tracer": "FBP"},
    )
    write_locked_endpoint(
        [{"patient_hash": "tau-1", "event": 1, "time_days": 100.0, "cohort": "A4", "treatment_arm": "PLACEBO"}],
        output_path=output / "endpoints" / "tau.parquet",
        contract_path=output / "endpoints" / "tau_endpoint_contract.json",
        source_hashes={},
        endpoint_definition={"tracer": "FTP"},
    )
    _write_csv(raw / "amyloid.csv", [{"SUBSTUDY": "A4", "BID": "1", "VISCODE": "1", "scan_number": "1", "scan_date_DAYS_T0": "0", "ligand": "Florbetapir", "scan_analyzed": "Yes", "brain_region": "Composite_Summary", "suvr_cer": "1", "suvr_persi": "1", "suvr_crus": "1"}])
    _write_csv(project / "private.csv", [{"patient_hash": amyloid_patient, "visit_code": "1", "pet_day": "0", "sample_hash": "sample-1"}])
    _write_csv(project / "pet.csv", [{"subject_id_hash": "sample-1", "tracer": "FBP", "scanner": "GE"}])
    _write_csv(project / "tau_manifest.csv", [{"patient_hash": "tau-1", "manufacturer": "Philips", "qc_status": "PASS"}])
    config = {
        "paths": {
            "raw_root": str(raw),
            "amyloid_input_manifest": "pet.csv",
            "amyloid_private_manifest": "private.csv",
            "tau_image_manifest": "tau_manifest.csv",
        },
        "clinical_sources": {"amyloid_suvr": "amyloid.csv"},
    }

    audit = build_scanner_audit_from_sources(
        config=config, output_root=output, project_root=project
    )

    linkage = pd.read_csv(output / "qc" / "scanner_linkage_private.csv")
    assert len(linkage) == 2
    assert set(linkage.columns) == {
        "branch",
        "patient_hash",
        "manufacturer",
        "linkage_status",
        "scanner_sensitivity_eligible",
    }
    assert audit["amyloid"]["exact"] == 1
    assert audit["tau"]["eligible"] == 1
    assert audit["amyloid_sensitivity_feasibility"]["status"].startswith("BLOCKED")
    assert audit["tau_sensitivity_feasibility"]["status"].startswith("BLOCKED")


def test_scanner_feasibility_requires_two_manufacturers_with_enough_events():
    import pandas as pd

    from src.method_y_revival.scanner_audit import scanner_sensitivity_feasibility

    endpoint = pd.DataFrame(
        {
            "patient_hash": [f"p{i}" for i in range(12)],
            "event": [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            "cohort": ["A4"] * 12,
        }
    )
    linkage = [
        {
            "patient_hash": f"p{i}",
            "manufacturer": "GE" if i < 6 else "SIEMENS",
            "scanner_sensitivity_eligible": True,
        }
        for i in range(12)
    ]

    ready = scanner_sensitivity_feasibility(
        endpoint,
        linkage,
        development_cohort="A4",
        minimum_subjects=5,
        minimum_events=2,
        minimum_manufacturers=2,
    )
    blocked = scanner_sensitivity_feasibility(
        endpoint,
        linkage,
        development_cohort="A4",
        minimum_subjects=5,
        minimum_events=3,
        minimum_manufacturers=2,
    )

    assert ready["status"] == "READY_FOR_LEAVE_MANUFACTURER_OUT"
    assert blocked["status"] == "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT"
