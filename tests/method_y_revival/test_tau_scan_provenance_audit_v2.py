from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_parse_and_sidecar_identity_helpers_are_exact_only(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import (
        identity_fields_from_sidecar,
        parse_ftp_identity,
    )

    assert parse_ftp_identity(r"H:\A4 Learn\FTP\A4_PET_FTP_B12345678_004.nii.gz") == (
        "B12345678",
        "4",
    )
    assert identity_fields_from_sidecar(
        {"StudyInstanceUID": "1.2.3", "Nested": {"AccessionNumber": "A1"}}
    ) == {"StudyInstanceUID": "1.2.3", "AccessionNumber": "A1"}
    assert identity_fields_from_sidecar({"StudyDescription": "B12345678"}) == {}


def test_audit_records_a4_direct_bridge_but_keeps_gate_blocked(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    raw = tmp_path / "raw"
    raw.mkdir()
    image = raw / "A4_PET_FTP_B1_004.nii.gz"
    sidecar = raw / "A4_PET_FTP_B1_004.json"
    image.write_bytes(b"")
    sidecar.write_text(json.dumps({"StudyDescription": "B1"}), encoding="utf-8")
    linkage = tmp_path / "linkage.parquet"
    pd.DataFrame(
        [{"scan_key": "s1", "subject_key": "h1", "ftp_path": str(image)}]
    ).to_parquet(linkage, index=False)
    suvr = tmp_path / "imaging_SUVR_tau.csv"
    pd.DataFrame(
        [
            {"BID": "B1", "VISCODE": 4, "ligand": "Flortaucipir", "scan_analyzed": "Yes"},
            {"BID": "B1", "VISCODE": 4, "ligand": "Flortaucipir", "scan_analyzed": "Yes"},
        ]
    ).to_csv(suvr, index=False)
    stanford = tmp_path / "imaging_Tau_PET_Stanford.csv"
    pd.DataFrame([{"BID": "B1", "VISCODE": 2}]).to_csv(stanford, index=False)

    report = audit_tau_scan_provenance(
        linkage, source_suvr_path=suvr, source_stanford_path=stanford
    )

    assert report["a4_linear_suvr_direct_bridge_scan_count"] == 1
    assert report["exact_dicom_identity_scan_count"] == 0
    assert report["science_gate_status"] == "BLOCKED"
    assert "EXACT_DICOM_SCAN_IDENTITY_UNAVAILABLE" in report["blockers"]
    assert "REFERENCE_PIPELINE_EQUIVALENCE_NOT_PROVEN" in report["blockers"]
