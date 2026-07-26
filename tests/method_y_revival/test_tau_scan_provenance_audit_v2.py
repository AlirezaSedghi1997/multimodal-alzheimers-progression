from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest


def _write_sources(tmp_path: Path, *, tracer: str = "Flortaucipir") -> tuple[Path, Path]:
    suvr = tmp_path / "imaging_SUVR_tau.csv"
    pd.DataFrame(
        [
            {"BID": "B1", "VISCODE": 4, "ligand": tracer, "scan_analyzed": "Yes"},
            {"BID": "B1", "VISCODE": 4, "ligand": tracer, "scan_analyzed": "Yes"},
        ]
    ).to_csv(suvr, index=False)
    stanford = tmp_path / "imaging_Tau_PET_Stanford.csv"
    pd.DataFrame([{"BID": "B1", "VISCODE": 2}]).to_csv(stanford, index=False)
    return suvr, stanford


def _write_linkage(tmp_path: Path, image: Path) -> Path:
    linkage = tmp_path / "linkage.parquet"
    pd.DataFrame(
        [{"scan_key": "s1", "subject_key": "h1", "ftp_path": str(image)}]
    ).to_parquet(linkage, index=False)
    return linkage


def test_identity_aliases_reject_null_false_and_normalize_nested_keys():
    from src.method_y_revival.tau_scan_provenance_audit import identity_fields_from_sidecar

    assert identity_fields_from_sidecar(
        {
            "Study_Instance_UID": "1.2.3",
            "Nested": {"accession-number": "A1"},
            "SeriesInstanceUID": None,
            "AccessionNumber": False,
            "StudyInstanceUID": "null",
        }
    ) == {"StudyInstanceUID": "1.2.3", "AccessionNumber": "A1"}


def test_nifti_header_fields_and_extensions_are_scanned(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import _read_nifti_header

    image = tmp_path / "scan.nii.gz"
    nifti = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    nifti.header["descrip"] = b"Study_Instance_UID=1.2.840.9"
    nifti.header.extensions.append(nib.nifti1.Nifti1Extension(40, b'{"accession-number":"A9"}'))
    nib.save(nifti, image)
    status, fields = _read_nifti_header(image)
    assert status == "EXACT_ID_FIELDS_PRESENT"
    assert fields == {"StudyInstanceUID": "1.2.840.9", "AccessionNumber": "A9"}


def test_subject_visit_associations_never_claim_exact_scan_identity(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    raw = tmp_path / "raw"
    raw.mkdir()
    image = raw / "A4_PET_FTP_B1_004.nii.gz"
    image.write_bytes(b"not-a-nifti")
    (raw / "A4_PET_FTP_B1_004.json").write_text(
        json.dumps({"StudyDescription": "B1"}), encoding="utf-8"
    )
    linkage = _write_linkage(tmp_path, image)
    suvr, stanford = _write_sources(tmp_path)
    report = audit_tau_scan_provenance(
        linkage, source_suvr_path=suvr, source_stanford_path=stanford
    )
    assert report["a4_linear_suvr_direct_bridge_scan_count"] == 1
    assert report["stanford_bridge_status_counts"] == {
        "NUMERICAL_VISIT_ASSOCIATION_TO_STANFORD_VISCODE_2": 1
    }
    assert report["exact_scan_identity_scan_count"] == 0
    assert "EXACT_DICOM_SCAN_IDENTITY_UNAVAILABLE" in report["blockers"]
    assert report["science_gate_status"] == "BLOCKED"


def test_positive_sidecar_identity_is_aggregated_with_header_identity(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    raw = tmp_path / "raw"
    raw.mkdir()
    image = raw / "A4_PET_FTP_B1_004.nii.gz"
    nifti = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    nifti.header["aux_file"] = b"Series_Instance_UID=9.8.7"
    nib.save(nifti, image)
    (raw / "A4_PET_FTP_B1_004.json").write_text(
        json.dumps({"StudyInstanceUID": "1.2.3"}), encoding="utf-8"
    )
    linkage = _write_linkage(tmp_path, image)
    suvr, stanford = _write_sources(tmp_path)
    report = audit_tau_scan_provenance(
        linkage, source_suvr_path=suvr, source_stanford_path=stanford
    )
    assert report["sidecar_identity_scan_count"] == 1
    assert report["nifti_header_identity_scan_count"] == 1
    assert report["exact_scan_identity_scan_count"] == 1
    assert report["exact_scan_identity_fields"] == ["SeriesInstanceUID", "StudyInstanceUID"]
    assert "EXACT_DICOM_SCAN_IDENTITY_UNAVAILABLE" not in report["blockers"]
    assert report["science_gate_status"] == "BLOCKED"


def test_source_schema_and_tracer_rejection_are_fail_closed(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    image = tmp_path / "A4_PET_FTP_B1_004.nii.gz"
    image.write_bytes(b"")
    (tmp_path / "A4_PET_FTP_B1_004.json").write_text("{}", encoding="utf-8")
    linkage = _write_linkage(tmp_path, image)
    _, stanford = _write_sources(tmp_path, tracer="Amyloid")
    with pytest.raises(ValueError, match="TAU_SOURCE_TRACER_NO_VALID_FLORTAUCIPIR_ROWS"):
        audit_tau_scan_provenance(
            linkage,
            source_suvr_path=tmp_path / "imaging_SUVR_tau.csv",
            source_stanford_path=stanford,
        )
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"BID": "B1", "VISCODE": 4}]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="TAU_SOURCE_TRACER_COLUMNS_MISSING"):
        audit_tau_scan_provenance(linkage, source_suvr_path=bad, source_stanford_path=stanford)


def test_output_is_aggregate_private_and_report_refresh_is_atomic(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    image = tmp_path / "A4_PET_FTP_B1_004.nii.gz"
    image.write_bytes(b"")
    (tmp_path / "A4_PET_FTP_B1_004.json").write_text("{}", encoding="utf-8")
    linkage = _write_linkage(tmp_path, image)
    suvr, stanford = _write_sources(tmp_path)
    output_json = tmp_path / "public" / "audit.json"
    output_report = tmp_path / "public" / "TAU_QUANTITATIVE_VALIDATION_V2.md"
    private_root = tmp_path / "private"
    private_rows = private_root / "rows.parquet"
    report = audit_tau_scan_provenance(
        linkage,
        source_suvr_path=suvr,
        source_stanford_path=stanford,
        output_json=output_json,
        output_report=output_report,
        output_rows=private_rows,
        private_rows_root=private_root,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "ftp_path" not in serialized and "sidecar_path" not in serialized
    assert "B1" not in serialized and "rows_path" not in payload
    assert private_rows.is_file()
    assert not list(private_root.glob("*.tmp*"))
    assert "scan-provenance audit" in output_report.read_text(encoding="utf-8").lower()
    assert report["science_gate_status"] == "BLOCKED"


def test_row_output_requires_explicit_private_root(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    image = tmp_path / "A4_PET_FTP_B1_004.nii.gz"
    image.write_bytes(b"")
    (tmp_path / "A4_PET_FTP_B1_004.json").write_text("{}", encoding="utf-8")
    linkage = _write_linkage(tmp_path, image)
    suvr, stanford = _write_sources(tmp_path)
    with pytest.raises(ValueError, match="PRIVATE_ROWS_ROOT_REQUIRED"):
        audit_tau_scan_provenance(
            linkage,
            source_suvr_path=suvr,
            source_stanford_path=stanford,
            output_rows=tmp_path / "rows.parquet",
        )


def test_atomic_private_parquet_replaces_and_preserves_last_good_on_failure(tmp_path: Path):
    from src.method_y_revival.atomic_io import atomic_write_parquet

    target = tmp_path / "private" / "rows.parquet"
    atomic_write_parquet(target, pd.DataFrame([{"status": "old"}]))
    atomic_write_parquet(target, pd.DataFrame([{"status": "current"}]))
    assert pd.read_parquet(target).to_dict(orient="records") == [{"status": "current"}]

    class FailingFrame:
        def to_parquet(self, path: str | Path, *, index: bool) -> None:
            Path(path).write_bytes(b"partial")
            raise RuntimeError("injected parquet failure")

    with pytest.raises(RuntimeError, match="injected parquet failure"):
        atomic_write_parquet(target, FailingFrame())
    assert pd.read_parquet(target).to_dict(orient="records") == [{"status": "current"}]
    assert not list(target.parent.glob(f".{target.name}.*.tmp.parquet"))


def test_report_failure_does_not_publish_a_new_gate(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    image = tmp_path / "A4_PET_FTP_B1_004.nii.gz"
    image.write_bytes(b"")
    (tmp_path / "A4_PET_FTP_B1_004.json").write_text("{}", encoding="utf-8")
    linkage = _write_linkage(tmp_path, image)
    suvr, stanford = _write_sources(tmp_path)
    gate = tmp_path / "audit.json"
    gate.write_text('{"generation":"old"}\n', encoding="utf-8")
    invalid_report_target = tmp_path / "report-target-is-a-directory"
    invalid_report_target.mkdir()

    with pytest.raises(OSError):
        audit_tau_scan_provenance(
            linkage,
            source_suvr_path=suvr,
            source_stanford_path=stanford,
            output_json=gate,
            output_report=invalid_report_target,
        )
    assert json.loads(gate.read_text(encoding="utf-8")) == {"generation": "old"}
