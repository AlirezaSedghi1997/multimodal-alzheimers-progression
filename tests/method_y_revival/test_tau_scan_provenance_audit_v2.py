from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile

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


def _write_quantitative_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    quantitative = tmp_path / "scan_level_tau_quantitative_validation.json"
    quantitative.write_text(
        json.dumps(
            {
                "science_gate_status": "BLOCKED",
                "claim_status": "NO_CLAIM",
                "official_source_match_count": 7,
                "metrics": [
                    {
                        "mask_name": "Mesial_CenTauR",
                        "n_pairs": 7,
                        "pearson_r": 0.812345,
                        "spearman_rho": 0.712345,
                        "bias_image_minus_source": 0.012345,
                        "mae": 0.045678,
                        "bland_altman_mean_bias": 0.012345,
                        "bland_altman_sd": 0.056789,
                        "bland_altman_lower_95": -0.098962,
                        "bland_altman_upper_95": 0.123652,
                        "status": "COMPUTED",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    crosswalk = tmp_path / "tau_source_region_crosswalk_audit_v2.json"
    crosswalk.write_text(
        json.dumps(
            {
                "science_gate_status": "BLOCKED",
                "claim_status": "NO_CLAIM",
                "reference_contract_id": "test_reference_contract",
                "reference_equivalence_proven": False,
            }
        ),
        encoding="utf-8",
    )
    return quantitative, crosswalk


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
    assert report["a4_linear_suvr_subject_visit_association_scan_count"] == 1
    assert "a4_linear_suvr_direct_bridge_scan_count" not in report
    assert report["association_semantics"]["a4_linear_suvr"] == (
        "SUBJECT_VISIT_ASSOCIATION_ONLY_NOT_ACQUISITION_IDENTITY"
    )
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
    quantitative, crosswalk = _write_quantitative_artifacts(tmp_path)
    # pytest's configured temp root is inside this worktree. The core contract
    # permits private rows only in an external location, so make a transient
    # directory on the worktree's own H: volume and clean it in all outcomes.
    repository_root = Path(__file__).resolve().parents[2]
    assert repository_root.drive.upper() == "H:"
    with tempfile.TemporaryDirectory(prefix="methody-private-", dir=repository_root.anchor) as private_dir:
        private_root = Path(private_dir)
        assert private_root.drive.upper() == "H:"
        assert not (private_root / ".git").exists()
        private_rows = private_root / "rows.parquet"
        report = audit_tau_scan_provenance(
            linkage,
            source_suvr_path=suvr,
            source_stanford_path=stanford,
            output_json=output_json,
            output_report=output_report,
            output_rows=private_rows,
            private_rows_root=private_root,
            quantitative_validation_path=quantitative,
            crosswalk_audit_path=crosswalk,
        )
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        serialized = json.dumps(payload)
        assert "ftp_path" not in serialized and "sidecar_path" not in serialized
        assert "B1" not in serialized and "rows_path" not in payload
        assert private_rows.is_file()
        assert not list(private_root.glob("*.tmp*"))
        rendered = output_report.read_text(encoding="utf-8")
        assert "scan-provenance audit" in rendered.lower()
        assert "0.8123" in rendered
        assert "-0.09896" in rendered
        assert "0.12365" in rendered
        assert "0.9299" not in rendered
        assert "provenance/" not in rendered
        assert "B1" not in rendered
        metric_sources = report["quantitative_metric_sources"]
        assert metric_sources["scan_level_tau_quantitative_validation"]["artifact_name"] == (
            "scan_level_tau_quantitative_validation.json"
        )
        assert metric_sources["tau_source_region_crosswalk_audit"]["artifact_name"] == (
            "tau_source_region_crosswalk_audit_v2.json"
        )
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            for item in metric_sources.values()
        )
        assert report["science_gate_status"] == "BLOCKED"
        metric = report["quantitative_metrics"][0]
        assert metric["bland_altman_mean_bias"] == pytest.approx(0.012345)
        assert metric["bland_altman_sd"] == pytest.approx(0.056789)
        assert metric["bland_altman_lower_95"] == pytest.approx(-0.098962)
        assert metric["bland_altman_upper_95"] == pytest.approx(0.123652)


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


def test_core_api_does_not_accept_a_repository_root_override(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    image = tmp_path / "A4_PET_FTP_B1_004.nii.gz"
    image.write_bytes(b"")
    (tmp_path / "A4_PET_FTP_B1_004.json").write_text("{}", encoding="utf-8")
    linkage = _write_linkage(tmp_path, image)
    suvr, stanford = _write_sources(tmp_path)
    private_root = tmp_path / "private"
    with pytest.raises(TypeError, match="repository_root"):
        audit_tau_scan_provenance(
            linkage,
            source_suvr_path=suvr,
            source_stanford_path=stanford,
            output_rows=private_root / "rows.parquet",
            private_rows_root=private_root,
            repository_root=tmp_path / "unrelated",
        )


def test_core_api_rejects_private_rows_under_any_git_ancestor_before_input_read(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    checkout = tmp_path / "fake-worktree"
    checkout.mkdir()
    # A worktree has a .git file, while a normal checkout has a directory.
    (checkout / ".git").write_text("gitdir: /irrelevant/common-dir\n", encoding="utf-8")
    private_root = checkout / "private"
    with pytest.raises(ValueError, match="PRIVATE_ROWS_OUTPUT_INSIDE_REPOSITORY"):
        audit_tau_scan_provenance(
            tmp_path / "missing-input.parquet",
            source_suvr_path=tmp_path / "missing-suvr.csv",
            source_stanford_path=tmp_path / "missing-stanford.csv",
            output_rows=private_root / "rows.parquet",
            private_rows_root=private_root,
        )
    assert not private_root.exists()


def test_nonblocked_quantitative_input_cannot_publish_a_gate(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    image = tmp_path / "A4_PET_FTP_B1_004.nii.gz"
    image.write_bytes(b"")
    (tmp_path / "A4_PET_FTP_B1_004.json").write_text("{}", encoding="utf-8")
    linkage = _write_linkage(tmp_path, image)
    suvr, stanford = _write_sources(tmp_path)
    quantitative, crosswalk = _write_quantitative_artifacts(tmp_path)
    payload = json.loads(quantitative.read_text(encoding="utf-8"))
    payload["science_gate_status"] = "PASS"
    quantitative.write_text(json.dumps(payload), encoding="utf-8")
    gate = tmp_path / "audit.json"
    report = tmp_path / "report.md"
    gate.write_text('{"generation":"old"}\n', encoding="utf-8")
    report.write_text("old report\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TAU_QUANTITATIVE_ARTIFACT_GATE_NOT_BLOCKED"):
        audit_tau_scan_provenance(
            linkage,
            source_suvr_path=suvr,
            source_stanford_path=stanford,
            output_json=gate,
            output_report=report,
            quantitative_validation_path=quantitative,
            crosswalk_audit_path=crosswalk,
        )
    assert json.loads(gate.read_text(encoding="utf-8")) == {"generation": "old"}
    assert report.read_text(encoding="utf-8") == "old report\n"


def test_missing_bland_altman_field_cannot_publish_a_gate(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    image = tmp_path / "A4_PET_FTP_B1_004.nii.gz"
    image.write_bytes(b"")
    (tmp_path / "A4_PET_FTP_B1_004.json").write_text("{}", encoding="utf-8")
    linkage = _write_linkage(tmp_path, image)
    suvr, stanford = _write_sources(tmp_path)
    quantitative, crosswalk = _write_quantitative_artifacts(tmp_path)
    payload = json.loads(quantitative.read_text(encoding="utf-8"))
    del payload["metrics"][0]["bland_altman_upper_95"]
    quantitative.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="TAU_QUANTITATIVE_ARTIFACT_METRIC_SCHEMA_INVALID"):
        audit_tau_scan_provenance(
            linkage,
            source_suvr_path=suvr,
            source_stanford_path=stanford,
            output_report=tmp_path / "report.md",
            quantitative_validation_path=quantitative,
            crosswalk_audit_path=crosswalk,
        )


def test_report_rejects_noncanonical_metric_artifact_names(tmp_path: Path):
    from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance

    image = tmp_path / "A4_PET_FTP_B1_004.nii.gz"
    image.write_bytes(b"")
    (tmp_path / "A4_PET_FTP_B1_004.json").write_text("{}", encoding="utf-8")
    linkage = _write_linkage(tmp_path, image)
    suvr, stanford = _write_sources(tmp_path)
    quantitative, crosswalk = _write_quantitative_artifacts(tmp_path)
    noncanonical = tmp_path / "B1_quantitative.json"
    quantitative.rename(noncanonical)

    with pytest.raises(ValueError, match="TAU_QUANTITATIVE_ARTIFACT_NAME_INVALID"):
        audit_tau_scan_provenance(
            linkage,
            source_suvr_path=suvr,
            source_stanford_path=stanford,
            output_report=tmp_path / "report.md",
            quantitative_validation_path=noncanonical,
            crosswalk_audit_path=crosswalk,
        )


_FORBIDDEN_PRIVATE_HISTORY = (
    "results/method_y_revival_v2/provenance/tau_scan_source_identity_audit_v2.parquet",
    ".superpowers/sdd/pasted-text-1.txt/task-1-report.md",
)
_PUBLISHABLE_REF = "methody/pretraining-sanitized-v2"


def _git_text(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.replace("\\", "/")


def test_publishable_sanitized_ref_has_no_private_tau_or_sdd_report_objects():
    repository_root = Path(__file__).resolve().parents[2]
    objects = _git_text(repository_root, "rev-list", _PUBLISHABLE_REF, "--objects")
    history = _git_text(repository_root, "log", _PUBLISHABLE_REF, "--format=", "--name-only")
    for path in _FORBIDDEN_PRIVATE_HISTORY:
        assert path not in objects
        assert path not in history


def test_index_and_working_tree_track_no_parquet():
    repository_root = Path(__file__).resolve().parents[2]
    tracked = _git_text(repository_root, "ls-files", "*.parquet").strip()
    assert tracked == ""


def test_live_head_is_not_the_publishable_ref_when_private_history_is_reachable():
    """Report leaky HEAD as not publishable. Do not require the leak to remain."""

    repository_root = Path(__file__).resolve().parents[2]
    objects = _git_text(repository_root, "rev-list", "HEAD", "--objects")
    leak_present = any(path in objects for path in _FORBIDDEN_PRIVATE_HISTORY)
    head_ref = _git_text(repository_root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if leak_present:
        assert head_ref != _PUBLISHABLE_REF



def test_refreshed_public_audit_is_locked_to_transform_eligible_manifest():
    """Regression guard for the public 1,314-scan aggregate refresh.

    The private manifest itself remains outside this worktree. The public
    artifact may expose only its basename, aggregate row count, and digest.
    """

    repository_root = Path(__file__).resolve().parents[2]
    audit_path = repository_root / "results/method_y_revival_v2/provenance/tau_scan_source_identity_audit_v2.json"
    report_path = repository_root / "results/method_y_revival_v2/reports/TAU_QUANTITATIVE_VALIDATION_V2.md"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = payload["input_manifest"]
    assert manifest == {
        "artifact_name": "private_ftp_t1_transform_eligible.parquet",
        "row_count": 1314,
        "sha256": "73556b25cbf3c9a50b498f0c166aa8b9b8b6f95164f1571dff6aca52179f2917",
    }
    assert payload["scan_count"] == 1314
    assert payload["exact_scan_identity_scan_count"] == 0
    assert payload["science_gate_status"] == "BLOCKED"
    assert payload["claim_status"] == "NO_CLAIM"
    assert payload["quantitative_metric_sources"] == {
        "scan_level_tau_quantitative_validation": {
            "artifact_name": "scan_level_tau_quantitative_validation.json",
            "sha256": "05c2a4a016393e1079e330044210aad03e48a6be08744b10f6c9e493d3381da1",
        },
        "tau_source_region_crosswalk_audit": {
            "artifact_name": "tau_source_region_crosswalk_audit_v2.json",
            "sha256": "b5bc619fff4ba82d99cbaf3e279b568a217b335fcb51084d2ebd49b727caa55a",
        },
    }
    assert payload["quantitative_metrics"][0]["bland_altman_lower_95"] == pytest.approx(-0.11670741601445787)
    assert payload["quantitative_metrics"][0]["bland_altman_upper_95"] == pytest.approx(0.12140727562951524)
    rendered = report_path.read_text(encoding="utf-8")
    assert "private_ftp_t1_transform_eligible.parquet" in rendered
    assert manifest["sha256"] in rendered
    for metric in payload["quantitative_metrics"]:
        for field in (
            "bland_altman_mean_bias",
            "bland_altman_sd",
            "bland_altman_lower_95",
            "bland_altman_upper_95",
        ):
            assert field in metric


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
    quantitative, crosswalk = _write_quantitative_artifacts(tmp_path)

    with pytest.raises(OSError):
        audit_tau_scan_provenance(
            linkage,
            source_suvr_path=suvr,
            source_stanford_path=stanford,
            output_json=gate,
            output_report=invalid_report_target,
            quantitative_validation_path=quantitative,
            crosswalk_audit_path=crosswalk,
        )
    assert json.loads(gate.read_text(encoding="utf-8")) == {"generation": "old"}
