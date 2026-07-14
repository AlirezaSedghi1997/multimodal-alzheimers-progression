from __future__ import annotations

import pytest


def _row(**overrides):
    from src.method_y_revival.manifest import ScanManifestRow

    values = {
        "subject_key": "sub-hash-1",
        "session_key": "ses-hash-1",
        "scan_key": "scan-hash-1",
        "cohort": "A4",
        "branch": "amyloid",
        "tracer": "FBP",
        "raw_path": r"H:\\A4 Learn\\FBP\\image.nii.gz",
        "raw_sha256": "a" * 64,
        "source_subject_id": "REAL001",
        "manufacturer": "UNKNOWN",
        "metadata_linkage_method": "exact_bids_stem",
        "metadata_linkage_confidence": "EXACT",
        "training_eligible": True,
    }
    values.update(overrides)
    return ScanManifestRow(**values)


def test_fbp_can_only_enter_amyloid_and_ftp_can_only_enter_tau():
    from src.method_y_revival.manifest import ManifestError

    _row(branch="amyloid", tracer="FBP")
    _row(branch="tau", tracer="FTP")

    with pytest.raises(ManifestError, match="TRACER_BRANCH_MISMATCH"):
        _row(branch="tau", tracer="FBP")
    with pytest.raises(ManifestError, match="TRACER_BRANCH_MISMATCH"):
        _row(branch="amyloid", tracer="FTP")


def test_ambiguous_metadata_cannot_be_training_eligible():
    from src.method_y_revival.manifest import ManifestError

    with pytest.raises(ManifestError, match="NON_EXACT_METADATA_CANNOT_TRAIN"):
        _row(metadata_linkage_confidence="AMBIGUOUS", training_eligible=True)


def test_unknown_scanner_is_preserved_not_inferred_from_site():
    row = _row(manufacturer="UNKNOWN")
    assert row.manufacturer == "UNKNOWN"


def test_public_projection_removes_paths_identifiers_and_reversible_scan_keys():
    from src.method_y_revival.privacy import public_manifest_row

    public = public_manifest_row(_row().to_dict())

    assert public["subject_key"] == "sub-hash-1"
    assert "raw_path" not in public
    assert "sidecar_path" not in public
    assert "source_subject_id" not in public
    assert "station_name" not in public
    assert "scan_key" not in public


def test_manifest_digest_is_order_independent_and_counts_subjects_not_frames():
    from src.method_y_revival.manifest import manifest_digest, summarize_manifest

    first = _row(scan_key="scan-a", session_key="ses-a").to_dict()
    second_frame = _row(scan_key="scan-b", session_key="ses-a").to_dict()

    assert manifest_digest([first, second_frame]) == manifest_digest([second_frame, first])
    summary = summarize_manifest([first, second_frame])
    assert summary["row_count"] == 2
    assert summary["unique_subject_count"] == 1
    assert summary["fbp_subject_count"] == 1


def test_conflicting_exact_tracer_evidence_is_quarantined():
    from src.method_y_revival.metadata_linkage import resolve_tracer

    decision = resolve_tracer(
        metadata_tracer="Florbetapir",
        sidecar_tracer="Flortaucipir",
        table_tracer=None,
        filename="sub_pet_FBP.nii.gz",
    )

    assert decision.tracer is None
    assert decision.status == "QUARANTINE"
    assert decision.reason == "CONFLICTING_EXACT_TRACER_EVIDENCE"
