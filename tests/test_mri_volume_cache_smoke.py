import csv
import json
from pathlib import Path

from scripts.mri_volume_cache_smoke import (
    build_public_smoke_summary,
    collect_smoke_inputs,
    ensure_smoke_output_path_allowed,
)


def write_catalog(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_hash",
        "patient_hash",
        "ab_label",
        "age",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
        "sex",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_smoke_selects_only_done_catalog_rows_with_preprocessed_files(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config_hash": "expected",
                "items": {
                    "s_done_with_file": {"status": "DONE"},
                    "s_done_missing_file": {"status": "DONE"},
                    "s_pending_with_file": {"status": "PENDING"},
                },
            }
        ),
        encoding="utf-8",
    )
    preprocessed = tmp_path / "preprocessed"
    preprocessed.mkdir()
    (preprocessed / "s_done_with_file.nii.gz").write_bytes(b"synthetic")
    (preprocessed / "s_pending_with_file.nii.gz").write_bytes(b"synthetic")
    scratch = preprocessed / "_scratch_spm"
    scratch.mkdir()
    (scratch / "s_done_missing_file.nii.gz").write_bytes(b"ignored")

    catalog = tmp_path / "catalog.csv"
    write_catalog(
        catalog,
        [
            {"sample_hash": "s_done_with_file", "patient_hash": "p1", "ab_label": "1", "age": "70", "education_years": "16", "apoe4_carrier": "1", "mmse_base": "29", "sex": "M"},
            {"sample_hash": "s_done_missing_file", "patient_hash": "p2", "ab_label": "0", "age": "71", "education_years": "12", "apoe4_carrier": "0", "mmse_base": "28", "sex": "F"},
            {"sample_hash": "s_pending_with_file", "patient_hash": "p3", "ab_label": "1", "age": "72", "education_years": "14", "apoe4_carrier": "0", "mmse_base": "27", "sex": "M"},
        ],
    )

    selected, preprocessed_index, counts = collect_smoke_inputs(
        catalog_path=catalog,
        manifest_path=manifest,
        preprocessed_dir=preprocessed,
        limit=10,
    )

    assert [row["sample_hash"] for row in selected] == ["s_done_with_file"]
    assert set(preprocessed_index) == {"s_done_with_file", "s_pending_with_file"}
    assert counts["manifest_done_count"] == 2
    assert counts["manifest_total"] == 3
    assert counts["indexed_preprocessed"] == 2
    assert counts["candidate_rows"] == 1


def test_smoke_refuses_official_or_training_cache_locations():
    bad_paths = [
        Path("data/cached/t1_volume_cache_96.h5"),
        Path("data/cached/mri_volume_cache_smoke.h5"),
        Path("results/tables/mri_volume_cache_summary.json"),
    ]
    for bad_path in bad_paths:
        try:
            ensure_smoke_output_path_allowed(bad_path)
        except ValueError as exc:
            assert "official" in str(exc) or "data/cached" in str(exc)
        else:
            raise AssertionError(f"expected {bad_path} to be rejected")


def test_smoke_summary_is_aggregate_and_not_for_training():
    summary = build_public_smoke_summary(
        counts={
            "catalog_rows": 3,
            "manifest_done_count": 2,
            "manifest_total": 3,
            "indexed_preprocessed": 2,
            "candidate_rows": 1,
            "limit": 10,
        },
        rows_written=1,
        errors=0,
        resolution=96,
        device="auto",
        hdf5_persisted=False,
    )

    assert summary["status"] == "SMOKE_ONLY_NOT_FOR_TRAINING"
    assert summary["official_cache_written"] is False
    assert summary["training_usable"] is False
    assert summary["subject_level_outputs_exported"] is False
    assert summary["contains_pet_input"] is False
    assert summary["raw_paths_exported"] is False
    assert "selected_sample_hashes" not in summary
    assert "hdf5_path" not in summary

def test_smoke_stage_is_explicit_and_not_in_main_pipeline():
    from scripts.run_project_stage import POST_SPM_GATED_STAGES, STAGE_COMMANDS, STAGE_ORDER

    assert STAGE_COMMANDS["mri-volume-cache-smoke"] == ["scripts/mri_volume_cache_smoke.py"]
    assert "mri-volume-cache-smoke" not in STAGE_ORDER
    assert "mri-volume-cache-smoke" not in POST_SPM_GATED_STAGES
