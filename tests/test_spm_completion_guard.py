import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.spm_completion_guard import spm_completion_ready, spm_completion_status


def write_manifest(path: Path, *, done: int, total: int, config_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = {
        f"s_{idx:04d}": {"status": "DONE" if idx < done else "PENDING"}
        for idx in range(total)
    }
    path.write_text(json.dumps({"config_hash": config_hash, "items": items}), encoding="utf-8")


def write_preprocessed_files(path: Path, count: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        image = nib.Nifti1Image(
            np.ones((128, 128, 128), dtype=np.float32),
            np.diag([1.5, 1.5, 1.5, 1.0]),
        )
        nib.save(image, str(path / f"s_{idx:04d}.nii.gz"))
    scratch = path / "_scratch_spm"
    scratch.mkdir()
    (scratch / "s_ignored.nii.gz").write_bytes(b"synthetic")


def write_qc_files(path: Path, count: int, *, invalid: set[int] | None = None) -> None:
    qc = path / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    invalid = invalid or set()
    for idx in range(count):
        normalized_nonzero_fraction = 0.0 if idx in invalid else 0.5
        normalized_brain_std = 0.0 if idx in invalid else 1.0
        (qc / f"s_{idx:04d}.json").write_text(
            json.dumps(
                {
                    "normalized_nonzero_fraction": normalized_nonzero_fraction,
                    "normalized_brain_std": normalized_brain_std,
                    "contains_pet_input": False,
                    "contains_raw_identifiers": False,
                }
            ),
            encoding="utf-8",
        )


def write_catalog(path: Path, sample_hashes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_hash"])
        writer.writeheader()
        for sample_hash in sample_hashes:
            writer.writerow({"sample_hash": sample_hash})


def test_spm_completion_guard_requires_manifest_hash_done_and_file_count(tmp_path):
    manifest = tmp_path / "manifest.json"
    preprocessed = tmp_path / "preprocessed"
    write_manifest(manifest, done=2, total=2, config_hash="expected")
    write_preprocessed_files(preprocessed, 2)

    status = spm_completion_status(
        manifest_path=manifest,
        preprocessed_dir=preprocessed,
        expected_config_hash="expected",
        expected_total=2,
    )

    assert spm_completion_ready(status) is True
    assert status["done_count"] == 2
    assert status["preprocessed_file_count"] == 2
    assert status["privacy"]["subject_level_outputs_exported"] is False


def test_spm_completion_guard_blocks_incomplete_done_count(tmp_path):
    manifest = tmp_path / "manifest.json"
    preprocessed = tmp_path / "preprocessed"
    write_manifest(manifest, done=1, total=2, config_hash="expected")
    write_preprocessed_files(preprocessed, 2)

    status = spm_completion_status(
        manifest_path=manifest,
        preprocessed_dir=preprocessed,
        expected_config_hash="expected",
        expected_total=2,
    )

    assert spm_completion_ready(status) is False
    assert status["checks"]["done_count_ok"] is False


def test_spm_completion_guard_accepts_terminal_rows_when_outputs_and_qc_reconcile(tmp_path):
    manifest = tmp_path / "manifest.json"
    preprocessed = tmp_path / "preprocessed"
    write_manifest(manifest, done=2, total=2, config_hash="expected")
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    obj["items"]["s_0001"]["status"] = "BLOCKED"
    manifest.write_text(json.dumps(obj), encoding="utf-8")
    write_preprocessed_files(preprocessed, 2)
    write_qc_files(preprocessed, 2)

    status = spm_completion_status(
        manifest_path=manifest,
        preprocessed_dir=preprocessed,
        expected_config_hash="expected",
        expected_total=2,
    )

    assert spm_completion_ready(status) is True
    assert status["status"] == "SPM_COMPLETE_WITH_RECONCILED_TERMINAL_ROWS"
    assert status["done_count"] == 1
    assert status["manifest_terminal_count"] == 2


def test_spm_completion_guard_blocks_wrong_config_hash(tmp_path):
    manifest = tmp_path / "manifest.json"
    preprocessed = tmp_path / "preprocessed"
    write_manifest(manifest, done=2, total=2, config_hash="wrong")
    write_preprocessed_files(preprocessed, 2)

    status = spm_completion_status(
        manifest_path=manifest,
        preprocessed_dir=preprocessed,
        expected_config_hash="expected",
        expected_total=2,
    )

    assert spm_completion_ready(status) is False
    assert status["checks"]["config_hash_ok"] is False


def test_spm_completion_guard_blocks_empty_normalized_qc(tmp_path):
    manifest = tmp_path / "manifest.json"
    preprocessed = tmp_path / "preprocessed"
    write_manifest(manifest, done=2, total=2, config_hash="expected")
    write_preprocessed_files(preprocessed, 2)
    write_qc_files(preprocessed, 2, invalid={1})

    status = spm_completion_status(
        manifest_path=manifest,
        preprocessed_dir=preprocessed,
        expected_config_hash="expected",
        expected_total=2,
    )

    assert spm_completion_ready(status) is False
    assert status["checks"]["preprocessed_qc_ok"] is False
    assert status["preprocessed_qc_invalid_count"] == 1

def test_spm_completion_guard_counts_active_catalog_only(tmp_path):
    manifest = tmp_path / "manifest.json"
    preprocessed = tmp_path / "preprocessed"
    catalog = tmp_path / "catalog.csv"
    preprocessed.mkdir(parents=True)
    image = nib.Nifti1Image(np.ones((128, 128, 128), dtype=np.float32), np.diag([1.5, 1.5, 1.5, 1.0]))
    nib.save(image, str(preprocessed / "s_keep.nii.gz"))
    nib.save(image, str(preprocessed / "s_excluded.nii.gz"))
    write_catalog(catalog, ["s_keep"])
    qc = preprocessed / "qc"
    qc.mkdir(parents=True)
    (qc / "s_keep.json").write_text(json.dumps({"normalized_nonzero_fraction": 0.5, "normalized_brain_std": 1.0}), encoding="utf-8")
    (qc / "s_excluded.json").write_text(json.dumps({"normalized_nonzero_fraction": 0.0, "normalized_brain_std": 0.0}), encoding="utf-8")
    manifest.write_text(json.dumps({"config_hash": "expected", "items": {"s_keep": {"status": "DONE"}, "s_excluded": {"status": "DONE"}}}), encoding="utf-8")

    status = spm_completion_status(
        manifest_path=manifest,
        preprocessed_dir=preprocessed,
        expected_config_hash="expected",
        catalog_path=catalog,
    )

    assert spm_completion_ready(status) is True
    assert status["expected_total"] == 1
    assert status["preprocessed_qc_invalid_count"] == 0
    assert status["preprocessed_qc_missing_count"] == 0


def test_spm_completion_guard_reports_missing_qc_separately_from_invalid(tmp_path):
    manifest = tmp_path / "manifest.json"
    preprocessed = tmp_path / "preprocessed"
    catalog = tmp_path / "catalog.csv"
    write_manifest(manifest, done=1, total=2, config_hash="expected")
    write_preprocessed_files(preprocessed, 1)
    write_qc_files(preprocessed, 1)
    write_catalog(catalog, ["s_0000", "s_0001"])

    status = spm_completion_status(
        manifest_path=manifest,
        preprocessed_dir=preprocessed,
        expected_config_hash="expected",
        catalog_path=catalog,
    )

    assert spm_completion_ready(status) is False
    assert status["preprocessed_qc_file_count"] == 1
    assert status["preprocessed_qc_missing_count"] == 1
    assert status["preprocessed_qc_invalid_count"] == 0


def test_run_project_stage_blocks_gated_stage_before_spm_complete(monkeypatch):
    from scripts import run_project_stage

    monkeypatch.setattr(run_project_stage, "spm_completion_status", lambda: {"ready": False})
    monkeypatch.setattr(run_project_stage, "spm_completion_ready", lambda status: bool(status.get("ready")))
    assert run_project_stage.run_stage("train-3d", data_root=None) == 4


def test_spm_completion_guard_reports_pre_spm_boundaries(tmp_path):
    manifest = tmp_path / "manifest.json"
    preprocessed = tmp_path / "preprocessed"
    write_manifest(manifest, done=1, total=2, config_hash="expected")
    write_preprocessed_files(preprocessed, 1)

    status = spm_completion_status(
        manifest_path=manifest,
        preprocessed_dir=preprocessed,
        expected_config_hash="expected",
        expected_total=2,
    )

    assert "mri-volume-cache" in status["blocked_post_spm_stages"]
    assert "mri-volume-cache-smoke" in status["safe_before_spm_complete"]
    assert status["source_of_truth"]["manifest_path"] == str(manifest)
    assert status["next_action"] == "continue_spm_preprocessing"
