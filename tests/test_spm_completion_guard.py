import json
from pathlib import Path

from scripts.spm_completion_guard import spm_completion_ready, spm_completion_status


def write_manifest(path: Path, *, done: int, total: int, config_hash: str) -> None:
    path.parent.mkdir(parents=True)
    items = {
        f"s_{idx:04d}": {"status": "DONE" if idx < done else "PENDING"}
        for idx in range(total)
    }
    path.write_text(json.dumps({"config_hash": config_hash, "items": items}), encoding="utf-8")


def write_preprocessed_files(path: Path, count: int) -> None:
    path.mkdir(parents=True)
    for idx in range(count):
        (path / f"s_{idx:04d}.nii.gz").write_bytes(b"synthetic")
    scratch = path / "_scratch_spm"
    scratch.mkdir()
    (scratch / "s_ignored.nii.gz").write_bytes(b"synthetic")


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

def test_run_project_stage_blocks_gated_stage_before_spm_complete(monkeypatch):
    from scripts import run_project_stage

    monkeypatch.setattr(run_project_stage, "spm_completion_status", lambda: {"ready": False})
    monkeypatch.setattr(run_project_stage, "spm_completion_ready", lambda status: bool(status.get("ready")))
    assert run_project_stage.run_stage("train-3d", data_root=None) == 4
