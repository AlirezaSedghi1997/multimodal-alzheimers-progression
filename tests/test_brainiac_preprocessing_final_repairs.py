"""Supervisor acceptance tests for the final BrainIAC preprocessing repair.

These tests exercise production helpers.  They intentionally avoid restating
the expected logic inside the test (the defect in the previous repair suite).
All image fixtures are synthetic and contain no study identifiers or labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from scripts.research import prepare_brainiac_raw_t1_v2 as runner
from src.research import brainiac_dependency_closure as closure
from src.research import brainiac_preprocessing_v2 as v2
from src.research import brainiac_preprocessing_v2_schema as schema


def test_preprocessing_exposes_shared_project_gpu_writer_boundary() -> None:
    """Removing the all-project guard would regress preprocessing to Method-X only."""

    assert callable(getattr(v2, "assert_no_project_gpu_writer", None))


def test_preprocessing_writer_boundary_accepts_test_only_process_providers() -> None:
    """Provider injection keeps writer exclusion deterministic under unit tests."""

    import inspect

    parameters = inspect.signature(v2.assert_no_project_gpu_writer).parameters
    assert "process_provider" in parameters
    assert "gpu_provider" in parameters


def test_preprocessing_boundary_blocks_method_y_writer_before_sample_io() -> None:
    """A live Method-Y process must stop raw-T1 preparation before any write."""

    method_y = {
        "ProcessId": 43213,
        "Name": "python.exe",
        "CommandLine": (
            r"python.exe H:\New project\scripts\run_method_y_image_training_v3.py "
            r"--resume"
        ),
        "CreationDate": "20260828070300.000000+210",
    }
    with pytest.raises(RuntimeError, match="BRAINIAC_PROJECT_GPU_WRITER_BLOCKED"):
        v2.assert_no_project_gpu_writer(
            process_provider=lambda: [method_y],
            gpu_provider=lambda: [],
        )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bindings() -> dict[str, str]:
    return {
        "policy_sha256": _sha("policy"),
        "config_sha256": _sha("config"),
        "pipeline_contract_sha256": _sha("pipeline"),
        "dependency_closure_sha256": _sha("closure"),
        "membership_sha256": _sha("membership"),
        "cache_sha256": _sha("cache"),
        "catalog_sha256": _sha("catalog"),
        "runtime_lock_sha256": _sha("runtime"),
        "vendor_tree_sha256": _sha("vendor"),
        "template_sha256": _sha("template"),
        "template_lock_sha256": _sha("template-lock"),
        "official_transform_source_sha256": _sha("transform-source"),
        "qc_threshold_manifest_sha256": _sha("thresholds"),
    }


def _profile_leg(
    samples: list[str],
    *,
    role: str,
    seconds: float,
    contract: str,
    fresh: bool = True,
) -> dict[str, object]:
    receipt_map = {sample: _sha("receipt-" + sample) for sample in samples}
    output_map = {sample: _sha("output-" + sample) for sample in samples}
    transform_map = {sample: _sha("transform-" + sample) for sample in samples}
    scientific_map = {sample: _sha("scientific-" + sample) for sample in samples}
    return {
        "role": role,
        "root_relative_path": f"fixture/{role}",
        "root_tree_sha256": _sha("tree-" + role),
        "sample_hashes": samples,
        "selection_sha256": runner.canonical_sha(sorted(samples)),
        "receipt_file_sha256_by_sample": receipt_map,
        "output_sha256_by_sample": output_map,
        "transform_sha256_by_sample": transform_map,
        "canonical_scientific_sha256_by_sample": scientific_map,
        "contract_sha256": contract,
        "started_at_utc": "2026-08-28T00:00:00+00:00",
        "finished_at_utc": "2026-08-28T00:02:00+00:00",
        "elapsed_seconds": seconds,
        "fresh_run": fresh,
        "resume_event_count": 0 if fresh else 1,
        "timing_receipt_sha256": _sha("timing-" + role),
        "integrity_status": "PASS",
        "live_evidence_status": "PASS_RECOMPUTED_FROM_LIVE_BYTES",
    }


class _Pipeline:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, item: dict[str, str], scratch: Path) -> dict[str, object]:
        import nibabel as nib

        self.calls += 1
        output = scratch / "output.nii.gz"
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        volume[2:6, 2:6, 2:6] = 1.0
        nib.save(nib.Nifti1Image(volume, np.eye(4)), str(output))
        transform = scratch / "transform.tfm"
        transform.write_bytes(b"#tfm\n")
        affine = np.eye(4, dtype=np.float64).reshape(-1).tolist()
        return {
            "output_path": output,
            "transform_path": transform,
            "source_sha256": _sha("source"),
            "source_size": 10,
            "affine_before": affine,
            "affine_after": affine,
            "spacing_before": [1.0, 1.0, 1.0],
            "spacing_after": [1.0, 1.0, 1.0],
            "origin_before": [0.0, 0.0, 0.0],
            "origin_after": [0.0, 0.0, 0.0],
            "direction_before": np.eye(3).reshape(-1).tolist(),
            "direction_after": np.eye(3).reshape(-1).tolist(),
            "com_coordinate_frame": "TEMPLATE_PHYSICAL_MM_RELATIVE",
            "rigid_determinant": 1.0,
            "orthogonality_error": 0.0,
            "rotation_deg": 0.0,
            "translation_norm_mm": 0.0,
            "foreground_fraction": 0.125,
            "finite": True,
            "normalized_nonzero_mean": 0.0,
            "normalized_nonzero_std": 1.0,
            "center_of_mass_before": [1.0, 0.0, 0.0],
            "center_of_mass_after": [0.5, 0.0, 0.0],
            "registration_similarity_before": -1.0,
            "registration_similarity_after": -1.2,
        }


@pytest.fixture(autouse=True)
def _no_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v2, "assert_no_cnn_writer", lambda: None)


def test_dependency_closure_includes_from_import_submodule_and_static_contract_inputs():
    entry = Path(runner.__file__).resolve()
    files = closure.build_dependency_closure(entry)
    assert "src/research/brainiac_preprocessing_v2.py" in files

    manifest = closure.build_execution_dependency_manifest(
        entry,
        static_inputs={
            "policy": v2.POLICY_PATH,
            "config": runner.ROOT
            / "configs/research/brainiac_preprocessing_r1_primary_v2.yaml",
        },
    )
    assert manifest["entrypoint"] == "scripts/research/prepare_brainiac_raw_t1_v2.py"
    assert set(manifest["static_inputs"]) == {"config", "policy"}
    assert len(manifest["closure_sha256"]) == 64


def test_preprocessing_closure_excludes_audit_only_training_and_pet_modules():
    """Scientific receipts must not drift when unrelated trainers are edited."""

    files = set(runner._dependency_manifest()["python_closure"])
    forbidden = {
        "scripts/train_3d_mri.py",
        "scripts/train_pet_method_y.py",
        "scripts/build_mri_volume_cache.py",
        "src/utils/pet_spatial_qc.py",
    }
    assert not files.intersection(forbidden)


def test_isolated_method_x_source_verifier_matches_canonical_training_contract():
    from scripts.train_3d_mri import method_x_training_source_contract
    from src.research.brainiac_method_x_isolation import (
        verify_frozen_method_x_source_contract,
    )

    canonical, _runtime = method_x_training_source_contract()
    report = verify_frozen_method_x_source_contract(expected_digest=canonical)
    assert report["status"] == "PASS"
    assert report["source_contract_digest"] == canonical
    assert report["file_count"] > 40


def test_config_and_versioned_policy_forbid_runtime_mutation_and_cpu_fallback():
    config = runner._config()
    policy = runner._policy()
    assert config["device"] == {
        "default": "cuda",
        "fallback": "none",
        "cpu_role": "test_fixtures_only_not_production",
    }
    assert policy["immutable"] is True
    assert policy["orchestration"]["runtime_policy_mutation_forbidden"] is True
    assert policy["execution"]["cpu_fallback"] is False
    assert "operator amendment" not in json.dumps(policy).casefold()


def test_dynamic_vendor_runtime_files_are_explicit_dependency_inputs():
    manifest = runner._dependency_manifest()
    dynamic = manifest["dynamic_code_inputs"]
    assert set(dynamic) == set(runner.VENDOR_RUNTIME_CODE)
    assert all(len(binding["sha256"]) == 64 for binding in dynamic.values())


def test_receipt_reuse_rejects_stale_expected_binding(tmp_path: Path):
    item = {"sample_hash": "s_" + "a" * 16, "t1_file_hash": "t1"}
    pipeline = _Pipeline()
    receipt = v2.run_sample_transaction(
        item, tmp_path, pipeline=pipeline, bindings=_bindings()
    )
    expected = _bindings()
    expected["policy_sha256"] = _sha("new-policy")
    with pytest.raises(RuntimeError, match="EXPECTED_BINDING_MISMATCH:policy_sha256"):
        schema.validate_receipt(
            receipt, artifact_root=tmp_path, expected_bindings=expected
        )


def test_threshold_manifest_is_bound_to_receipts_samples_and_contract():
    sample_hashes = [f"s_{i:064x}" for i in range(32)]
    metrics = {
        sample: {
            "foreground_fraction": 0.20 + (i % 5) * 0.005,
            "normalized_nonzero_std": 0.98 + (i % 3) * 0.01,
            "rotation_deg": 2.0 + i * 0.01,
            "translation_norm_mm": 5.0 + i * 0.02,
        }
        for i, sample in enumerate(sample_hashes)
    }
    receipt_hashes = {sample: _sha(f"receipt-{sample}") for sample in sample_hashes}
    contract = {
        "policy_sha256": _sha("policy"),
        "pipeline_contract_sha256": _sha("pipeline"),
        "membership_sha256": _sha("membership"),
    }
    manifest = v2.build_threshold_manifest(
        metrics,
        receipt_integrity_sha256s=receipt_hashes,
        contract_bindings=contract,
    )
    checked = schema.validate_threshold_manifest(
        manifest,
        expected_sample_hashes=sample_hashes,
        expected_receipt_integrity_sha256s=receipt_hashes,
        expected_contract_bindings=contract,
    )
    assert checked["sample_count"] == 32

    stale = dict(contract)
    stale["policy_sha256"] = _sha("stale")
    with pytest.raises(RuntimeError, match="THRESHOLD_MANIFEST_BINDING_MISMATCH"):
        schema.validate_threshold_manifest(
            manifest,
            expected_sample_hashes=sample_hashes,
            expected_receipt_integrity_sha256s=receipt_hashes,
            expected_contract_bindings=stale,
        )


def test_primary_mad_outlier_is_quarantined_by_production_gate():
    manifest = {
        "thresholds": {
            "foreground_fraction": {"low": 0.10, "high": 0.20},
            "rotation_deg": {"low": 0.0, "high": 10.0},
        }
    }
    receipt_like = {"foreground_fraction": 0.25, "rotation_deg": 2.0}
    assert v2.qc_threshold_anomalies(receipt_like, manifest) == [
        "foreground_fraction"
    ]


def test_foreground_normalization_keeps_background_exactly_zero():
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(volume, dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    volume[mask] = np.linspace(10.0, 90.0, int(mask.sum()), dtype=np.float32)
    normalized = v2.normalize_foreground(volume, mask)
    assert np.count_nonzero(normalized[~mask]) == 0
    assert abs(float(normalized[mask].mean())) < 1e-6
    assert abs(float(normalized[mask].std()) - 1.0) < 1e-6


def test_physical_geometry_records_full_affine_and_relative_template_com():
    import SimpleITK as sitk

    image = sitk.Image([7, 7, 7], sitk.sitkFloat32)
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    array = sitk.GetArrayFromImage(image)
    array[3, 3, 3] = 1.0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((10.0, 20.0, 30.0))

    geometry = runner._image_geometry(image)
    assert len(geometry["affine_4x4"]) == 16
    assert geometry["spacing"] == [2.0, 3.0, 4.0]
    com = runner._foreground_com_physical(image)
    assert com == pytest.approx([16.0, 29.0, 42.0])
    assert runner._relative_com(com, com) == pytest.approx([0.0, 0.0, 0.0])


def test_receipt_without_final_journal_recovers_without_reprocessing(
    tmp_path: Path,
):
    item = {"sample_hash": "s_" + "b" * 16, "t1_file_hash": "t1"}
    pipeline = _Pipeline()

    def crash(state: str) -> None:
        if state == "RECEIPT_COMMITTED":
            raise RuntimeError("INJECTED_CRASH_AFTER_RECEIPT")

    with pytest.raises(RuntimeError, match="INJECTED_CRASH_AFTER_RECEIPT"):
        v2.run_sample_transaction(
            item,
            tmp_path,
            pipeline=pipeline,
            bindings=_bindings(),
            fault_hook=crash,
        )
    assert pipeline.calls == 1
    assert not (
        tmp_path / "journals" / item["sample_hash"] / "committed.json"
    ).exists()

    recovered = v2.run_sample_transaction(
        item,
        tmp_path,
        pipeline=lambda *_args: (_ for _ in ()).throw(
            AssertionError("pipeline reran instead of journal recovery")
        ),
        bindings=_bindings(),
    )
    assert recovered["status"] == "COMPLETED"
    assert (
        tmp_path / "journals" / item["sample_hash"] / "committed.json"
    ).is_file()


@pytest.mark.parametrize("state", ["SCRATCH", "PREPARED", "PAYLOAD_COMMITTED"])
def test_pre_receipt_faults_archive_partial_state_and_reprocess(
    tmp_path: Path, state: str
):
    item = {"sample_hash": "s_" + _sha(state)[:16], "t1_file_hash": "t1"}
    first = _Pipeline()

    def crash(observed: str) -> None:
        if observed == state:
            raise RuntimeError("INJECTED_TRANSACTION_FAULT")

    with pytest.raises(RuntimeError, match="INJECTED_TRANSACTION_FAULT"):
        v2.run_sample_transaction(
            item,
            tmp_path,
            pipeline=first,
            bindings=_bindings(),
            fault_hook=crash,
        )
    second = _Pipeline()
    receipt = v2.run_sample_transaction(
        item, tmp_path, pipeline=second, bindings=_bindings()
    )
    assert receipt["status"] == "COMPLETED"
    assert second.calls == 1


def test_corrupt_committed_receipt_and_truncated_journal_fail_closed(tmp_path: Path):
    item = {"sample_hash": "s_" + "e" * 16, "t1_file_hash": "t1"}
    v2.run_sample_transaction(item, tmp_path, pipeline=_Pipeline(), bindings=_bindings())
    receipt_path = tmp_path / "receipts" / f"{item['sample_hash']}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="INTEGRITY_MISMATCH"):
        v2.run_sample_transaction(
            item, tmp_path, pipeline=_Pipeline(), bindings=_bindings()
        )

    # Restore the sealed receipt, then prove a truncated final journal is not
    # treated as an absent journal and silently regenerated.
    receipt_path.unlink()
    second_root = tmp_path / "second"
    v2.run_sample_transaction(
        item, second_root, pipeline=_Pipeline(), bindings=_bindings()
    )
    committed = second_root / "journals" / item["sample_hash"] / "committed.json"
    committed.write_bytes(b'{"status":')
    with pytest.raises(json.JSONDecodeError):
        v2.run_sample_transaction(
            item, second_root, pipeline=_Pipeline(), bindings=_bindings()
        )


def test_proven_dead_sample_lock_is_archived_with_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sample = "s_" + "f" * 16
    lock = tmp_path / "journals" / sample / ".sample.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"pid": 999999}), encoding="utf-8")
    monkeypatch.setattr(v2, "_pid_identity", lambda _pid: (False, None))
    v2._reconcile_stale_sample_lock(lock, root=tmp_path, sample_hash=sample)
    assert not lock.exists()
    archived = list((tmp_path / "quarantine").rglob(".sample.lock"))
    reasons = list((tmp_path / "quarantine").rglob("reason.json"))
    assert len(archived) == 1
    assert any(
        json.loads(path.read_text(encoding="utf-8"))["reason_code"]
        == "PROVEN_DEAD_SAMPLE_LOCK_ARCHIVED"
        for path in reasons
    )


def test_streaming_validation_reports_incomplete_without_building_false_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = {"sample_hash": "s_" + "1" * 16, "t1_file_hash": "t1"}
    second = {"sample_hash": "s_" + "2" * 16, "t1_file_hash": "t2"}
    v2.run_sample_transaction(
        first, tmp_path, pipeline=_Pipeline(), bindings=_bindings()
    )
    monkeypatch.setattr(v2, "PRIMARY_ROOT", tmp_path)
    monkeypatch.setattr(v2, "load_locked_membership", lambda: [first, second])
    monkeypatch.setattr(runner, "_base_bindings", _bindings)
    report = runner._streaming_validate_primary()
    assert report["status"] == "BLOCKED"
    assert report["valid_committed_n"] == 1
    assert report["missing_n"] == 1
    assert report["aggregate"] is None
    assert report["aggregate_integrity_sha256"] is None


def test_status_rejects_receipt_from_stale_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    item = {"sample_hash": "s_" + "3" * 16, "t1_file_hash": "t1"}
    v2.run_sample_transaction(item, tmp_path, pipeline=_Pipeline(), bindings=_bindings())
    stale = _bindings()
    stale["policy_sha256"] = _sha("replacement-policy")
    monkeypatch.setattr(v2, "PRIMARY_ROOT", tmp_path)
    monkeypatch.setattr(v2, "load_locked_membership", lambda: [item])
    monkeypatch.setattr(runner, "_base_bindings", lambda: stale)
    report = runner.run_status()
    assert report["status"] == "BLOCKED"
    assert report["completed_rows"] == 0
    assert report["invalid_rows"] == [item["sample_hash"][:10]]


def test_real_process_kill_restart_matches_uninterrupted_transaction_bytes(
    tmp_path: Path,
):
    worker = (
        Path(__file__).parent / "fixtures" / "brainiac_transaction_worker.py"
    )
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    marker = tmp_path / "kill.marker"
    clean = subprocess.run(
        [sys.executable, str(worker), str(uninterrupted)],
        cwd=runner.ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert clean.returncode == 0, clean.stderr

    environment = dict(os.environ)
    environment["BRAINIAC_FIXTURE_SLEEP_AFTER_MARKER"] = "1"
    child = subprocess.Popen(
        [sys.executable, str(worker), str(resumed), str(marker)],
        cwd=runner.ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 30
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.is_file(), "kill fixture did not reach SCRATCH boundary"
    child.kill()
    child.wait(timeout=15)
    resumed_run = subprocess.run(
        [sys.executable, str(worker), str(resumed)],
        cwd=runner.ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert resumed_run.returncode == 0, resumed_run.stderr

    for role in ("outputs", "transforms", "receipts"):
        clean_files = {
            path.name: path.read_bytes()
            for path in (uninterrupted / role).glob("*")
        }
        resumed_files = {
            path.name: path.read_bytes() for path in (resumed / role).glob("*")
        }
        assert resumed_files == clean_files
    assert (resumed / "aggregate.json").read_bytes() == (
        uninterrupted / "aggregate.json"
    ).read_bytes()


def _assert_real_signal_pause_resume(
    tmp_path: Path, *, signal_name: str, parallel: bool
) -> None:
    worker = Path(__file__).parent / "fixtures" / "brainiac_transaction_worker.py"
    suffix = f"{signal_name.casefold()}-{'parallel' if parallel else 'sequential'}"
    uninterrupted = tmp_path / f"uninterrupted-{suffix}"
    resumed = tmp_path / f"resumed-{suffix}"
    marker = tmp_path / f"{suffix}.marker"
    base_environment = dict(os.environ)
    if parallel:
        base_environment["BRAINIAC_FIXTURE_PARALLEL"] = "1"

    clean = subprocess.run(
        [sys.executable, str(worker), str(uninterrupted)],
        cwd=runner.ROOT,
        env=base_environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert clean.returncode == 0, clean.stderr

    signal_environment = dict(base_environment)
    signal_environment["BRAINIAC_FIXTURE_SIGNAL"] = signal_name
    signal_environment["BRAINIAC_FIXTURE_SLEEP_AFTER_MARKER"] = "1"
    paused = subprocess.run(
        [sys.executable, str(worker), str(resumed), str(marker)],
        cwd=runner.ROOT,
        env=signal_environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert paused.returncode == 75, (paused.stdout, paused.stderr)
    assert marker.is_file()
    committed_before_resume = list((resumed / "receipts").glob("*.json"))
    assert 1 <= len(committed_before_resume) <= (2 if parallel else 1)

    resumed_run = subprocess.run(
        [sys.executable, str(worker), str(resumed)],
        cwd=runner.ROOT,
        env=base_environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert resumed_run.returncode == 0, resumed_run.stderr
    for role in ("outputs", "transforms", "receipts"):
        expected = {
            path.name: path.read_bytes() for path in (uninterrupted / role).glob("*")
        }
        observed = {
            path.name: path.read_bytes() for path in (resumed / role).glob("*")
        }
        assert observed == expected
    assert (resumed / "aggregate.json").read_bytes() == (
        uninterrupted / "aggregate.json"
    ).read_bytes()


def test_real_subprocess_sigint_sequential_pause_resume(tmp_path: Path) -> None:
    _assert_real_signal_pause_resume(tmp_path, signal_name="SIGINT", parallel=False)


def test_real_subprocess_sigint_parallel_pause_resume(tmp_path: Path) -> None:
    _assert_real_signal_pause_resume(tmp_path, signal_name="SIGINT", parallel=True)


def test_real_subprocess_sigterm_sequential_pause_resume(tmp_path: Path) -> None:
    _assert_real_signal_pause_resume(tmp_path, signal_name="SIGTERM", parallel=False)


def test_real_subprocess_sigterm_parallel_pause_resume(tmp_path: Path) -> None:
    _assert_real_signal_pause_resume(tmp_path, signal_name="SIGTERM", parallel=True)


def test_sample_lock_pid_reuse_is_rejected_without_archiving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = "s_" + "9" * 16
    lock = tmp_path / "journals" / sample / ".sample.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps({"pid": 42, "process_create_time": "old-generation"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(v2, "_pid_identity", lambda _pid: (True, "new-generation"))
    with pytest.raises(RuntimeError, match="PID_REUSE"):
        v2._reconcile_stale_sample_lock(lock, root=tmp_path, sample_hash=sample)
    assert lock.is_file()


def test_corrupt_transform_is_rejected_from_committed_receipt(tmp_path: Path) -> None:
    item = {"sample_hash": "s_" + "8" * 16, "t1_file_hash": "t1"}
    receipt = v2.run_sample_transaction(
        item, tmp_path, pipeline=_Pipeline(), bindings=_bindings()
    )
    (tmp_path / receipt["transform_relative_path"]).write_bytes(b"corrupt-transform")
    with pytest.raises(RuntimeError, match="COMMITTED_MISMATCH"):
        v2.load_committed_receipt(
            item["sample_hash"], tmp_path, expected_bindings=_bindings()
        )


def test_legacy_generation_is_moved_with_byte_verified_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "old.bin").write_bytes(b"old-calibration-evidence")
    smoke = tmp_path / "smoke"
    old_seq = smoke / "sequential"
    old_seq.mkdir(parents=True)
    (old_seq / "old-output.bin").write_bytes(b"old-smoke-evidence")
    registry = tmp_path / "registry"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "REGISTRY_ROOT", registry)
    monkeypatch.setattr(runner, "CALIBRATION_ROOT", calibration)
    monkeypatch.setattr(runner, "EXECUTION_PROFILE_PATH", registry / "profile.json")
    monkeypatch.setattr(v2, "SMOKE_ROOT", smoke)

    receipt = runner._archive_legacy_incompatible_artifacts()
    assert receipt is not None
    assert receipt["deleted"] is False
    assert receipt["artifact_count"] == 2
    assert not calibration.exists()
    assert not old_seq.exists()
    archives = list(
        (tmp_path / "data/derived/method_x_brainiac/superseded").rglob("old.bin")
    )
    assert len(archives) == 1
    assert archives[0].read_bytes() == b"old-calibration-evidence"
    assert Path(receipt["receipt_path"]).is_file()


def test_current_equivalence_tree_with_stale_contract_is_archived_as_one_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = tmp_path / "smoke"
    leg = smoke / "equivalence_v2" / "sequential"
    receipts = leg / "receipts"
    outputs = leg / "outputs"
    transforms = leg / "transforms"
    receipts.mkdir(parents=True)
    outputs.mkdir()
    transforms.mkdir()
    sample_hash = "s_" + "e" * 16
    (outputs / f"{sample_hash}.nii.gz").write_bytes(b"old-output")
    (transforms / f"{sample_hash}.tfm").write_bytes(b"old-transform")
    (receipts / f"{sample_hash}.json").write_text(
        json.dumps({"policy_sha256": "old-contract"}), encoding="utf-8"
    )
    registry = tmp_path / "registry"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "REGISTRY_ROOT", registry)
    monkeypatch.setattr(runner, "CALIBRATION_ROOT", tmp_path / "calibration")
    monkeypatch.setattr(runner, "EXECUTION_PROFILE_PATH", registry / "profile.json")
    monkeypatch.setattr(v2, "SMOKE_ROOT", smoke)
    monkeypatch.setattr(runner, "_base_bindings", _bindings)

    receipt = runner._archive_legacy_incompatible_artifacts()

    assert receipt is not None
    assert receipt["artifact_count"] == 1
    assert not (smoke / "equivalence_v2").exists()
    archived = list(
        (tmp_path / "data/derived/method_x_brainiac/superseded").rglob(
            f"{sample_hash}.json"
        )
    )
    assert len(archived) == 1


def test_stale_scratch_is_archived_outside_live_run_with_reason(tmp_path: Path):
    item = {"sample_hash": "s_" + "c" * 16, "t1_file_hash": "t1"}
    run_root = tmp_path / "live_run"
    scratch = run_root / "scratch" / item["sample_hash"]
    scratch.mkdir(parents=True)
    (scratch / "evidence.bin").write_bytes(b"partial")
    v2.run_sample_transaction(
        item, run_root, pipeline=_Pipeline(), bindings=_bindings()
    )
    archive_root = tmp_path / "superseded_stale_scratch"
    archived = list(archive_root.rglob("evidence.bin"))
    reasons = list(archive_root.rglob("reason.json"))
    assert len(archived) == 1
    assert not list((run_root / "quarantine").rglob("*"))
    assert not archived[0].is_relative_to(run_root)
    assert any(
        (
            json.loads(path.read_text(encoding="utf-8"))["reason_code"]
            == "STALE_SCRATCH_ARCHIVED_OUTSIDE_LIVE_RUN"
        )
        for path in reasons
    )


def test_execution_profile_requires_three_complete_12_sample_legs():
    sample_hashes = [f"s_{i:064x}" for i in range(12)]
    contract = _sha("contract")

    profile = runner.build_execution_profile(
        sequential=_profile_leg(
            sample_hashes, role="sequential", seconds=120.0, contract=contract
        ),
        parallel_runs=[
            _profile_leg(sample_hashes, role="parallel_1", seconds=80.0, contract=contract),
            _profile_leg(sample_hashes, role="parallel_2", seconds=82.0, contract=contract),
        ],
        expected_contract_sha256=contract,
    )
    assert profile["status"] == "PASS"
    assert profile["execution_mode"] == "PARALLEL_SAFE"

    too_short = _profile_leg(
        sample_hashes, role="sequential", seconds=10.0, contract=contract
    )
    too_short["sample_hashes"] = sample_hashes[:2]
    with pytest.raises(RuntimeError, match="EQUIVALENCE_SAMPLE_COUNT_INVALID"):
        runner.build_execution_profile(
            sequential=too_short,
            parallel_runs=[
                _profile_leg(sample_hashes, role="parallel_1", seconds=8.0, contract=contract),
                _profile_leg(sample_hashes, role="parallel_2", seconds=8.0, contract=contract),
            ],
            expected_contract_sha256=contract,
        )


def test_execution_profile_missing_fresh_run_is_rejected() -> None:
    samples = [f"s_{i:064x}" for i in range(12)]
    contract = _sha("contract")
    missing = _profile_leg(
        samples, role="sequential", seconds=100.0, contract=contract
    )
    missing.pop("fresh_run")

    with pytest.raises(RuntimeError, match="EQUIVALENCE_FRESH_RUN_EVIDENCE_MISSING"):
        runner.build_execution_profile(
            sequential=missing,
            parallel_runs=[
                _profile_leg(samples, role="parallel_1", seconds=70.0, contract=contract),
                _profile_leg(samples, role="parallel_2", seconds=72.0, contract=contract),
            ],
            expected_contract_sha256=contract,
        )

    profile = runner.build_execution_profile(
        sequential=_profile_leg(samples, role="sequential", seconds=100.0, contract=contract),
        parallel_runs=[
            _profile_leg(
                samples, role="parallel_1", seconds=70.0, contract=contract, fresh=False
            ),
            _profile_leg(samples, role="parallel_2", seconds=72.0, contract=contract),
        ],
        expected_contract_sha256=contract,
    )
    assert profile["execution_mode"] == "SEQUENTIAL_SAFE"
    assert profile["timing_fresh"] is False


def test_execution_profile_rejects_self_asserted_legs_without_live_root_evidence() -> None:
    samples = [f"s_{i:064x}" for i in range(12)]
    leg = {
        "sample_hashes": samples,
        "receipt_digests": {sample: _sha(sample) for sample in samples},
        "scientific_digests": {sample: _sha("sci-" + sample) for sample in samples},
        "elapsed_seconds": 100.0,
        "contract_sha256": _sha("contract"),
        "integrity_status": "PASS",
        "fresh_run": True,
    }

    with pytest.raises(RuntimeError, match="EQUIVALENCE_LIVE_EVIDENCE_MISSING"):
        runner.build_execution_profile(
            sequential=leg,
            parallel_runs=[dict(leg), dict(leg)],
            expected_contract_sha256=_sha("contract"),
        )


def test_execution_profile_recomputes_live_receipts_and_rejects_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = tmp_path / "smoke"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(v2, "SMOKE_ROOT", smoke)
    bindings = _bindings()
    contract = runner.canonical_sha(runner._threshold_contract_bindings(bindings))
    samples = [f"s_{index:064x}" for index in range(12)]
    roots = runner._execution_profile_roots()
    elapsed = {"sequential": 120.0, "parallel_1": 80.0, "parallel_2": 82.0}

    for role, root in roots.items():
        receipts = [
            v2.run_sample_transaction(
                {"sample_hash": sample, "t1_file_hash": _sha("t1-" + sample)},
                root,
                pipeline=_Pipeline(),
                bindings=bindings,
            )
            for sample in samples
        ]
        runner.write_execution_leg_timing_receipt(
            root,
            role=role,
            sample_hashes=sorted(str(row["sample_hash"]) for row in receipts),
            contract_sha256=contract,
            started_at_utc="2026-08-28T00:00:00+00:00",
            finished_at_utc="2026-08-28T00:02:00+00:00",
            elapsed_seconds=elapsed[role],
            fresh_run=True,
            resume_event_count=0,
        )

    live = {
        role: runner.recompute_execution_leg(
            role=role,
            expected_contract_sha256=contract,
            expected_receipt_bindings=bindings,
        )
        for role in ("sequential", "parallel_1", "parallel_2")
    }
    profile = runner.build_execution_profile(
        sequential=live["sequential"],
        parallel_runs=[live["parallel_1"], live["parallel_2"]],
        expected_contract_sha256=contract,
    )
    assert profile["execution_mode"] == "PARALLEL_SAFE"
    profile.update(
        {
            "created_at_utc": "2026-08-28T00:03:00+00:00",
            "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        }
    )
    profile["integrity_sha256"] = runner.canonical_sha(
        {key: value for key, value in profile.items() if key != "integrity_sha256"}
    )
    runner.validate_execution_profile(
        profile,
        expected_contract_sha256=contract,
        expected_receipt_bindings=bindings,
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(runner, "EXECUTION_PROFILE_PATH", profile_path)
    monkeypatch.setattr(runner, "_base_bindings", lambda: bindings)
    monkeypatch.setattr(runner, "_current_equivalence_contract_sha256", lambda: contract)
    assert runner._parallel_unlock_state() == (False, True)

    receipt = json.loads(
        next((roots["parallel_2"] / "receipts").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    output = roots["parallel_2"] / receipt["output_relative_path"]
    output.write_bytes(output.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="OUTPUT_BINDING_MISMATCH"):
        runner.validate_execution_profile(
            profile,
            expected_contract_sha256=contract,
            expected_receipt_bindings=bindings,
        )


def test_readiness_builder_requires_complete_evidence_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    primary = tmp_path / "primary"
    primary.mkdir()
    monkeypatch.setattr(v2, "PRIMARY_ROOT", primary)
    monkeypatch.setattr(
        v2,
        "load_locked_membership",
        lambda: [
            {"sample_hash": f"s_{index:064x}", "t1_file_hash": _sha(str(index))}
            for index in range(1669)
        ],
    )
    bindings = _bindings()
    monkeypatch.setattr(runner, "_base_bindings", lambda: bindings)
    monkeypatch.setattr(
        runner,
        "_current_equivalence_contract_sha256",
        lambda: _sha("profile-contract"),
    )
    profile = {"execution_mode": "SEQUENTIAL_SAFE", "integrity_sha256": _sha("profile")}
    monkeypatch.setattr(runner, "validate_execution_profile", lambda *args, **kwargs: profile)
    test_evidence = {"integrity_sha256": _sha("tests")}
    monkeypatch.setattr(runner, "_validated_acceptance_test_manifest", lambda: test_evidence)
    monkeypatch.setattr(
        runner,
        "_acceptance_scoped_git_state",
        lambda: ("branch", "a" * 40, []),
    )
    dependency_sha = _sha("dependency-closure")
    monkeypatch.setattr(
        runner,
        "_dependency_manifest",
        lambda: {"closure_sha256": dependency_sha},
    )
    preflight = {
        "status": "PASS",
        "dependency_closure_sha256": dependency_sha,
        "dependency_untracked_count": 0,
        "method_x_source_digest": runner.EXPECTED_METHOD_X_SOURCE_DIGEST,
        "battery_manifest_sha256": runner.EXPECTED_BATTERY_MANIFEST_SHA256,
    }
    preflight["integrity_sha256"] = runner.canonical_sha(preflight)

    path_names = {
        "PREFLIGHT_REPORT": "preflight.json",
        "DEPENDENCY_MANIFEST_PATH": "dependencies.json",
        "EXECUTION_PROFILE_PATH": "profile.json",
        "THRESHOLD_MANIFEST_PATH": "thresholds.json",
        "SMOKE_REPORT": "smoke.json",
        "ACCEPTANCE_TEST_MANIFEST_PATH": "tests.json",
        "PRE_REPAIR_LIVE_FREEZE_PATH": "freeze.json",
        "ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH": "preflight-snapshot.json",
        "ACCEPTANCE_CANDIDATE_PATH": "candidate.json",
    }
    for attribute, name in path_names.items():
        path = registry / name
        monkeypatch.setattr(runner, attribute, path)
        if attribute not in {
            "ACCEPTANCE_CANDIDATE_PATH",
            "ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH",
        }:
            path.write_text("{}", encoding="utf-8")

    ready = runner.build_preprocessing_readiness_candidate(
        preflight_report=preflight,
        execution_profile=profile,
        calibration_report={"status": "PASS", "selected_n": 32},
        smoke_report={
            "status": "PASS",
            "rows": 12,
            "physical_qc_validated_from_live_receipts": True,
        },
    )
    assert ready["status"] == "ACCEPTANCE_CANDIDATE_PASS"
    assert ready["readiness_candidate"] is True
    assert ready["completion"] is False
    assert ready["training_authorization"] is False
    first = runner.ACCEPTANCE_CANDIDATE_PATH.read_bytes()
    repeated = runner.build_preprocessing_readiness_candidate(
        preflight_report=preflight,
        execution_profile=profile,
        calibration_report={"status": "PASS", "selected_n": 32},
        smoke_report={
            "status": "PASS",
            "rows": 12,
            "physical_qc_validated_from_live_receipts": True,
        },
    )
    assert repeated == ready
    assert runner.ACCEPTANCE_CANDIDATE_PATH.read_bytes() == first


def test_readiness_builder_never_promotes_unknown_or_missing_test_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "primary"
    primary.mkdir()
    monkeypatch.setattr(v2, "PRIMARY_ROOT", primary)
    monkeypatch.setattr(v2, "load_locked_membership", lambda: [])
    monkeypatch.setattr(runner, "_base_bindings", lambda: _bindings())
    monkeypatch.setattr(
        runner,
        "validate_execution_profile",
        lambda *args, **kwargs: {"execution_mode": "SEQUENTIAL_SAFE"},
    )
    monkeypatch.setattr(
        runner,
        "_validated_acceptance_test_manifest",
        lambda: (_ for _ in ()).throw(
            RuntimeError("BRAINIAC_ACCEPTANCE_TEST_MANIFEST_MISSING")
        ),
    )
    threshold = tmp_path / "threshold.json"
    threshold.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "THRESHOLD_MANIFEST_PATH", threshold)

    blocked = runner.build_preprocessing_readiness_candidate(
        preflight_report={"status": "PASS"},
        execution_profile={},
        calibration_report={"status": "PASS", "selected_n": 32},
        smoke_report={
            "status": "PASS",
            "rows": 12,
            "physical_qc_validated_from_live_receipts": True,
        },
    )
    assert blocked["status"] == "BLOCKED"
    assert "ACCEPTANCE_TEST_EVIDENCE_NOT_VALID" in blocked["blocking_reasons"]


def test_acceptance_test_manifest_requires_live_bound_schema2_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file = tmp_path / "test_signal.py"
    source_file = tmp_path / "runner.py"
    junit = tmp_path / "junit.xml"
    test_file.write_text("def test_signal(): pass\n", encoding="utf-8")
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    junit.write_text("<testsuites tests='1' failures='0' errors='0'/>\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "_git_live_state", lambda: ("branch", "a" * 40, [])
    )
    monkeypatch.setattr(
        runner,
        "_dependency_manifest",
        lambda: {"closure_sha256": "c" * 64},
    )
    required = {
        name: {"status": "PASS", "nodeids": [f"tests::{name}"]}
        for name in runner.ACCEPTANCE_REQUIRED_CHECKS
    }
    body = {
        "schema_version": 2,
        "artifact": "BRAINIAC_PREPROCESSING_ACCEPTANCE_TEST_MANIFEST",
        "status": "PASS",
        "created_at_utc": "2026-08-28T00:00:00+00:00",
        "git_head": "a" * 40,
        "dependency_closure_sha256": "c" * 64,
        "exit_code": 0,
        "platform": "Windows",
        "summary": {"passed": len(required), "failed": 0, "errors": 0},
        "required_checks": required,
        "evidence_bindings": {
            "junit": runner._artifact_binding(junit),
            "test_files": [runner._artifact_binding(test_file)],
            "source_files": [runner._artifact_binding(source_file)],
        },
    }
    body["integrity_sha256"] = runner.canonical_sha(body)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(body), encoding="utf-8")
    monkeypatch.setattr(runner, "ACCEPTANCE_TEST_MANIFEST_PATH", manifest)
    assert runner._validated_acceptance_test_manifest()["status"] == "PASS"

    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="BINDING_DRIFT"):
        runner._validated_acceptance_test_manifest()


@pytest.mark.parametrize(
    "checks",
    [
        {"storage": False, "writer": True, "anchor": True},
        {"storage": True, "writer": "UNKNOWN", "anchor": True},
        {"storage": True, "writer": True, "anchor": False},
    ],
)
def test_preflight_decision_fails_closed_on_false_or_unknown(checks):
    report = runner.build_preflight_decision(checks)
    assert report["status"] == "BLOCKED"
    assert report["blocking_checks"]


def test_traceability_revalidates_frozen_battery_before_document_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan 018 must fail before touching prose when battery preservation drifts."""

    from scripts.research import build_brainiac_preprocessing_release as release

    monkeypatch.setattr(
        release.runner,
        "validate_acceptance_candidate",
        lambda: {"integrity_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        release.runner,
        "_validated_full_test_manifest",
        lambda: {"summary": {"passed": 1, "failed": 0, "errors": 0}},
    )

    def sealed(_path: Path, *, expected_status: str) -> dict[str, object]:
        if expected_status == "R0_COMPLETE_NOVELTY_CANDIDATE":
            return {
                "status": expected_status,
                "registry_sha256": (
                    "e397b9dd03653bf7d4c3b80e470b8ffa9389b40e04a2208530bb5eb5c0a7cd8f"
                ),
            }
        return {
            "status": expected_status,
            "brainiac_training_allowed": False,
            "r0": {"novelty_status": "NOVELTY_CANDIDATE"},
        }

    monkeypatch.setattr(release, "_validate_sealed", sealed)
    monkeypatch.setattr(
        release.runner,
        "run_status",
        lambda: {
            "status": "PASS",
            "expected_rows": 1669,
            "completed_rows": 0,
            "invalid_rows": [],
        },
    )
    monkeypatch.setattr(
        release,
        "_preservation_report",
        lambda: (_ for _ in ()).throw(RuntimeError("BATTERY_TREE_DRIFT")),
    )
    monkeypatch.setattr(
        release,
        "_update_human_traceability",
        lambda _facts: (_ for _ in ()).throw(
            AssertionError("traceability documents mutated before preservation")
        ),
    )

    with pytest.raises(RuntimeError, match="BATTERY_TREE_DRIFT"):
        release.build_traceability()


def test_final_readiness_never_accepts_an_acceptance_candidate_as_release(
    tmp_path: Path,
) -> None:
    candidate = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PREPROCESSING_ACCEPTANCE_CANDIDATE",
        "status": "ACCEPTANCE_CANDIDATE_PASS",
        "readiness_candidate": True,
        "completion": False,
        "training_authorization": False,
        "manual_override_allowed": False,
        "human_approval_required": False,
    }
    candidate["integrity_sha256"] = runner.canonical_sha(candidate)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(RuntimeError, match="FINAL_PREPROCESSING_READINESS_INVALID"):
        runner.validate_final_preprocessing_readiness(path)


def test_full_test_evidence_covers_dynamic_brainiac_code_and_contracts() -> None:
    from scripts.research import build_brainiac_preprocessing_full_test_manifest as full

    relative = {path.relative_to(full.ROOT).as_posix() for path in full._scientific_source_paths()}
    required = {
        ".gitattributes",
        "configs/research/brainiac_preprocessing_r1_primary_v2.yaml",
        "governance/brainiac_preprocessing_r1_primary_v2_policy.json",
        "scripts/research/prepare_brainiac_raw_t1_v2.py",
        "scripts/research/train_brainiac_caft_r1.py",
        "src/research/brainiac_preprocessing_v2.py",
        "src/research/brainiac_caft_training.py",
        "src/research/brainiac_adapter.py",
        "tests/fixtures/brainiac_transaction_worker.py",
    }
    assert required <= relative


def test_exception_sanitizer_removes_raw_path_and_sample_hash():
    raw = r"H:\A4 Learn\subject-x\T1.nii.gz"
    sample_hash = "s_1234567890abcdef"
    cleaned = runner._sanitize_exception_message(
        RuntimeError(f"failed {raw} for {sample_hash}")
    )
    assert "H:\\A4 Learn" not in cleaned
    assert sample_hash not in cleaned
    assert "<raw-redacted>" in cleaned


def test_proven_dead_shared_guard_is_archived_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.research import brainiac_action_guard as guard

    battery = tmp_path / "battery"
    battery.mkdir()
    lock = battery / guard.SHARED_GUARD_NAME
    lock.write_text(
        json.dumps(
            {
                "pid": 999999,
                "process_create_time": "dead",
                "nonce": "n",
                "ownership_token": "t",
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry"
    monkeypatch.setattr(guard, "BATTERY_ROOT", battery)
    monkeypatch.setattr(runner, "REGISTRY_ROOT", registry)
    monkeypatch.setattr(
        guard,
        "_recorded_lock_status",
        lambda _path, *, role: {
            "role": role,
            "present": True,
            "liveness": "DEAD",
            "pid": 999999,
        },
    )
    receipt = runner._archive_proven_stale_shared_guard()
    assert receipt["status"] == "ARCHIVED_PROVEN_DEAD"
    assert not lock.exists()
    assert Path(receipt["archive_path"]).is_file()
    assert Path(receipt["receipt_path"]).is_file()


def test_stale_shared_guard_archive_accepts_test_only_writer_providers() -> None:
    """Archive safety must be testable without consulting live workstation state."""

    import inspect

    parameters = inspect.signature(
        runner._archive_proven_stale_shared_guard
    ).parameters
    assert "process_provider" in parameters
    assert "gpu_provider" in parameters


def test_stale_shared_guard_is_not_archived_while_method_y_writer_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead lock must remain untouched if another project writer is active."""

    from src.research import brainiac_action_guard as guard

    battery = tmp_path / "battery"
    battery.mkdir()
    lock = battery / guard.SHARED_GUARD_NAME
    lock.write_text(
        json.dumps(
            {
                "pid": 999999,
                "process_create_time": "dead-created",
                "nonce": "n",
                "ownership_token": "t",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(guard, "BATTERY_ROOT", battery)
    monkeypatch.setattr(runner, "REGISTRY_ROOT", tmp_path / "registry")
    monkeypatch.setattr(
        guard,
        "_recorded_lock_status",
        lambda _path, *, role: {
            "role": role,
            "present": True,
            "liveness": "DEAD",
            "pid": 999999,
        },
    )
    method_y = {
        "ProcessId": 43216,
        "Name": "python.exe",
        "CommandLine": r"python.exe scripts\run_method_y_image_training_v3.py --resume",
        "CreationDate": "20260828070500.000000+210",
    }

    with pytest.raises(
        RuntimeError,
        match="BRAINIAC_STALE_GUARD_ARCHIVE_BLOCKED:ACTIVE_BRAINIAC_OR_CNN_WRITER",
    ):
        runner._archive_proven_stale_shared_guard(
            process_provider=lambda: [method_y],
            gpu_provider=lambda: [],
        )
    assert lock.is_file()
    assert not (tmp_path / "registry").exists()


def test_pre_repair_freeze_digest_excludes_operational_timestamp() -> None:
    """Rebuilding unchanged evidence at a later time must keep its snapshot digest."""

    snapshot = {
        "branch": "test",
        "head": "a" * 40,
        "battery_inventory": [{"path": "x", "bytes": 1, "sha256": "b" * 64}],
    }
    first = runner.build_pre_repair_freeze_artifact(
        snapshot=snapshot,
        created_at_utc="2026-08-28T00:00:00+00:00",
    )
    second = runner.build_pre_repair_freeze_artifact(
        snapshot=snapshot,
        created_at_utc="2026-08-28T00:01:00+00:00",
    )

    assert first["snapshot_sha256"] == second["snapshot_sha256"]
    assert first["integrity_sha256"] != second["integrity_sha256"]
    assert first["snapshot"] == snapshot


def test_live_git_state_helper_executes_real_git_without_name_error() -> None:
    """The production freeze must be able to read the live repository state."""

    branch, head, dirty = runner._git_live_state()

    assert branch
    assert len(head) == 40
    assert all(character in "0123456789abcdef" for character in head.lower())
    assert isinstance(dirty, list)


def test_authoritative_freeze_does_not_overwrite_agent_schema1_predecessor() -> None:
    assert runner.PRE_REPAIR_LIVE_FREEZE_PATH.name == (
        "BRAINIAC_PRE_REPAIR_LIVE_FREEZE_SCHEMA2_20260828.json"
    )


def test_freeze_inventory_rejects_non_mapping_rows_fail_closed() -> None:
    malformed = {
        "files": ["not-a-row"],
        "file_count": 1,
        "total_bytes": 0,
        "inventory_sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="BRAINIAC_FREEZE_INVENTORY_INVALID"):
        runner._validate_freeze_inventory("malformed", malformed)


def test_freeze_dirty_filter_excludes_only_untracked_operational_churn() -> None:
    entries = [
        "?? .runtime/tmp/session.json",
        "?? tmp/pytest-output.bin",
        "?? .tmp_probe/generated.json",
        "?? .claude/session.json",
        "?? .agents/cache.json",
        "?? .opencode/state.json",
        "?? src/research/new_runtime.py",
        "?? advisor-plans/new-card.md",
        " M .runtime/tracked_policy.json",
        " M scripts/research/prepare_brainiac_raw_t1_v2.py",
    ]

    assert runner._filter_freeze_dirty_entries(entries) == sorted(
        [
            "?? src/research/new_runtime.py",
            "?? advisor-plans/new-card.md",
            " M .runtime/tracked_policy.json",
            " M scripts/research/prepare_brainiac_raw_t1_v2.py",
        ]
    )


def test_non_idempotent_freeze_is_archived_append_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "freeze.json"
    archive = tmp_path / "superseded"
    payload = runner.build_pre_repair_freeze_artifact(
        snapshot={"stable": False},
        created_at_utc="2026-08-28T00:00:00+00:00",
    )
    runner._atomic_write_json(source, payload)
    monkeypatch.setattr(
        runner,
        "BATTERY_MANIFEST_PATH",
        tmp_path / "battery_manifest.json",
    )
    runner._atomic_write_json(runner.BATTERY_MANIFEST_PATH, {"models": []})

    receipt = runner.archive_non_idempotent_pre_repair_freeze(
        source,
        archive_root=archive,
        process_provider=lambda: [],
        gpu_provider=lambda: [],
        created_at_utc="2026-08-28T00:01:00+00:00",
    )

    assert receipt["status"] == "ARCHIVED_APPEND_ONLY"
    assert not source.exists()
    target = archive / receipt["archive_name"]
    assert target.read_bytes()
    assert target.with_suffix(".receipt.json").is_file()


def test_pre_repair_freeze_is_append_only_and_idempotent(
    tmp_path: Path,
) -> None:
    """A changed snapshot must never overwrite the preservation authority."""

    target = tmp_path / "freeze.json"
    snapshot = {"head": "a" * 40, "source_digest": "b" * 64}
    first = runner.write_pre_repair_freeze(
        target,
        snapshot=snapshot,
        created_at_utc="2026-08-28T00:00:00+00:00",
    )
    before = target.read_bytes()
    second = runner.write_pre_repair_freeze(
        target,
        snapshot=snapshot,
        created_at_utc="2026-08-28T01:00:00+00:00",
    )
    assert second == first
    assert target.read_bytes() == before

    with pytest.raises(RuntimeError, match="BRAINIAC_PRE_REPAIR_FREEZE_DRIFT"):
        runner.write_pre_repair_freeze(
            target,
            snapshot={**snapshot, "source_digest": "c" * 64},
            created_at_utc="2026-08-28T02:00:00+00:00",
        )
    assert target.read_bytes() == before


def test_file_inventory_binds_relative_path_size_and_bytes(tmp_path: Path) -> None:
    """Changing a preserved artifact's bytes must change the inventory authority."""

    first = tmp_path / "z.bin"
    second = tmp_path / "a.bin"
    first.write_bytes(b"z")
    second.write_bytes(b"alpha")

    inventory = runner.build_file_inventory(
        [first, second], project_root=tmp_path
    )

    expected_files = [
        {"path": "a.bin", "bytes": 5, "sha256": hashlib.sha256(b"alpha").hexdigest()},
        {"path": "z.bin", "bytes": 1, "sha256": hashlib.sha256(b"z").hexdigest()},
    ]
    assert inventory["files"] == expected_files
    assert inventory["file_count"] == 2
    assert inventory["total_bytes"] == 6
    expected_digest = hashlib.sha256(
        json.dumps(expected_files, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert inventory["inventory_sha256"] == expected_digest


def test_pre_repair_snapshot_binds_all_preservation_authorities() -> None:
    """Omitting any named authority must prevent creation of a usable freeze."""

    battery_binding = {
        "path": "results/battery_manifest.json",
        "bytes": 10,
        "sha256": runner.EXPECTED_BATTERY_MANIFEST_SHA256,
    }
    inventory_files = [
        {"path": "models/final.pt", "bytes": 1, "sha256": "c" * 64}
    ]
    inventory = {
        "files": inventory_files,
        "file_count": 1,
        "total_bytes": 1,
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                inventory_files, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    snapshot = runner.build_pre_repair_snapshot(
        branch="branch",
        head="a" * 40,
        dirty_paths=[" M source.py", "?? new.py"],
        writer_state={
            "inspection_status": "PASS",
            "active_writers": [],
            "foreign_gpu_processes": [],
            "ambient_gpu_processes": [],
            "recorded_owners": [],
        },
        method_x_source_digest=runner.EXPECTED_METHOD_X_SOURCE_DIGEST,
        battery_manifest_binding=battery_binding,
        battery_inventory=inventory,
        c_checkpoint_inventory={
            "files": [],
            "file_count": 0,
            "total_bytes": 0,
            "inventory_sha256": hashlib.sha256(b"[]").hexdigest(),
        },
        dependency_manifest={"closure_sha256": "e" * 64},
        lock_inventory=[],
        legacy_inventory=[],
        primary_valid_n=0,
    )

    assert snapshot["method_x_source_digest"] == runner.EXPECTED_METHOD_X_SOURCE_DIGEST
    assert snapshot["battery_manifest"] == battery_binding
    assert snapshot["battery_finalized_inventory"] == inventory
    assert snapshot["primary_valid_n"] == 0
    assert snapshot["dirty_inventory_sha256"] == hashlib.sha256(
        json.dumps(
            [" M source.py", "?? new.py"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(RuntimeError, match="BRAINIAC_FREEZE_SOURCE_DIGEST_DRIFT"):
        runner.build_pre_repair_snapshot(
            branch="branch",
            head="a" * 40,
            dirty_paths=[],
            writer_state=snapshot["writer_state"],
            method_x_source_digest="0" * 64,
            battery_manifest_binding=battery_binding,
            battery_inventory=inventory,
            c_checkpoint_inventory=inventory,
            dependency_manifest={"closure_sha256": "e" * 64},
            lock_inventory=[],
            legacy_inventory=[],
            primary_valid_n=0,
        )
