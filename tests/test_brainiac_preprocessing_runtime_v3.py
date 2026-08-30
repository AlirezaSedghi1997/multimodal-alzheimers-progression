"""Runtime-v3 acceptance tests for fast, deterministic BrainIAC preprocessing.

Only synthetic byte payloads are used.  The tests prove that local NVMe
scratch never becomes the authoritative artifact store and that registration
cannot race while full-resolution N4 is allowed to use the locked thread
budget.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest


def _bindings() -> dict[str, str]:
    return {
        "policy_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "pipeline_contract_sha256": "3" * 64,
        "dependency_closure_sha256": "4" * 64,
        "membership_sha256": "5" * 64,
        "cache_sha256": "6" * 64,
        "catalog_sha256": "7" * 64,
        "runtime_lock_sha256": "8" * 64,
        "vendor_tree_sha256": "9" * 64,
        "template_lock_sha256": "a" * 64,
        "official_transform_source_sha256": "b" * 64,
        "qc_threshold_manifest_sha256": "c" * 64,
    }


class _Pipeline:
    def __init__(self) -> None:
        self.seen_scratch: Path | None = None

    def __call__(self, _item, scratch: Path):
        self.seen_scratch = scratch
        output = scratch / "output.nii.gz"
        transform = scratch / "transform.tfm"
        output.write_bytes(b"synthetic-nifti")
        transform.write_bytes(b"synthetic-transform")
        identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        return {
            "output_path": output,
            "transform_path": transform,
            "source_sha256": hashlib.sha256(b"source").hexdigest(),
            "source_size": 6,
            "affine_before": identity,
            "affine_after": identity,
            "spacing_before": [1.0, 1.0, 1.0],
            "spacing_after": [1.0, 1.0, 1.0],
            "origin_before": [0.0, 0.0, 0.0],
            "origin_after": [0.0, 0.0, 0.0],
            "direction_before": direction,
            "direction_after": direction,
            "com_coordinate_frame": "TEMPLATE_PHYSICAL_MM_RELATIVE",
            "rigid_determinant": 1.0,
            "orthogonality_error": 0.0,
            "rotation_deg": 0.0,
            "translation_norm_mm": 0.0,
            "foreground_fraction": 0.25,
            "finite": True,
            "normalized_nonzero_mean": 0.0,
            "normalized_nonzero_std": 1.0,
            "center_of_mass_before": [0.0, 0.0, 0.0],
            "center_of_mass_after": [0.0, 0.0, 0.0],
            "registration_similarity_before": -1.0,
            "registration_similarity_after": -0.5,
            "runtime_stage_seconds": {
                "n4": 1.0,
                "registration_serial": 0.5,
                "hd_bet_serial": 0.2,
                "total": 1.7,
            },
        }


@pytest.fixture(autouse=True)
def _no_live_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import brainiac_preprocessing_v2 as v2

    monkeypatch.setattr(v2, "assert_no_cnn_writer", lambda: None)


def test_policy_v4_locks_fast_profile_and_deterministic_registration_repair() -> None:
    import json

    from src.research import brainiac_preprocessing_v2 as v2

    policy = json.loads(v2.POLICY_PATH.read_text(encoding="utf-8"))
    assert policy["schema_version"] == 3
    assert policy["orchestration"]["itk_threads_default"] == 8
    assert policy["orchestration"]["n4_samples_in_flight"] == 2
    assert policy["orchestration"]["registration_workers_max"] == 1
    assert policy["orchestration"]["gpu_workers_max"] == 1
    registration = policy["registration_contract"]
    assert registration["sampling_strategy"] == "NONE_FULL"
    assert registration["registration_threads"] == 1
    assert registration["registration_work_units"] == 1
    assert registration["parameter_quantization_decimals"] == 4
    assert registration["metric_quantization_decimals"] == 7
    assert registration["quality_selection_or_fallback"] is False
    storage = policy["storage_locality"]
    assert storage["authoritative_artifacts"] == "H_ONLY"
    assert storage["local_scratch_max_bytes"] == 1_073_741_824
    assert storage["local_scratch_minimum_free_bytes"] == 21_474_836_480
    assert storage["silent_fallback"] is False


def test_registration_stage_is_serialized() -> None:
    import scripts.research.prepare_brainiac_raw_t1_v2 as prep

    active = 0
    maximum = 0
    state_lock = threading.Lock()

    def operation():
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _index: prep._serialized_registration_stage(operation), range(2))) == ["ok", "ok"]
    assert maximum == 1


def test_local_scratch_is_hashed_bounded_and_not_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research import brainiac_preprocessing_v2 as v2

    authoritative = tmp_path / "h" / "run"
    local = tmp_path / "c" / "staging"
    local.mkdir(parents=True)
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_ROOT", local)
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_MIN_FREE_BYTES", 0)
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_MAX_BYTES", 1024)
    scratch = v2.resolve_sample_scratch(
        authoritative, "s_" + "1" * 16, scratch_base=local
    )
    assert scratch.is_relative_to(local)
    assert authoritative.name not in scratch.parts
    (local / "occupied.bin").write_bytes(b"x" * 1025)
    with pytest.raises(RuntimeError, match="LOCAL_SCRATCH_CAP_EXCEEDED"):
        v2.resolve_sample_scratch(
            authoritative, "s_" + "2" * 16, scratch_base=local
        )


def test_transaction_stages_locally_but_commits_and_journals_authoritatively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research import brainiac_preprocessing_v2 as v2

    root = tmp_path / "h" / "run"
    local = tmp_path / "c" / "staging"
    local.mkdir(parents=True)
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_ROOT", local)
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_MIN_FREE_BYTES", 0)
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_MAX_BYTES", 1024 * 1024)
    pipeline = _Pipeline()
    item = {"sample_hash": "s_" + "3" * 16, "t1_file_hash": "t1"}
    receipt = v2.run_sample_transaction(
        item,
        root,
        pipeline=pipeline,
        bindings=_bindings(),
        scratch_base=local,
    )
    assert pipeline.seen_scratch is not None
    assert pipeline.seen_scratch.is_relative_to(local)
    assert not pipeline.seen_scratch.exists()
    assert (root / receipt["output_relative_path"]).read_bytes() == b"synthetic-nifti"
    assert (root / receipt["transform_relative_path"]).read_bytes() == b"synthetic-transform"
    journal = root / "journals" / item["sample_hash"]
    assert (journal / "committed.json").is_file()
    assert (journal / "runtime.json").is_file()
    assert not list(local.rglob("*.json"))


def test_cross_volume_commit_is_copy_verified_then_source_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research import brainiac_preprocessing_v2 as v2

    source = tmp_path / "c" / "payload.bin"
    target = tmp_path / "h" / "payload.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"payload")
    monkeypatch.setattr(v2, "_same_volume", lambda *_args: False)
    v2._commit_staged_file(source, target)
    assert target.read_bytes() == b"payload"
    assert not source.exists()
    assert not list(target.parent.glob("*.tmp-*"))


def test_cross_volume_stale_scratch_is_archived_before_local_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research import brainiac_preprocessing_v2 as v2

    root = tmp_path / "h" / "run"
    scratch = tmp_path / "c" / "stage" / ("s_" + "4" * 16)
    scratch.mkdir(parents=True)
    (scratch / "registered.nii").write_bytes(b"registered")
    monkeypatch.setattr(v2, "_same_volume", lambda *_args: False)
    archived = v2._archive_stale_scratch(
        scratch, root, "s_" + "4" * 16
    )
    assert archived is not None and archived.is_dir()
    assert (archived / "registered.nii").read_bytes() == b"registered"
    assert (archived / "reason.json").is_file()
    assert not scratch.exists()


def test_parallel_batch_propagates_local_scratch_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research import brainiac_preprocessing_v2 as v2

    seen: list[Path | None] = []

    def fake_transaction(_item, _root, *, scratch_base=None, **_kwargs):
        seen.append(scratch_base)
        return {"sample_hash": _item["sample_hash"]}

    monkeypatch.setattr(v2, "run_sample_transaction", fake_transaction)
    items = [{"sample_hash": f"s_{index:016x}"} for index in range(4)]
    local = tmp_path / "local"
    report = v2.run_orchestrated_batch(
        items,
        tmp_path / "root",
        pipeline_factory=lambda: None,
        bindings={},
        cpu_workers=2,
        gpu_workers=1,
        scratch_base=local,
    )
    assert report["status"] == "COMPLETED"
    assert seen == [local] * 4


def test_local_storage_failure_releases_authoritative_sample_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research import brainiac_preprocessing_v2 as v2

    root = tmp_path / "h" / "run"
    item = {"sample_hash": "s_" + "5" * 16, "t1_file_hash": "t1"}
    monkeypatch.setattr(
        v2,
        "resolve_sample_scratch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("BRAINIAC_LOCAL_SCRATCH_FREE_SPACE_INSUFFICIENT")
        ),
    )
    with pytest.raises(RuntimeError, match="FREE_SPACE_INSUFFICIENT"):
        v2.run_sample_transaction(
            item,
            root,
            pipeline=_Pipeline(),
            bindings=_bindings(),
            scratch_base=tmp_path / "c",
        )
    assert not (root / "journals" / item["sample_hash"] / ".sample.lock").exists()


def test_local_scratch_cap_is_rechecked_after_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.research import brainiac_preprocessing_v2 as v2

    root = tmp_path / "h" / "run"
    local = tmp_path / "c" / "staging"
    local.mkdir(parents=True)
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_ROOT", local)
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_MIN_FREE_BYTES", 0)
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_MAX_BYTES", 30)
    item = {"sample_hash": "s_" + "6" * 16, "t1_file_hash": "t1"}
    with pytest.raises(RuntimeError, match="LOCAL_SCRATCH_CAP_EXCEEDED"):
        v2.run_sample_transaction(
            item,
            root,
            pipeline=_Pipeline(),
            bindings=_bindings(),
            scratch_base=local,
        )
    assert not (root / "journals" / item["sample_hash"] / ".sample.lock").exists()
    assert not list((root / "receipts").glob("*.json"))


def test_acceptance_manifest_supersession_is_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import (
        build_brainiac_preprocessing_acceptance_test_manifest as builder,
    )

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder.runner, "ROOT", tmp_path)
    target = tmp_path / "registry" / "acceptance.json"
    target.parent.mkdir(parents=True)
    existing = {
        "schema_version": 2,
        "artifact": "BRAINIAC_PREPROCESSING_ACCEPTANCE_TEST_MANIFEST",
        "status": "PASS",
        "created_at_utc": "2026-08-28T00:00:00+00:00",
        "git_head": "a" * 40,
    }
    existing["integrity_sha256"] = builder.runner.canonical_sha(existing)
    candidate = dict(existing)
    candidate["git_head"] = "b" * 40
    candidate.pop("integrity_sha256")
    candidate["integrity_sha256"] = builder.runner.canonical_sha(candidate)
    builder.runner._atomic_write_json(target, existing)

    archived = builder._archive_superseded_manifest(target, existing, candidate)
    assert not target.exists()
    assert archived.is_file()
    assert json.loads(archived.read_text(encoding="utf-8")) == existing
    receipts = list(archived.parent.glob("*.supersession.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["superseded_integrity_sha256"] == existing["integrity_sha256"]
    assert receipt["replacement_integrity_sha256"] == candidate["integrity_sha256"]


def test_rigid_transform_canonicalization_removes_subprecision_optimizer_jitter():
    """Sub-voxel optimizer jitter must not change the locked transform bytes."""

    from scripts.research import prepare_brainiac_raw_t1_v2 as prep

    canonical_first = prep._canonical_rigid_parameters(
        (-0.1338013622, -0.0072943913, 0.0693304712, 1.6131162, -0.3580641, -56.9335221)
    )
    canonical_second = prep._canonical_rigid_parameters(
        (-0.1338013608, -0.0072943903, 0.0693304723, 1.6131155, -0.3580635, -56.9335214)
    )

    assert canonical_first == canonical_second
    assert canonical_first == (
        -0.1338,
        -0.0073,
        0.0693,
        1.6131,
        -0.3581,
        -56.9335,
    )


def test_registration_source_uses_full_sampling_and_per_object_single_thread():
    """A seed on RANDOM sampling was empirically non-deterministic on Windows."""

    import inspect
    from scripts.research import prepare_brainiac_raw_t1_v2 as prep

    registration_source = inspect.getsource(prep._registration)
    metric_source = inspect.getsource(prep._metric_value)
    for source in (registration_source, metric_source):
        assert "SetMetricSamplingStrategy(method.NONE)" in source
        assert "SetNumberOfThreads(1)" in source
        assert "SetNumberOfWorkUnits(1)" in source
        assert "method.RANDOM" not in source


def test_both_worsened_quarantine_records_numeric_scientific_evidence(tmp_path: Path):
    """A quarantine decision must remain independently auditable after failure."""

    from src.research import brainiac_preprocessing_v2 as v2

    class BothWorsened(_Pipeline):
        def __call__(self, item, scratch: Path):
            result = super().__call__(item, scratch)
            result["center_of_mass_before"] = [1.0, 0.0, 0.0]
            result["center_of_mass_after"] = [2.0, 0.0, 0.0]
            result["registration_similarity_before"] = -1.2
            result["registration_similarity_after"] = -0.8
            return result

    item = {"sample_hash": "s_" + "d" * 16, "t1_file_hash": "t1"}
    with pytest.raises(RuntimeError, match="QUARANTINE_BOTH_WORSENED"):
        v2.run_sample_transaction(
            item,
            tmp_path,
            pipeline=BothWorsened(),
            bindings=_bindings(),
        )

    evidence = json.loads(
        (tmp_path / "journals" / item["sample_hash"] / "quarantined.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence == {
        "sample_hash": item["sample_hash"],
        "status": "QUARANTINED",
        "reason_code": "BOTH_COM_AND_MATTES_MI_WORSENED",
        "com_coordinate_frame": "TEMPLATE_PHYSICAL_MM_RELATIVE",
        "center_of_mass_before": [1.0, 0.0, 0.0],
        "center_of_mass_after": [2.0, 0.0, 0.0],
        "com_norm_before_mm": 1.0,
        "com_norm_after_mm": 2.0,
        "registration_similarity_before": -1.2,
        "registration_similarity_after": -0.8,
        "similarity_semantics": "NEGATIVE_MATTES_MI_LOWER_IS_BETTER",
    }
