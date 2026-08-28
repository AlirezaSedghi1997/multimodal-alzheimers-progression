"""Adversarial contract tests for BrainIAC schema-v2 raw-T1 preprocessing.

Plan 002 RED-first suite.  All fixtures are deidentified synthetic phantoms
generated at test time; no study data, labels, or PET are read anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.research import brainiac_preprocessing_v2 as v2
from src.research import brainiac_preprocessing_v2_schema as schema


ROOT = Path(__file__).resolve().parents[1]


def _contract_bindings() -> dict[str, str]:
    return {
        "policy_sha256": v2.policy_sha256(),
        "config_sha256": "7" * 64,
        "pipeline_contract_sha256": "8" * 64,
        "dependency_closure_sha256": "9" * 64,
        "membership_sha256": "a" * 64,
        "cache_sha256": "b" * 64,
        "catalog_sha256": "c" * 64,
        "runtime_lock_sha256": "2" * 64,
        "vendor_tree_sha256": "3" * 64,
        "template_lock_sha256": "4" * 64,
        "official_transform_source_sha256": "5" * 64,
        "qc_threshold_manifest_sha256": "6" * 64,
    }


# --------------------------------------------------------------------------
# synthetic deidentified fixture helpers (tiny phantoms, no study data)
# --------------------------------------------------------------------------


def _synthetic_t1(path: Path, shape=(12, 14, 16), ellipsoid=True) -> Path:
    import numpy as np
    import nibabel as nib

    array = np.zeros(shape, dtype=np.float32)
    if ellipsoid:
        center = tuple(s // 2 for s in shape)
        radius = tuple(s / 4.0 for s in shape)
        grid = np.ogrid[tuple(slice(0, s) for s in shape)]
        mask = sum(
            ((g - c) / r) ** 2 for g, c, r in zip(grid, center, radius)
        ) <= 1.0
        array[mask] = np.linspace(50.0, 200.0, int(mask.sum())).astype(np.float32)
    affine = np.eye(4, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), str(path))
    return path


@pytest.fixture(autouse=True)
def _isolated_writer_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests run hermetically; the dedicated CNN-writer test overrides this."""

    monkeypatch.setattr(v2, "assert_no_cnn_writer", lambda: None)


def _receipt_body(root: Path, sample_hash: str = "s_" + "a" * 16, **overrides):
    output = _synthetic_t1(root / "outputs" / f"{sample_hash}.nii.gz", (8, 8, 8))
    transform = root / "transforms" / f"{sample_hash}.tfm"
    transform.parent.mkdir(parents=True, exist_ok=True)
    transform.write_bytes(b"#Insight Transform File V1.0\n")
    raw_output = output.read_bytes()
    raw_transform = transform.read_bytes()
    body = {
        "schema_version": 2,
        "artifact": schema.RECEIPT_ARTIFACT_TYPE,
        "sample_hash": sample_hash,
        "generation": 1,
        "status": "COMPLETED",
        "source_sha256": hashlib.sha256(b"synthetic-source").hexdigest(),
        "source_size": 16,
        "output_relative_path": f"outputs/{sample_hash}.nii.gz",
        "output_sha256": hashlib.sha256(raw_output).hexdigest(),
        "output_size": len(raw_output),
        "transform_relative_path": f"transforms/{sample_hash}.tfm",
        "transform_sha256": hashlib.sha256(raw_transform).hexdigest(),
        "transform_size": len(raw_transform),
        **_contract_bindings(),
        "vendor_commit": v2.VENDOR_COMMIT,
        "hd_bet_weight_sha256": v2.EXPECTED_HD_BET_WEIGHT_SHA256,
        "template_sha256": v2.EXPECTED_TEMPLATE_SHA256,
        "affine_before": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "affine_after": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "spacing_before": [1.0, 1.0, 1.0],
        "spacing_after": [1.0, 1.0, 1.0],
        "origin_before": [0.0, 0.0, 0.0],
        "origin_after": [0.0, 0.0, 0.0],
        "direction_before": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "direction_after": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "com_coordinate_frame": "TEMPLATE_PHYSICAL_MM_RELATIVE",
        "rigid_determinant": 1.0,
        "orthogonality_error": 0.0,
        "rotation_deg": 0.0,
        "translation_norm_mm": 0.0,
        "foreground_fraction": 0.25,
        "finite": True,
        "normalized_nonzero_mean": 0.0,
        "normalized_nonzero_std": 1.0,
        "center_of_mass_before": [4.0, 4.0, 4.0],
        "center_of_mass_after": [4.0, 4.0, 4.0],
        "registration_similarity_before": -1.0,
        "registration_similarity_after": -0.5,
        "labels_read": False,
        "pet_read": False,
        "raw_identifier_persisted": False,
        "raw_path_persisted": False,
        "direct_128_to_96_used": False,
        "created_at_utc": "2026-08-23T00:00:00+00:00",
    }
    body.update(overrides)
    return body


def _sealed(root: Path, **overrides) -> dict:
    return schema.seal_receipt(_receipt_body(root, **overrides))


# --------------------------------------------------------------------------
# schema / privacy recursive denylist
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        {"label": "AD"},
        {"nested": {"pet_input": "x"}},
        {"rows": [{"raw_path": "H:\\secret\\t1.nii"}]},
        {"raw_identifier": "sub-001"},
        {"meta": {"subject_id": 7, "deep": [{"age": 71}]}},
    ],
)
def test_privacy_denylist_rejects_forbidden_keys_anywhere(
    tmp_path: Path, forbidden: dict
) -> None:
    receipt = _sealed(tmp_path)
    receipt["extra"] = forbidden
    with pytest.raises(RuntimeError, match="PRIVACY_FIELD_FORBIDDEN"):
        schema.validate_receipt(receipt, artifact_root=tmp_path)


def test_audit_flags_must_be_false_and_present(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="AUDIT_FLAG_INVALID"):
        _sealed(tmp_path, labels_read=True)
    body = _receipt_body(tmp_path)
    del body["pet_read"]
    with pytest.raises(RuntimeError, match="FIELD_MISSING"):
        schema.seal_receipt(body)


def test_receipt_integrity_tamper_is_rejected(tmp_path: Path) -> None:
    receipt = _sealed(tmp_path)
    receipt["foreground_fraction"] = 0.9
    with pytest.raises(RuntimeError, match="INTEGRITY_MISMATCH"):
        schema.validate_receipt(receipt, artifact_root=tmp_path)


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    body = _receipt_body(tmp_path)
    del body["rigid_determinant"]
    with pytest.raises(RuntimeError, match="FIELD_MISSING"):
        schema.seal_receipt(body)


# --------------------------------------------------------------------------
# hash / size / reparse / zero / temp binding
# --------------------------------------------------------------------------


def test_output_hash_or_size_mismatch_rejects(tmp_path: Path) -> None:
    receipt = _sealed(tmp_path)
    (tmp_path / receipt["output_relative_path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="OUTPUT_BINDING_MISMATCH"):
        schema.validate_receipt(receipt, artifact_root=tmp_path)
    receipt2 = _sealed(tmp_path, sample_hash="s_" + "b" * 16)
    receipt2["output_size"] += 1
    receipt2["integrity_sha256"] = schema.canonical_sha(
        {k: v for k, v in receipt2.items() if k != "integrity_sha256"}
    )
    with pytest.raises(RuntimeError, match="OUTPUT_BINDING_MISMATCH"):
        schema.validate_receipt(receipt2, artifact_root=tmp_path)


def test_zero_byte_or_temp_or_reparse_artifact_rejects(tmp_path: Path) -> None:
    receipt = _sealed(tmp_path, sample_hash="s_" + "c" * 16)
    (tmp_path / receipt["output_relative_path"]).write_bytes(b"")
    with pytest.raises(RuntimeError, match="ARTIFACT_INVALID"):
        schema.validate_receipt(receipt, artifact_root=tmp_path)
    receipt = _sealed(tmp_path, sample_hash="s_" + "d" * 16)
    (tmp_path / receipt["transform_relative_path"]).unlink()
    (tmp_path / receipt["transform_relative_path"].replace(".tfm", ".tmp.tfm")).write_bytes(b"x")
    with pytest.raises(RuntimeError, match="ARTIFACT_INVALID"):
        schema.validate_receipt(receipt, artifact_root=tmp_path)


def test_reparse_output_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = _sealed(tmp_path, sample_hash="s_" + "e" * 16)
    monkeypatch.setattr(v2, "_is_reparse", lambda path: True)
    with pytest.raises(RuntimeError, match="ARTIFACT_INVALID"):
        v2.verify_receipt_artifacts(receipt, artifact_root=tmp_path)


# --------------------------------------------------------------------------
# geometry / numeric QC boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("rigid_determinant", 0.9989, "QC_OUT_OF_BOUNDS"),
        ("rigid_determinant", 1.0011, "QC_OUT_OF_BOUNDS"),
        ("rigid_determinant", 0.999, None),  # boundary inclusive
        ("rigid_determinant", 1.001, None),
        ("orthogonality_error", 1.1e-3, "QC_OUT_OF_BOUNDS"),
        ("rotation_deg", 45.1, "QC_OUT_OF_BOUNDS"),
        ("translation_norm_mm", 100.1, "QC_OUT_OF_BOUNDS"),
        ("foreground_fraction", 0.049, "QC_OUT_OF_BOUNDS"),
        ("foreground_fraction", 0.51, "QC_OUT_OF_BOUNDS"),
        ("foreground_fraction", 0.05, None),
        ("foreground_fraction", 0.50, None),
        ("normalized_nonzero_std", 0.94, "QC_OUT_OF_BOUNDS"),
        ("normalized_nonzero_std", 1.06, "QC_OUT_OF_BOUNDS"),
    ],
)
def test_geometry_threshold_boundaries(
    tmp_path: Path, field: str, value: float, match: str | None
) -> None:
    if match is None:
        receipt = _sealed(tmp_path, **{field: value})
        schema.validate_receipt(receipt, artifact_root=tmp_path)
    else:
        # Invalid QC cannot even be sealed; the rejection must happen there.
        with pytest.raises(RuntimeError, match=match):
            _sealed(tmp_path, **{field: value})


def test_nonfinite_or_nonzero_mean_rejects(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="QC_NONFINITE"):
        _sealed(tmp_path, finite=False)
    with pytest.raises(RuntimeError, match="QC_OUT_OF_BOUNDS"):
        _sealed(tmp_path, normalized_nonzero_mean=2e-3)


# --------------------------------------------------------------------------
# COM + registration-similarity quarantine rule
# --------------------------------------------------------------------------


def test_com_and_similarity_both_worse_quarantines() -> None:
    # Metric convention (verified on this SimpleITK build): lower = better
    # aligned, so a metric INCREASE means similarity worsened.
    assert (
        schema.quarantine_decision(
            com_before=(0.0, 0.0, 0.0),
            com_after=(3.0, 0.0, 0.0),
            similarity_before=-0.9,
            similarity_after=-0.5,  # metric increased -> worse
        )
        == "QUARANTINE_BOTH_WORSENED"
    )
    assert (
        schema.quarantine_decision(
            com_before=(0.0, 0.0, 0.0),
            com_after=(3.0, 0.0, 0.0),
            similarity_before=-0.5,
            similarity_after=-0.9,  # metric decreased -> better
        )
        == "OK"
    )
    assert (
        schema.quarantine_decision(
            com_before=(0.0, 0.0, 0.0),
            com_after=(0.0, 0.0, 0.0),  # geometry unchanged
            similarity_before=-0.5,
            similarity_after=-0.4,
        )
        == "OK"
    )


# --------------------------------------------------------------------------
# MAD calibration label-blind
# --------------------------------------------------------------------------


def _metrics(values: list[float]) -> dict[str, dict[str, float]]:
    return {
        f"s_{i:04d}": {"foreground_fraction": v} for i, v in enumerate(values)
    }


def test_calibration_selection_is_32_lowest_hash_of_sample_hash() -> None:
    members = [f"s_{i:04d}" for i in range(100)]
    selected = v2.select_calibration_samples(members, count=32)
    expected = sorted(members, key=lambda m: hashlib.sha256(m.encode()).hexdigest())[:32]
    assert selected == expected
    assert len(set(selected)) == 32


def test_mad_thresholds_and_mad_zero_fallback() -> None:
    values = [round(0.20 + 0.01 * i, 4) for i in range(32)]
    thresholds = v2.compute_mad_thresholds(
        values, fixed_bounds=(0.05, 0.50)
    )
    assert thresholds["mad_zero_fallback"] is False
    assert thresholds["low"] < min(values) <= thresholds["low"] + 6 * thresholds["mad"] + 1e-9
    assert thresholds["high"] > max(values)
    degenerate = v2.compute_mad_thresholds([0.25] * 32, fixed_bounds=(0.05, 0.50))
    assert degenerate["mad_zero_fallback"] is True
    assert (degenerate["low"], degenerate["high"]) == (0.05, 0.50)


def test_threshold_manifest_is_deterministic_and_immutable() -> None:
    metrics = _metrics([round(0.20 + 0.01 * (i % 5), 4) for i in range(32)])
    selected = {m: v for m, v in list(metrics.items())[:32]}
    receipt_hashes = {sample: hashlib.sha256(sample.encode()).hexdigest() for sample in selected}
    contract = {
        "policy_sha256": "1" * 64,
        "pipeline_contract_sha256": "2" * 64,
        "membership_sha256": "3" * 64,
    }
    first = v2.build_threshold_manifest(
        selected,
        receipt_integrity_sha256s=receipt_hashes,
        contract_bindings=contract,
    )
    second = v2.build_threshold_manifest(
        selected,
        receipt_integrity_sha256s=receipt_hashes,
        contract_bindings=contract,
    )
    assert first == second
    schema.validate_threshold_manifest(first)
    tampered = dict(first)
    tampered["thresholds"]["foreground_fraction"]["low"] = 0.0
    with pytest.raises(RuntimeError, match="THRESHOLD_MANIFEST_INVALID"):
        schema.validate_threshold_manifest(tampered)


# --------------------------------------------------------------------------
# journal / transaction / resume
# --------------------------------------------------------------------------


class _FakePipeline:
    """Deterministic synthetic pipeline stage for transaction tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0

    def __call__(self, item: dict, scratch: Path) -> dict:
        self.calls += 1
        output = _synthetic_t1(scratch / "output.nii.gz", (8, 8, 8))
        transform = scratch / "transform.tfm"
        transform.write_bytes(b"#tfm\n")
        return {
            "output_path": output,
            "transform_path": transform,
            "source_sha256": hashlib.sha256(b"synthetic-source").hexdigest(),
            "source_size": 16,
            "affine_before": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "affine_after": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "spacing_before": [1.0, 1.0, 1.0],
            "spacing_after": [1.0, 1.0, 1.0],
            "origin_before": [0.0, 0.0, 0.0],
            "origin_after": [0.0, 0.0, 0.0],
            "direction_before": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "direction_after": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "com_coordinate_frame": "TEMPLATE_PHYSICAL_MM_RELATIVE",
            "rigid_determinant": 1.0,
            "orthogonality_error": 0.0,
            "rotation_deg": 0.0,
            "translation_norm_mm": 0.0,
            "foreground_fraction": 0.25,
            "normalized_nonzero_mean": 0.0,
            "normalized_nonzero_std": 1.0,
            "center_of_mass_before": [4.0, 4.0, 4.0],
            "center_of_mass_after": [4.0, 4.0, 4.0],
            "registration_similarity_before": -1.0,
            "registration_similarity_after": -0.5,
            "finite": True,
        }


def _binding_overrides(root: Path) -> dict:
    return _contract_bindings()


def test_transaction_completes_and_rerun_skips(tmp_path: Path) -> None:
    root = tmp_path / "v2root"
    pipeline = _FakePipeline(root)
    item = {"sample_hash": "s_" + "f" * 16, "t1_file_hash": "t1abc"}
    receipt = v2.run_sample_transaction(
        item, root, pipeline=pipeline, bindings=_binding_overrides(root)
    )
    schema.validate_receipt(receipt, artifact_root=root)
    journal_dir = root / "journals" / item["sample_hash"]
    assert (journal_dir / "committed.json").is_file()
    again = v2.run_sample_transaction(
        item, root, pipeline=pipeline, bindings=_binding_overrides(root)
    )
    assert again["status"] == "COMPLETED"
    assert pipeline.calls == 1  # committed valid sample is skipped


def test_payload_without_journal_is_quarantined_not_overwritten(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v2root"
    item = {"sample_hash": "s_" + "1" * 16, "t1_file_hash": "t1abc"}
    (root / "outputs").mkdir(parents=True)
    stale = root / "outputs" / f"{item['sample_hash']}.nii.gz"
    stale.write_bytes(b"stale uncommitted payload")
    receipt = v2.run_sample_transaction(
        item, root, pipeline=_FakePipeline(root), bindings=_binding_overrides(root)
    )
    schema.validate_receipt(receipt, artifact_root=root)
    assert stale.read_bytes() != b"stale uncommitted payload"
    quarantined = list((root / "quarantine").rglob("*.nii.gz"))
    assert any(b"stale uncommitted payload" == q.read_bytes() for q in quarantined)


def test_committed_mismatch_blocks_recompute(tmp_path: Path) -> None:
    root = tmp_path / "v2root"
    item = {"sample_hash": "s_" + "2" * 16, "t1_file_hash": "t1abc"}
    receipt = v2.run_sample_transaction(
        item, root, pipeline=_FakePipeline(root), bindings=_binding_overrides(root)
    )
    # Simulate live-byte drift after commit.
    (root / receipt["output_relative_path"]).write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="COMMITTED_MISMATCH"):
        v2.run_sample_transaction(
            item, root, pipeline=_FakePipeline(root), bindings=_binding_overrides(root)
        )


def test_receipt_without_journal_commit_recovers_when_bytes_match(tmp_path: Path) -> None:
    root = tmp_path / "v2root"
    item = {"sample_hash": "s_" + "3" * 16, "t1_file_hash": "t1abc"}
    receipt = v2.run_sample_transaction(
        item, root, pipeline=_FakePipeline(root), bindings=_binding_overrides(root)
    )
    (root / "journals" / item["sample_hash"] / "committed.json").unlink()
    recovered = v2.load_committed_receipt(
        item["sample_hash"], root, recover_missing_journal=True
    )
    assert recovered["integrity_sha256"] == receipt["integrity_sha256"]
    assert (root / "journals" / item["sample_hash"] / "committed.json").is_file()


def test_signal_mid_sample_completes_then_stops_with_75(tmp_path: Path) -> None:
    root = tmp_path / "v2root"
    items = [
        {"sample_hash": f"s_{i:04x}" + "0" * 12, "t1_file_hash": f"t1{i}"}
        for i in range(3)
    ]
    stop_after = {"n": 1}

    def stop_requested() -> bool:
        return stop_after["n"] <= 0

    state = {"completed": 0}
    original = v2.run_sample_transaction

    def counting(item, root_arg, **kwargs):
        result = original(item, root_arg, **kwargs)
        state["completed"] += 1
        stop_after["n"] -= 1
        return result

    import unittest.mock as mock

    with mock.patch.object(v2, "run_sample_transaction", side_effect=counting):
        report = v2.run_batch(
            items,
            root,
            pipeline_factory=lambda: _FakePipeline(root),
            bindings=_binding_overrides(root),
            stop_requested=stop_requested,
        )
    assert report["status"] == "PAUSED_RESUMABLE"
    assert report["exit_code"] == 75
    assert report["completed"] == 1


def test_stale_sample_lock_is_reconciled_with_dead_pid_proof(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v2root"
    item = {"sample_hash": "s_" + "9" * 16, "t1_file_hash": "t1abc"}
    journal_dir = root / "journals" / item["sample_hash"]
    journal_dir.mkdir(parents=True)
    (journal_dir / ".sample.lock").write_text(
        json.dumps({"pid": 999998, "nonce": "n", "ownership_token": "t"}), encoding="utf-8"
    )
    receipt = v2.run_sample_transaction(
        item, root, pipeline=_FakePipeline(root), bindings=_binding_overrides(root)
    )
    schema.validate_receipt(receipt, artifact_root=root)
    assert not (journal_dir / ".sample.lock").exists()


def test_live_sample_lock_owner_blocks_recompute(tmp_path: Path) -> None:
    root = tmp_path / "v2root"
    item = {"sample_hash": "s_" + "8" * 16, "t1_file_hash": "t1abc"}
    journal_dir = root / "journals" / item["sample_hash"]
    journal_dir.mkdir(parents=True)
    (journal_dir / ".sample.lock").write_text(
        json.dumps({"pid": os.getpid(), "nonce": "n", "ownership_token": "t"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="SAMPLE_LOCK_OWNER_ACTIVE"):
        v2.run_sample_transaction(
            item, root, pipeline=_FakePipeline(root), bindings=_binding_overrides(root)
        )


def test_duplicate_sample_and_generation_rollback_block(tmp_path: Path) -> None:
    root = tmp_path / "v2root"
    item = {"sample_hash": "s_" + "4" * 16, "t1_file_hash": "t1abc"}
    receipt = v2.run_sample_transaction(
        item, root, pipeline=_FakePipeline(root), bindings=_binding_overrides(root)
    )
    rolled = dict(receipt)
    rolled["generation"] = 0
    rolled["integrity_sha256"] = schema.canonical_sha(
        {k: v for k, v in rolled.items() if k != "integrity_sha256"}
    )
    (root / "receipts" / f"{item['sample_hash']}.json").write_text(
        json.dumps(rolled, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="GENERATION_ROLLBACK"):
        v2.load_committed_receipt(item["sample_hash"], root)


# --------------------------------------------------------------------------
# aggregate v2 / schema nonmixing
# --------------------------------------------------------------------------


def test_aggregate_rejects_schema_v1_and_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "v2root"
    receipt = v2.run_sample_transaction(
        {"sample_hash": "s_" + "5" * 16, "t1_file_hash": "t1abc"},
        root,
        pipeline=_FakePipeline(root),
        bindings=_binding_overrides(root),
    )
    v1 = dict(receipt)
    v1["schema_version"] = 1
    v1["integrity_sha256"] = schema.canonical_sha(
        {k: v for k, v in v1.items() if k != "integrity_sha256"}
    )
    with pytest.raises(RuntimeError, match="SCHEMA_V1_FORBIDDEN"):
        schema.validate_receipt(v1, artifact_root=root)
    with pytest.raises(RuntimeError, match="DUPLICATE_SAMPLE"):
        schema.build_aggregate(
            receipts=[receipt, receipt],
            expected_sample_hashes=[receipt["sample_hash"]],
            artifact_root=root,
        )


def test_aggregate_count_must_match_expected(tmp_path: Path) -> None:
    root = tmp_path / "v2root"
    receipt = v2.run_sample_transaction(
        {"sample_hash": "s_" + "6" * 16, "t1_file_hash": "t1abc"},
        root,
        pipeline=_FakePipeline(root),
        bindings=_binding_overrides(root),
    )
    with pytest.raises(RuntimeError, match="COUNT_MISMATCH"):
        schema.build_aggregate(
            receipts=[receipt],
            expected_sample_hashes=[receipt["sample_hash"], "s_missing"],
            artifact_root=root,
        )


# --------------------------------------------------------------------------
# lease enforcement / CNN writer
# --------------------------------------------------------------------------


def test_execute_requires_action_lease_before_output_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "v2root"

    from src.research import brainiac_execution_gate as gate

    def _no_lease(action: str):
        raise gate.BrainIACExecutionBlocked(
            "BRAINIAC_EXECUTION_HARD_DISABLED:BRAINIAC_ACTION_LEASE_REQUIRED"
        )

    monkeypatch.setattr(v2, "brainiac_action_lease", _no_lease)
    with pytest.raises(RuntimeError, match="BRAINIAC_EXECUTION_HARD_DISABLED"):
        v2.execute(mode="smoke", limit=2, root=root, device="cpu", pipeline_factory=lambda: _FakePipeline(root), bindings=_binding_overrides(root), membership=[{"sample_hash": "s_x", "t1_file_hash": "t"}])
    assert not (root / "outputs").exists()


def test_active_cnn_writer_blocks_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        v2,
        "assert_no_cnn_writer",
        lambda: (_ for _ in ()).throw(RuntimeError("BRAINIAC_CNN_WRITER_ACTIVE")),
    )
    root = tmp_path / "v2root"
    with pytest.raises(RuntimeError, match="BRAINIAC_CNN_WRITER_ACTIVE"):
        v2.run_sample_transaction(
            {"sample_hash": "s_" + "7" * 16, "t1_file_hash": "t1abc"},
            root,
            pipeline=_FakePipeline(root),
            bindings=_binding_overrides(root),
        )


# --------------------------------------------------------------------------
# official transform identity
# --------------------------------------------------------------------------


def test_official_transform_identity_assertion_blocks_unknown_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Wrong:
        def get_validation_transform(self, image_size=(96, 96, 96)):
            return object()  # not a Compose

    monkeypatch.setattr(v2, "_import_vendor_dataset", lambda: _Wrong())
    with pytest.raises(RuntimeError, match="TRANSFORM_IDENTITY_UNKNOWN"):
        v2.build_contract_transform()


# --------------------------------------------------------------------------
# morphology without wrap-around
# --------------------------------------------------------------------------


def test_morphology_never_wraps() -> None:
    import numpy as np

    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[0, 2, 2] = True  # voxel at the border face
    dilated = v2.dilate_mask(mask, iterations=1)
    assert dilated[4, 2, 2] is False or dilated[4, 2, 2] == np.False_  # no wrap
    assert dilated[1, 2, 2]
    eroded = v2.erode_mask(dilated, iterations=1)
    assert eroded[0, 2, 2]


# --------------------------------------------------------------------------
# sequential vs orchestrated equivalence
# --------------------------------------------------------------------------


def test_orchestrated_matches_sequential_receipts(tmp_path: Path) -> None:
    root = tmp_path / "v2root"
    items = [
        {"sample_hash": f"s_{i:04x}" + "9" * 12, "t1_file_hash": f"t1{i}"}
        for i in range(4)
    ]
    sequential_root = root / "sequential"
    orchestrated_root = root / "orchestrated"
    seq_receipts = v2.run_batch(
        items,
        sequential_root,
        pipeline_factory=lambda: _FakePipeline(sequential_root),
        bindings=_binding_overrides(root),
    )["receipts"]
    orch_receipts = v2.run_orchestrated(
        items,
        orchestrated_root,
        pipeline_factory=lambda: _FakePipeline(orchestrated_root),
        bindings=_binding_overrides(root),
        cpu_workers=2,
        gpu_workers=1,
    )
    by_sample_seq = {r["sample_hash"]: r for r in seq_receipts}
    by_sample_orch = {r["sample_hash"]: r for r in orch_receipts}
    assert set(by_sample_seq) == set(by_sample_orch)
    for sample_hash, seq in by_sample_seq.items():
        orch = by_sample_orch[sample_hash]
        for field in (
            "output_sha256",
            "transform_sha256",
            "foreground_fraction",
            "rigid_determinant",
            "affine_before",
            "affine_after",
            "normalized_nonzero_std",
        ):
            assert seq[field] == orch[field], field


# --------------------------------------------------------------------------
# content digest determinism (card's byte-identity exception path)
# --------------------------------------------------------------------------


def test_synthetic_content_digest_is_deterministic(tmp_path: Path) -> None:
    first = _synthetic_t1(tmp_path / "a.nii.gz")
    second = _synthetic_t1(tmp_path / "b.nii.gz")
    assert v2.content_digest(first) == v2.content_digest(second)
