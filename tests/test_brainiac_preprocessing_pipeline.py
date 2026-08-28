from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _item_receipt(sample_hash: str, output: Path, transform: Path) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETED",
        "sample_hash": sample_hash,
        "source_role": "raw_t1",
        "label_read": False,
        "direct_128_to_96_used": False,
        "output_path": output.name,
        "output_sha256": _sha(output),
        "output_bytes": output.stat().st_size,
        "output_shape": [96, 96, 96],
        "output_affine_sha256": "a" * 64,
        "source_affine_sha256": "b" * 64,
        "n4_artifact_sha256": "c" * 64,
        "registered_artifact_sha256": "d" * 64,
        "brain_mask_sha256": "e" * 64,
        "rigid_transform_path": transform.name,
        "rigid_transform_sha256": _sha(transform),
        "rigid_transform_bytes": transform.stat().st_size,
        "template_sha256": "f" * 64,
        "finite": True,
        "foreground_fraction": 0.25,
        "foreground_mean": 0.0,
        "foreground_std": 1.0,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["integrity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return body


def test_completed_item_validation_binds_bytes_and_rejects_tamper(tmp_path: Path) -> None:
    from src.research.brainiac_preprocessing import validate_completed_item

    output = tmp_path / "s_abc.nii.gz"
    transform = tmp_path / "s_abc.tfm"
    output.write_bytes(b"verified-output")
    transform.write_bytes(b"verified-transform")
    receipt = _item_receipt("s_abc", output, transform)

    report = validate_completed_item(receipt, artifact_root=tmp_path)
    assert report["status"] == "PASS"

    output.write_bytes(b"tampered-output")
    with pytest.raises(RuntimeError, match="BRAINIAC_PREPROCESS_OUTPUT_HASH_MISMATCH"):
        validate_completed_item(receipt, artifact_root=tmp_path)


def test_completed_item_receipt_cannot_persist_raw_paths_or_labels(tmp_path: Path) -> None:
    from src.research.brainiac_preprocessing import validate_completed_item

    output = tmp_path / "s_abc.nii.gz"
    transform = tmp_path / "s_abc.tfm"
    output.write_bytes(b"verified-output")
    transform.write_bytes(b"verified-transform")
    receipt = _item_receipt("s_abc", output, transform)
    receipt["raw_path"] = r"H:\A4 Learn\T1\identifier.nii.gz"
    receipt["ab_label"] = 1
    body = {key: value for key, value in receipt.items() if key != "integrity_sha256"}
    receipt["integrity_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(RuntimeError, match="BRAINIAC_PREPROCESS_PRIVACY_FIELD_FORBIDDEN"):
        validate_completed_item(receipt, artifact_root=tmp_path)


def test_aggregate_manifest_requires_exact_unique_membership(tmp_path: Path) -> None:
    from src.research.brainiac_preprocessing import build_aggregate_manifest

    receipts = []
    for sample_hash in ("s_a", "s_b"):
        output = tmp_path / f"{sample_hash}.nii.gz"
        transform = tmp_path / f"{sample_hash}.tfm"
        output.write_bytes(sample_hash.encode())
        transform.write_bytes((sample_hash + "-transform").encode())
        receipts.append(_item_receipt(sample_hash, output, transform))

    manifest = build_aggregate_manifest(
        receipts=receipts,
        expected_sample_hashes=["s_a", "s_b"],
        artifact_root=tmp_path,
        template_sha256="f" * 64,
        runtime_lock_sha256="1" * 64,
        strict_receipt_sha256="2" * 64,
        forward_receipt_sha256="3" * 64,
    )
    assert manifest["status"] == "COMPLETED"
    assert manifest["rows"] == 2
    assert manifest["real_only"] is True
    assert manifest["label_read"] is False
    assert manifest["route_role"] == "PROTOCOL_COMPATIBLE_INTERNAL_POSTHOC_PRIMARY"
    assert manifest["canonical_reproduction_claim_allowed"] is False
    assert manifest["upstream_template_identity"] == "UNKNOWN_NOT_DISTRIBUTED"

    with pytest.raises(RuntimeError, match="BRAINIAC_PREPROCESS_MEMBERSHIP_MISMATCH"):
        build_aggregate_manifest(
            receipts=receipts,
            expected_sample_hashes=["s_a", "s_b", "s_c"],
            artifact_root=tmp_path,
            template_sha256="f" * 64,
            runtime_lock_sha256="1" * 64,
            strict_receipt_sha256="2" * 64,
            forward_receipt_sha256="3" * 64,
        )


def test_official_96_normalization_is_nonzero_channelwise_and_finite() -> None:
    from src.research.brainiac_preprocessing import normalize_nonzero_channelwise

    volume = np.zeros((1, 8, 8, 8), dtype=np.float32)
    volume[:, 2:6, 2:6, 2:6] = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4) + 1
    normalized = normalize_nonzero_channelwise(volume)
    foreground = normalized[volume != 0]

    assert normalized.shape == volume.shape
    assert np.isfinite(normalized).all()
    assert np.all(normalized[volume == 0] == 0)
    assert float(foreground.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(foreground.std()) == pytest.approx(1.0, abs=1e-6)


def test_status_probe_does_not_quarantine_uncommitted_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import preprocess_brainiac_raw_t1 as pipeline

    artifact_root = tmp_path / "derived"
    output_root = artifact_root / "preprocessed_96"
    transform_root = artifact_root / "rigid_transforms"
    receipt_root = artifact_root / "preprocessing_receipts"
    orphan_root = artifact_root / "orphans"
    for directory in (output_root, transform_root, receipt_root, orphan_root):
        directory.mkdir(parents=True)
    monkeypatch.setattr(pipeline, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(pipeline, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(pipeline, "TRANSFORM_ROOT", transform_root)
    monkeypatch.setattr(pipeline, "RECEIPT_ROOT", receipt_root)
    monkeypatch.setattr(pipeline, "ORPHAN_ROOT", orphan_root)

    item = pipeline.WorkItem("s_abc", "t1_abc", "A4", tmp_path / "raw.nii.gz")
    uncommitted = output_root / "s_abc.nii.gz"
    uncommitted.write_bytes(b"not committed")

    assert pipeline._load_valid_completed(item, recover_uncommitted=False) is None
    assert uncommitted.read_bytes() == b"not committed"
    assert not any(orphan_root.iterdir())
