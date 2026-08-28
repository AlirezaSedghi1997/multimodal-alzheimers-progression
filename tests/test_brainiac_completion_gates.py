from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest


def test_pinned_vendor_bytes_match_strict_receipt_and_lf_checkout_policy() -> None:
    from src.research.brainiac_adapter import stable_source_tree_sha256

    root = Path(__file__).resolve().parents[1]
    vendor = (
        root
        / "vendor"
        / "BrainIAC"
        / "ba60f45bed832ee0e5683c678caeaeefe2072f0d"
    )
    receipt = json.loads(
        (
            root
            / "results"
            / "method_x_registry"
            / "brainiac_r1"
            / "brainiac_official_strict_load_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert stable_source_tree_sha256(vendor) == receipt["source_tree_sha256"]
    assert (
        hashlib.sha256((vendor / "LICENSE").read_bytes()).hexdigest()
        == receipt["license_sha256"]
    )
    attributes = (root / ".gitattributes").read_text(encoding="utf-8")
    pinned = "vendor/BrainIAC/ba60f45bed832ee0e5683c678caeaeefe2072f0d"
    for suffix in ("LICENSE", "requirements.txt", "**/*.csv", "**/*.ipynb"):
        assert f"{pinned}/{suffix} text eol=lf" in attributes


def test_official_runtime_identity_requires_vendor_monai_and_symbols() -> None:
    from src.research.brainiac_model_loader import validate_official_runtime_identity

    report = validate_official_runtime_identity(
        python_version="3.11.15",
        torch_version="2.5.1+cu121",
        monai_version="1.3.2",
        vendor_commit="ba60f45bed832ee0e5683c678caeaeefe2072f0d",
        module_path="src/model.py",
        class_name="ViTBackboneNet",
        loader_symbol="load_brainiac",
    )
    assert report["status"] == "PASS"
    assert report["official_vendor_runtime"] is True

    with pytest.raises(RuntimeError, match="BRAINIAC_MONAI_VERSION_MISMATCH"):
        validate_official_runtime_identity(
            python_version="3.11.15",
            torch_version="2.5.1+cu121",
            monai_version="1.5.2",
            vendor_commit="ba60f45bed832ee0e5683c678caeaeefe2072f0d",
            module_path="src/model.py",
            class_name="ViTBackboneNet",
            loader_symbol="load_brainiac",
        )


def test_primary_embedding_source_rejects_sensitivity_cache_route() -> None:
    from scripts.research.extract_brainiac_embeddings import (
        validate_primary_embedding_source_manifest,
    )

    with pytest.raises(RuntimeError, match="PRIMARY_EMBEDDING_SOURCE_ROUTE_INVALID"):
        validate_primary_embedding_source_manifest(
            {
                "schema_version": 1,
                "status": "COMPLETED",
                "route_role": "SENSITIVITY_ONLY_POST_HOC_RETROFIT",
                "source_role": "cache_128",
                "rows": 1669,
            }
        )


def test_primary_embedding_source_requires_real_affine_aware_96_outputs() -> None:
    from scripts.research.extract_brainiac_embeddings import (
        validate_primary_embedding_source_manifest,
    )

    report = validate_primary_embedding_source_manifest(
        {
            "schema_version": 2,
            "status": "COMPLETED",
            "route_role": "PROTOCOL_COMPATIBLE_INTERNAL_POSTHOC_PRIMARY",
            "source_role": "raw_t1",
            "label_read": False,
            "direct_128_to_96_used": False,
            "real_only": True,
            "rows": 1669,
            "output_shape": [96, 96, 96],
            "affine_aware": True,
            "manifest_sha256": "a" * 64,
        }
    )
    assert report["status"] == "PASS"
    assert report["rows"] == 1669


def test_sensitivity_audit_never_promotes_unknown_geometry_to_pass() -> None:
    from scripts.research.run_brainiac_r1a_sensitivity import (
        build_r1a_sensitivity_report,
    )

    report = build_r1a_sensitivity_report(
        geometry_status="UNKNOWN_GEOMETRY_PENDING_VENDOR_TEMPLATE_AFFINE",
        relative_differences=[0.041, 0.055],
        mask_edge_differences=[0.012, 0.019],
        sample_records=[],
    )
    assert report["status"] == "SENSITIVITY_COMPLETE_WITH_GEOMETRY_BLOCK"
    assert report["primary_route_authorized"] is False
    assert report["skull_stripping_shortcut_risk_level"] == "UNRESOLVED"
    assert "4.800%" in report["conclusion"]
    assert "31.4%" not in report["conclusion"]


def test_sensitivity_audit_fails_closed_on_nonfinite_metrics() -> None:
    from scripts.research.run_brainiac_r1a_sensitivity import (
        build_r1a_sensitivity_report,
    )

    with pytest.raises(RuntimeError, match="R1A_SENSITIVITY_METRIC_NONFINITE"):
        build_r1a_sensitivity_report(
            geometry_status="UNKNOWN_GEOMETRY_PENDING_VENDOR_TEMPLATE_AFFINE",
            relative_differences=[math.nan],
            mask_edge_differences=[0.01],
            sample_records=[],
        )


def test_unknown_overlap_caps_claim_but_does_not_block_internal_execution() -> None:
    from src.research.brainiac_adapter import classify_r1a_blockers

    report = classify_r1a_blockers(
        [
            "UNKNOWN_PRETRAINING_OVERLAP",
            "UPSTREAM_TEMPLATE_IDENTITY_UNKNOWN",
        ]
    )
    assert report["internal_execution_ready"] is True
    assert report["canonical_reproduction_ready"] is False
    assert report["status"] == "PASS_WITH_CLAIM_BLOCKS"
    assert report["claim_ceiling"] == "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"


def test_missing_runtime_or_protocol_still_blocks_internal_execution() -> None:
    from src.research.brainiac_adapter import classify_r1a_blockers

    report = classify_r1a_blockers(
        ["OFFICIAL_FORWARD_NOT_PROVEN", "UNKNOWN_PRETRAINING_OVERLAP"]
    )
    assert report["internal_execution_ready"] is False
    assert report["status"] == "BLOCKED"


def test_mask_edge_probe_does_not_wrap_foreground_across_boundaries() -> None:
    from src.research.brainiac_adapter import mask_edge_perturbation

    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[0, 2, 2] = 4.0
    dilated = mask_edge_perturbation(volume, mode="dilate", voxels=1)
    assert dilated[-1, 2, 2] == 0.0
    assert dilated[1, 2, 2] == 4.0


def test_vendor_source_hash_ignores_runtime_and_vcs_noise(tmp_path: Path) -> None:
    from src.research.brainiac_adapter import stable_source_tree_sha256

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    original = stable_source_tree_sha256(tmp_path)

    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "model.cpython-311.pyc").write_bytes(
        b"runtime cache"
    )
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "index").write_bytes(b"mutable vcs metadata")

    assert stable_source_tree_sha256(tmp_path) == original


def test_vendor_source_hash_changes_for_material_source_change(tmp_path: Path) -> None:
    from src.research.brainiac_adapter import stable_source_tree_sha256

    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "model.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    original = stable_source_tree_sha256(tmp_path)

    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert stable_source_tree_sha256(tmp_path) != original
