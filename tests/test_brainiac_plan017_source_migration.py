from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.research import migrate_brainiac_plan017_resume_v1 as migration
from src.research import brainiac_preprocessing_v2 as v2


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bindings(dependency: str) -> dict[str, str]:
    return {
        "policy_sha256": _sha("policy"),
        "config_sha256": _sha("config"),
        "pipeline_contract_sha256": _sha("pipeline"),
        "dependency_closure_sha256": dependency,
        "membership_sha256": _sha("membership"),
        "cache_sha256": _sha("cache"),
        "catalog_sha256": _sha("catalog"),
        "runtime_lock_sha256": _sha("runtime"),
        "vendor_tree_sha256": _sha("vendor"),
        "template_sha256": _sha("template"),
        "template_lock_sha256": _sha("template-lock"),
        "official_transform_source_sha256": _sha("transform"),
        "qc_threshold_manifest_sha256": migration.PLACEHOLDER_THRESHOLD_SHA256,
    }


class _PayloadPipeline:
    def __call__(self, _item, scratch: Path):
        output = scratch / "output.nii.gz"
        transform = scratch / "transform.tfm"
        output.write_bytes(b"byte-identical-nifti-payload")
        transform.write_bytes(b"byte-identical-rigid-transform")
        return {
            "output_path": output,
            "transform_path": transform,
            "source_sha256": _sha("raw-t1"),
            "source_size": 123,
            "affine_before": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "affine_after": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "spacing_before": [1, 1, 1],
            "spacing_after": [1, 1, 1],
            "origin_before": [0, 0, 0],
            "origin_after": [0, 0, 0],
            "direction_before": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "direction_after": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "com_coordinate_frame": "TEMPLATE_PHYSICAL_MM_RELATIVE",
            "center_of_mass_before": [1, 1, 1],
            "center_of_mass_after": [0.5, 0.5, 0.5],
            "registration_similarity_before": -0.2,
            "registration_similarity_after": -0.3,
            "rigid_determinant": 1.0,
            "orthogonality_error": 0.0,
            "rotation_deg": 1.0,
            "translation_norm_mm": 2.0,
            "foreground_fraction": 0.2,
            "finite": True,
            "normalized_nonzero_mean": 0.0,
            "normalized_nonzero_std": 1.0,
        }


def test_source_only_migration_recommits_identical_bytes_and_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "local_scratch"
    monkeypatch.setattr(v2, "LOCAL_SCRATCH_ROOT", scratch)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    sample = "s_" + "a" * 16
    old_bindings = _bindings(_sha("old-closure"))
    new_bindings = _bindings(_sha("new-closure"))
    old = v2.run_sample_transaction(
        {"sample_hash": sample, "t1_file_hash": "hash-only"},
        source,
        pipeline=_PayloadPipeline(),
        bindings=old_bindings,
        scratch_base=scratch,
    )

    migrated = migration.migrate_receipt_root(
        source,
        destination,
        source_receipts={sample: old},
        new_bindings=new_bindings,
    )[sample]

    assert migrated["dependency_closure_sha256"] == new_bindings[
        "dependency_closure_sha256"
    ]
    assert migrated["output_sha256"] == old["output_sha256"]
    assert migrated["transform_sha256"] == old["transform_sha256"]
    assert migration._receipt_projection(migrated) == migration._receipt_projection(old)
    assert v2.load_committed_receipt(
        sample, destination, expected_bindings=new_bindings
    )["integrity_sha256"] == migrated["integrity_sha256"]


def test_dependency_transition_rejects_any_scientific_input_drift() -> None:
    old_python = {
        path: transition[0]
        for path, transition in migration.ALLOWED_SOURCE_TRANSITIONS.items()
    }
    old_python["unchanged.py"] = _sha("unchanged")
    new_python = {
        path: transition[1]
        for path, transition in migration.ALLOWED_SOURCE_TRANSITIONS.items()
    }
    new_python["unchanged.py"] = _sha("unchanged")
    old = {
        "closure_sha256": migration.OLD_DEPENDENCY_CLOSURE_SHA256,
        "python_closure": old_python,
        "dynamic_code_inputs": {"vendor": {"sha256": _sha("vendor")}},
        "static_inputs": {"policy": {"sha256": _sha("policy")}},
    }
    new = {
        **old,
        "closure_sha256": _sha("new-closure"),
        "python_closure": new_python,
    }
    migration._assert_dependency_transition(old, new)
    new["static_inputs"] = {"policy": {"sha256": _sha("drift")}}
    with pytest.raises(RuntimeError, match="SCIENTIFIC_INPUT_DRIFT"):
        migration._assert_dependency_transition(old, new)


def test_migration_source_transition_allowlist_matches_live_bytes() -> None:
    for relative, (_old, expected_new) in migration.ALLOWED_SOURCE_TRANSITIONS.items():
        assert migration.runner.sha256_file(migration.ROOT / relative) == expected_new
