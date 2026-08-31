from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from scripts.research import migrate_brainiac_plan017_postcalibration_v2 as migration
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
        "official_transform_source_sha256": _sha("official-transform"),
    }


def _receipt(sample: str, integrity: str) -> dict[str, object]:
    return {
        "sample_hash": sample,
        "integrity_sha256": integrity,
        "foreground_fraction": 0.2,
        "normalized_nonzero_std": 1.0,
        "rotation_deg": 2.0,
        "translation_norm_mm": 3.0,
    }


def test_postcalibration_transition_rejects_any_unapproved_source_change() -> None:
    old = {
        "closure_sha256": migration.OLD_DEPENDENCY_CLOSURE_SHA256,
        "python_closure": {
            migration.RUNNER_RELATIVE_PATH: migration.OLD_RUNNER_SHA256,
            "unchanged.py": _sha("unchanged"),
        },
        "dynamic_code_inputs": {"vendor": {"sha256": _sha("vendor")}},
        "static_inputs": {"policy": {"sha256": _sha("policy")}},
    }
    new = {
        **old,
        "closure_sha256": _sha("new-closure"),
        "python_closure": {
            migration.RUNNER_RELATIVE_PATH: migration.NEW_RUNNER_SHA256,
            "unchanged.py": _sha("unchanged"),
        },
    }
    migration.assert_dependency_transition(old, new)

    new["python_closure"]["unchanged.py"] = _sha("unexpected-change")
    with pytest.raises(RuntimeError, match="UNAPPROVED_SOURCE_DRIFT"):
        migration.assert_dependency_transition(old, new)


def test_threshold_migration_changes_only_provenance_not_numeric_science() -> None:
    samples = [f"s_{index:016x}" for index in range(32)]
    old_receipts = {
        sample: _receipt(sample, _sha(f"old-{sample}")) for sample in samples
    }
    old_contract = _bindings(migration.OLD_DEPENDENCY_CLOSURE_SHA256)
    old_manifest = v2.build_threshold_manifest(
        {
            sample: {
                "foreground_fraction": 0.2,
                "normalized_nonzero_std": 1.0,
                "rotation_deg": 2.0,
                "translation_norm_mm": 3.0,
            }
            for sample in samples
        },
        receipt_integrity_sha256s={
            sample: str(receipt["integrity_sha256"])
            for sample, receipt in old_receipts.items()
        },
        contract_bindings=old_contract,
        fixed_bounds={
            "foreground_fraction": (0.05, 0.50),
            "normalized_nonzero_std": (0.95, 1.05),
        },
    )
    migrated_receipts = {
        sample: {**receipt, "integrity_sha256": _sha(f"new-{sample}")}
        for sample, receipt in old_receipts.items()
    }
    new_contract = _bindings(_sha("new-closure"))

    migrated = migration.build_migrated_threshold_manifest(
        old_manifest,
        migrated_receipts=migrated_receipts,
        new_contract_bindings=new_contract,
    )

    assert migration.threshold_scientific_projection(migrated) == (
        migration.threshold_scientific_projection(old_manifest)
    )
    assert migrated["contract_bindings"] == new_contract
    assert migrated["receipt_integrity_sha256s"] != old_manifest[
        "receipt_integrity_sha256s"
    ]
    assert migrated["thresholds"] == old_manifest["thresholds"]
    assert migrated["metrics"] == old_manifest["metrics"]


@pytest.mark.parametrize(
    ("primary_valid_n", "candidate_exists", "exclusion", "reason"),
    [
        (1, False, {"status": "PASS", "blockers": []}, "PRIMARY_ALREADY_STARTED"),
        (0, True, {"status": "PASS", "blockers": []}, "LATER_STAGE_ALREADY_EXISTS"),
        (0, False, {"status": "UNKNOWN", "blockers": []}, "WRITER_EXCLUSION_BLOCKED"),
        (0, False, {"status": "PASS", "blockers": ["writer"]}, "WRITER_EXCLUSION_BLOCKED"),
    ],
)
def test_postcalibration_safety_gate_fails_closed(
    primary_valid_n: int,
    candidate_exists: bool,
    exclusion: dict[str, object],
    reason: str,
) -> None:
    with pytest.raises(RuntimeError, match=reason):
        migration.assert_safety_state(
            primary_valid_n=primary_valid_n,
            candidate_exists=candidate_exists,
            exclusion=exclusion,
        )


def test_postcalibration_migration_source_allowlist_matches_sealed_report() -> None:
    report = migration._load_json(migration.REPORT_PATH)
    assert report["allowed_source_transitions"] == {
        migration.RUNNER_RELATIVE_PATH: {
            "old_sha256": migration.OLD_RUNNER_SHA256,
            "new_sha256": migration.NEW_RUNNER_SHA256,
        }
    }


def test_move_binding_accepts_same_bytes_at_planned_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "archive" / "destination.json"
    source.write_bytes(b"sealed-evidence")
    expected = migration._binding(source)
    destination.parent.mkdir(parents=True)
    os.replace(source, destination)

    migration._validate_binding_at(destination, expected)

    destination.write_bytes(b"mutated-evidence")
    with pytest.raises(RuntimeError, match="MOVE_BINDING_DRIFT"):
        migration._validate_binding_at(destination, expected)
