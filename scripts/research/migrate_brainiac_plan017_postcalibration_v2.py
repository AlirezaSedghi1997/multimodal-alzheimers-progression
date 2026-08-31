"""Append-only source migration after Plan-017 calibration was sealed.

This utility is deliberately narrower than a general compatibility mechanism.
It accepts one exact runner-source transition, re-commits already-finalized
NIfTI/transform bytes under the new dependency closure, rebuilds only derived
provenance (timing contracts, execution profile, and the QC threshold binding),
and preserves every superseded artifact.  It never invokes preprocessing,
training, an outer loader, or the primary 1669-subject route.

The filesystem commit is journaled one move at a time.  Re-running ``--mode
migrate`` after interruption resumes the same transaction; unknown or drifting
state fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import migrate_brainiac_plan017_resume_v1 as v1  # noqa: E402
from scripts.research import prepare_brainiac_raw_t1_v2 as runner  # noqa: E402
from src.research import brainiac_preprocessing_v2 as v2  # noqa: E402
from src.research.brainiac_action_guard import inspect_action_exclusion  # noqa: E402
from src.research.brainiac_preprocessing_v2_schema import (  # noqa: E402
    FIXED_QC_LIMITS,
    validate_threshold_manifest,
)


OLD_DEPENDENCY_CLOSURE_SHA256 = (
    "83a2af4cc37e960cb8463e752c2dec673107e54523cf4214c31dd78a5d60c4bf"
)
RUNNER_RELATIVE_PATH = "scripts/research/prepare_brainiac_raw_t1_v2.py"
OLD_RUNNER_SHA256 = (
    "224985f6bd6fd00ea0cb1bb92bde2dcd8194a3521cf26432a0d14223d307691b"
)
NEW_RUNNER_SHA256 = (
    "3084e52fccec3e19bcd77331c324ee0c7ea606f142d1bd5e9f45fc5984b26c63"
)
OLD_PROFILE_INTEGRITY_SHA256 = (
    "36ae8d4d3bfb5cdbe96fa92c106315529a79c20a66bae8d10d390e35411af166"
)
OLD_THRESHOLD_FILE_SHA256 = (
    "0b1e98ef76603320c3cd1e2c1e82a4bcadf696a916461a22cd4b884411275010"
)
PARENT_MIGRATION_INTEGRITY_SHA256 = (
    "5add52117701528853ebc17f96c075230be6ed0bd5933afc41a606d315220d7e"
)

REPORT_PATH = runner.POSTCALIBRATION_SOURCE_MIGRATION_PATH
BLOCKED_PATH = (
    runner.REGISTRY_ROOT
    / "BLOCKED_BRAINIAC_PLAN017_POSTCALIBRATION_SOURCE_MIGRATION_V2.json"
)
TRANSACTION_PATH = (
    runner.REGISTRY_ROOT
    / "BRAINIAC_PLAN017_POSTCALIBRATION_SOURCE_MIGRATION_TRANSACTION_V2.json"
)
LEGACY_EQUIVALENCE_REPORT = (
    runner.REGISTRY_ROOT / "brainiac_preprocessing_v2_equivalence.json"
)
MIGRATION_ARTIFACT = "BRAINIAC_PLAN017_POSTCALIBRATION_SOURCE_MIGRATION_V2"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _load_json(path: Path) -> dict[str, Any]:
    return v1._load_json(path)


def _relative(path: Path) -> str:
    return v1._relative(path)


def _resolve_relative(value: str) -> Path:
    candidate = (ROOT / str(value)).resolve(strict=False)
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_PATH_OUTSIDE_PROJECT") from exc
    return candidate


def _binding(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_file():
        return {"kind": "file", **v1._file_binding(candidate)}
    if candidate.is_dir():
        return {"kind": "tree", **v1._tree_binding(candidate)}
    raise RuntimeError("BRAINIAC_POSTCALIBRATION_EVIDENCE_MISSING")


def _validate_binding_at(path: Path, expected: Mapping[str, Any]) -> None:
    observed = _binding(path)
    observed_content = {key: value for key, value in observed.items() if key != "path"}
    expected_content = {
        key: value for key, value in expected.items() if key != "path"
    }
    if observed_content != expected_content:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_MOVE_BINDING_DRIFT")


def assert_dependency_transition(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> None:
    """Allow exactly one runner source hop; all scientific inputs stay fixed."""

    if old.get("closure_sha256") != OLD_DEPENDENCY_CLOSURE_SHA256:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_OLD_CLOSURE_INVALID")
    for role in ("dynamic_code_inputs", "static_inputs"):
        if old.get(role) != new.get(role):
            raise RuntimeError("BRAINIAC_POSTCALIBRATION_SCIENTIFIC_INPUT_DRIFT")
    old_python = dict(old.get("python_closure") or {})
    new_python = dict(new.get("python_closure") or {})
    if set(old_python) != set(new_python):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_CLOSURE_FILE_SET_DRIFT")
    changed = {
        path
        for path in old_python
        if str(old_python[path]).casefold() != str(new_python[path]).casefold()
    }
    if changed != {RUNNER_RELATIVE_PATH}:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_UNAPPROVED_SOURCE_DRIFT")
    if (
        str(old_python[RUNNER_RELATIVE_PATH]).casefold() != OLD_RUNNER_SHA256
        or str(new_python[RUNNER_RELATIVE_PATH]).casefold() != NEW_RUNNER_SHA256
        or str(new.get("closure_sha256") or "").casefold()
        == OLD_DEPENDENCY_CLOSURE_SHA256
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_SOURCE_SHA_MISMATCH")


def assert_safety_state(
    *,
    primary_valid_n: int,
    candidate_exists: bool,
    exclusion: Mapping[str, Any],
) -> None:
    if int(primary_valid_n) != 0:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_PRIMARY_ALREADY_STARTED")
    if candidate_exists:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_LATER_STAGE_ALREADY_EXISTS")
    if exclusion.get("status") != "PASS" or exclusion.get("blockers"):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_WRITER_EXCLUSION_BLOCKED")


def threshold_scientific_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "receipt_integrity_sha256s",
        "contract_bindings",
        "pipeline_contract_sha256",
        "integrity_sha256",
    }
    return {
        key: value for key, value in manifest.items() if key not in excluded
    }


def _threshold_metrics_from_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        sample_hash: {
            "foreground_fraction": float(receipt["foreground_fraction"]),
            "normalized_nonzero_std": float(receipt["normalized_nonzero_std"]),
            "rotation_deg": float(receipt["rotation_deg"]),
            "translation_norm_mm": float(receipt["translation_norm_mm"]),
        }
        for sample_hash, receipt in sorted(receipts.items())
    }


def build_migrated_threshold_manifest(
    old_manifest: Mapping[str, Any],
    *,
    migrated_receipts: Mapping[str, Mapping[str, Any]],
    new_contract_bindings: Mapping[str, str],
) -> dict[str, Any]:
    migrated = v2.build_threshold_manifest(
        _threshold_metrics_from_receipts(migrated_receipts),
        receipt_integrity_sha256s={
            sample_hash: str(receipt["integrity_sha256"])
            for sample_hash, receipt in sorted(migrated_receipts.items())
        },
        contract_bindings=new_contract_bindings,
        fixed_bounds={
            "foreground_fraction": FIXED_QC_LIMITS["foreground_fraction"],
            "normalized_nonzero_std": FIXED_QC_LIMITS["normalized_nonzero_std"],
        },
    )
    if threshold_scientific_projection(migrated) != threshold_scientific_projection(
        old_manifest
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_THRESHOLD_NUMERIC_DRIFT")
    validate_threshold_manifest(
        migrated,
        expected_sample_hashes=sorted(migrated_receipts),
        expected_receipt_integrity_sha256s={
            sample_hash: str(receipt["integrity_sha256"])
            for sample_hash, receipt in sorted(migrated_receipts.items())
        },
        expected_contract_bindings=new_contract_bindings,
    )
    return migrated


def _load_root_receipts(
    root: Path,
    *,
    expected_n: int,
    expected_bindings: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    return v1._load_root_receipts(
        root,
        expected_n=expected_n,
        expected_bindings=expected_bindings,
    )


def _recompute_leg_at_root(
    root: Path,
    *,
    role: str,
    public_root: Path,
    expected_contract_sha256: str,
    expected_receipt_bindings: Mapping[str, str],
) -> dict[str, Any]:
    timing_path = root / runner.EXECUTION_LEG_TIMING_FILENAME
    timing = _load_json(timing_path)
    timing_body = dict(timing)
    timing_integrity = timing_body.pop("integrity_sha256", None)
    if (
        timing.get("schema_version") != 2
        or timing.get("artifact") != "BRAINIAC_EXECUTION_LEG_TIMING_V2"
        or timing.get("role") != role
        or timing_integrity != runner.canonical_sha(timing_body)
        or timing.get("contract_sha256") != expected_contract_sha256
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_TIMING_RECEIPT_INVALID")
    receipts = _load_root_receipts(
        root,
        expected_n=12,
        expected_bindings=expected_receipt_bindings,
    )
    sample_hashes = sorted(receipts)
    if (
        timing.get("sample_hashes") != sample_hashes
        or timing.get("selection_sha256") != runner.canonical_sha(sample_hashes)
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_TIMING_SELECTION_DRIFT")
    tree = runner._tree_evidence(root)
    return {
        "role": role,
        "root_relative_path": _relative(public_root),
        "root_tree_sha256": tree["tree_sha256"],
        "sample_hashes": sample_hashes,
        "selection_sha256": runner.canonical_sha(sample_hashes),
        "receipt_file_sha256_by_sample": {
            sample: runner.sha256_file(root / "receipts" / f"{sample}.json")
            for sample in sample_hashes
        },
        "output_sha256_by_sample": {
            sample: str(receipts[sample]["output_sha256"])
            for sample in sample_hashes
        },
        "transform_sha256_by_sample": {
            sample: str(receipts[sample]["transform_sha256"])
            for sample in sample_hashes
        },
        "canonical_scientific_sha256_by_sample": {
            sample: runner.canonical_sha(
                {
                    field: receipts[sample][field]
                    for field in runner.EXECUTION_PROFILE_SCIENTIFIC_FIELDS
                }
            )
            for sample in sample_hashes
        },
        "contract_sha256": expected_contract_sha256,
        "started_at_utc": timing["started_at_utc"],
        "finished_at_utc": timing["finished_at_utc"],
        "elapsed_seconds": float(timing["elapsed_seconds"]),
        "fresh_run": timing["fresh_run"],
        "resume_event_count": int(timing["resume_event_count"]),
        "timing_receipt_sha256": runner.sha256_file(timing_path),
        "integrity_status": "PASS",
        "live_evidence_status": "PASS_RECOMPUTED_FROM_LIVE_BYTES",
    }


def _profile_scientific_projection(profile: Mapping[str, Any]) -> dict[str, Any]:
    legs = profile.get("legs")
    sequential = legs.get("sequential") if isinstance(legs, Mapping) else None
    parallel = legs.get("parallel") if isinstance(legs, Mapping) else None
    if not isinstance(sequential, Mapping) or not isinstance(parallel, list):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_PROFILE_INVALID")
    return {
        "status": profile.get("status"),
        "execution_mode": profile.get("execution_mode"),
        "sample_count": profile.get("sample_count"),
        "parallel_repeat_count": profile.get("parallel_repeat_count"),
        "exact_artifact_and_scientific_equivalence": profile.get(
            "exact_artifact_and_scientific_equivalence"
        ),
        "median_speedup_fraction": profile.get("median_speedup_fraction"),
        "minimum_speedup_fraction": profile.get("minimum_speedup_fraction"),
        "timing_fresh": profile.get("timing_fresh"),
        "legs": [
            {
                "role": leg.get("role"),
                "sample_hashes": leg.get("sample_hashes"),
                "output_sha256_by_sample": leg.get("output_sha256_by_sample"),
                "transform_sha256_by_sample": leg.get("transform_sha256_by_sample"),
                "canonical_scientific_sha256_by_sample": leg.get(
                    "canonical_scientific_sha256_by_sample"
                ),
                "started_at_utc": leg.get("started_at_utc"),
                "finished_at_utc": leg.get("finished_at_utc"),
                "elapsed_seconds": leg.get("elapsed_seconds"),
                "fresh_run": leg.get("fresh_run"),
                "resume_event_count": leg.get("resume_event_count"),
            }
            for leg in [sequential, *parallel]
        ],
    }


def _validate_parent_migration() -> dict[str, Any]:
    report = runner._validated_source_only_resume_migration()
    if report.get("integrity_sha256") != PARENT_MIGRATION_INTEGRITY_SHA256:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_PARENT_MIGRATION_INVALID")
    return report


def _primary_receipt_count() -> int:
    receipts = v2.PRIMARY_ROOT / "receipts"
    return len(list(receipts.glob("*.json"))) if receipts.is_dir() else 0


def _preflight() -> dict[str, Any]:
    if REPORT_PATH.is_file():
        return {
            "status": "ALREADY_COMPLETED",
            "report": runner._validated_postcalibration_source_migration(),
        }
    if TRANSACTION_PATH.is_file():
        transaction = _validate_transaction(_load_json(TRANSACTION_PATH))
        return {"status": "RESUME_TRANSACTION", "transaction": transaction}

    v2.assert_no_project_gpu_writer()
    exclusion = inspect_action_exclusion()
    assert_safety_state(
        primary_valid_n=_primary_receipt_count(),
        candidate_exists=runner.ACCEPTANCE_CANDIDATE_PATH.exists()
        or runner.TRACEABILITY_MANIFEST_PATH.exists()
        or runner.READY_REPORT_PATH.exists(),
        exclusion=exclusion,
    )
    parent = _validate_parent_migration()
    old_dependency = _load_json(runner.DEPENDENCY_MANIFEST_PATH)
    new_dependency = runner._dependency_manifest()
    assert_dependency_transition(old_dependency, new_dependency)
    if runner.sha256_file(ROOT / RUNNER_RELATIVE_PATH) != NEW_RUNNER_SHA256:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_LIVE_RUNNER_SHA_INVALID")

    if runner.sha256_file(runner.THRESHOLD_MANIFEST_PATH) != OLD_THRESHOLD_FILE_SHA256:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_OLD_THRESHOLD_INVALID")
    old_threshold = _load_json(runner.THRESHOLD_MANIFEST_PATH)
    old_profile = _load_json(runner.EXECUTION_PROFILE_PATH)
    if old_profile.get("integrity_sha256") != OLD_PROFILE_INTEGRITY_SHA256:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_OLD_PROFILE_INVALID")

    new_bindings = runner._equivalence_receipt_bindings()
    if new_bindings.get("dependency_closure_sha256") != new_dependency.get(
        "closure_sha256"
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_NEW_BINDINGS_INVALID")
    first = next(
        iter(
            sorted(
                (
                    v2.SMOKE_ROOT
                    / "equivalence_v2"
                    / "sequential"
                    / "receipts"
                ).glob("*.json")
            )
        ),
        None,
    )
    if first is None:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_EQUIVALENCE_MISSING")
    old_bindings = v1._binding_view(_load_json(first), set(new_bindings))
    if old_bindings.get("dependency_closure_sha256") != OLD_DEPENDENCY_CLOSURE_SHA256:
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_OLD_RECEIPT_CLOSURE_INVALID")
    for role in new_bindings:
        if role != "dependency_closure_sha256" and (
            str(old_bindings[role]).casefold()
            != str(new_bindings[role]).casefold()
        ):
            raise RuntimeError("BRAINIAC_POSTCALIBRATION_NON_SOURCE_BINDING_DRIFT")

    old_contract = runner.canonical_sha(
        runner._threshold_contract_bindings(old_bindings)
    )
    runner.validate_execution_profile(
        old_profile,
        expected_contract_sha256=old_contract,
        expected_receipt_bindings=old_bindings,
    )
    equivalence_root = v2.SMOKE_ROOT / "equivalence_v2"
    legs = {
        role: _load_root_receipts(
            equivalence_root / role,
            expected_n=12,
            expected_bindings=old_bindings,
        )
        for role in ("sequential", "parallel_1", "parallel_2")
    }
    calibration = _load_root_receipts(
        runner.CALIBRATION_ROOT,
        expected_n=32,
        expected_bindings=old_bindings,
    )
    selected = v2.select_calibration_samples(
        [row["sample_hash"] for row in v2.load_locked_membership()]
    )
    if sorted(calibration) != sorted(selected):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_CALIBRATION_SELECTION_DRIFT")
    validate_threshold_manifest(
        old_threshold,
        expected_sample_hashes=selected,
        expected_receipt_integrity_sha256s={
            sample: str(receipt["integrity_sha256"])
            for sample, receipt in sorted(calibration.items())
        },
        expected_contract_bindings=runner._threshold_contract_bindings(old_bindings),
    )
    return {
        "status": "PASS",
        "parent": parent,
        "old_dependency": old_dependency,
        "new_dependency": new_dependency,
        "old_bindings": old_bindings,
        "new_bindings": new_bindings,
        "old_threshold": old_threshold,
        "old_profile": old_profile,
        "equivalence_root": equivalence_root,
        "legs": legs,
        "calibration": calibration,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    runner._atomic_write_json(path, dict(payload))


def _seal_transaction(body: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(body)
    sealed.pop("integrity_sha256", None)
    sealed["integrity_sha256"] = runner.canonical_sha(sealed)
    return sealed


def _validate_transaction(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    moves = payload.get("moves")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact")
        != "BRAINIAC_PLAN017_POSTCALIBRATION_MIGRATION_TRANSACTION_V2"
        or recorded != runner.canonical_sha(body)
        or payload.get("new_dependency_closure_sha256")
        != runner._dependency_manifest().get("closure_sha256")
        or not isinstance(moves, list)
        or len(moves) != 12
        or len({str(row.get("role")) for row in moves if isinstance(row, Mapping)})
        != 12
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_TRANSACTION_INVALID")
    return dict(payload)


def _update_transaction(transaction: dict[str, Any], **changes: Any) -> dict[str, Any]:
    transaction.update(changes)
    transaction = _seal_transaction(transaction)
    _write_json(TRANSACTION_PATH, transaction)
    return transaction


def _move_plan_row(
    role: str,
    phase: str,
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    return {
        "role": role,
        "phase": phase,
        "source": _relative(source),
        "destination": _relative(destination),
        "expected_binding": _binding(source),
        "completed": False,
    }


def _execute_move_phase(
    transaction: dict[str, Any], phase: str
) -> dict[str, Any]:
    for index, raw in enumerate(list(transaction["moves"])):
        row = dict(raw)
        if row.get("phase") != phase or row.get("completed") is True:
            continue
        source = _resolve_relative(str(row["source"]))
        destination = _resolve_relative(str(row["destination"]))
        expected = dict(row["expected_binding"])
        if source.exists() and not destination.exists():
            _validate_binding_at(source, expected)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        elif not source.exists() and destination.exists():
            pass
        else:
            raise RuntimeError("BRAINIAC_POSTCALIBRATION_MOVE_STATE_AMBIGUOUS")
        _validate_binding_at(destination, expected)
        row["completed"] = True
        row["completed_at_utc"] = _utc_now()
        transaction["moves"][index] = row
        transaction = _update_transaction(
            transaction,
            phase=phase,
            last_completed_move=str(row["role"]),
        )
    return transaction


def _stage(state: Mapping[str, Any]) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    nonce = f"{stamp}_{time.time_ns()}"
    staging = (
        ROOT
        / "data"
        / "derived"
        / "method_x_brainiac"
        / f".plan017_postcalibration_migration_staging_{nonce}"
    )
    staging.mkdir(parents=True, exist_ok=False)
    staged_equivalence = staging / "equivalence_v2"
    staged_calibration = staging / "calibration"
    new_contract = runner.canonical_sha(
        runner._threshold_contract_bindings(state["new_bindings"])
    )
    migrated_legs: dict[str, dict[str, dict[str, Any]]] = {}
    for role in ("sequential", "parallel_1", "parallel_2"):
        old_root = state["equivalence_root"] / role
        new_root = staged_equivalence / role
        migrated_legs[role] = v1.migrate_receipt_root(
            old_root,
            new_root,
            source_receipts=state["legs"][role],
            new_bindings=state["new_bindings"],
        )
        old_timing = _load_json(old_root / runner.EXECUTION_LEG_TIMING_FILENAME)
        runner.write_execution_leg_timing_receipt(
            new_root,
            role=role,
            sample_hashes=sorted(migrated_legs[role]),
            contract_sha256=new_contract,
            started_at_utc=str(old_timing["started_at_utc"]),
            finished_at_utc=str(old_timing["finished_at_utc"]),
            elapsed_seconds=float(old_timing["elapsed_seconds"]),
            fresh_run=bool(old_timing["fresh_run"]),
            resume_event_count=int(old_timing["resume_event_count"]),
        )
    migrated_calibration = v1.migrate_receipt_root(
        runner.CALIBRATION_ROOT,
        staged_calibration,
        source_receipts=state["calibration"],
        new_bindings=state["new_bindings"],
    )
    new_threshold = build_migrated_threshold_manifest(
        state["old_threshold"],
        migrated_receipts=migrated_calibration,
        new_contract_bindings=runner._threshold_contract_bindings(
            state["new_bindings"]
        ),
    )
    staged_results = staging / "results"
    staged_results.mkdir(parents=True, exist_ok=True)
    staged_threshold = staging / "qc_threshold_manifest.json"
    staged_dependency = staged_results / runner.DEPENDENCY_MANIFEST_PATH.name
    staged_profile = staged_results / runner.EXECUTION_PROFILE_PATH.name
    staged_legacy = staged_results / LEGACY_EQUIVALENCE_REPORT.name
    _write_json(staged_threshold, new_threshold)
    _write_json(staged_dependency, state["new_dependency"])

    public_roots = runner._execution_profile_roots()
    recomputed = {
        role: _recompute_leg_at_root(
            staged_equivalence / role,
            role=role,
            public_root=public_roots[role],
            expected_contract_sha256=new_contract,
            expected_receipt_bindings=state["new_bindings"],
        )
        for role in ("sequential", "parallel_1", "parallel_2")
    }
    profile = runner.build_execution_profile(
        sequential=recomputed["sequential"],
        parallel_runs=[recomputed["parallel_1"], recomputed["parallel_2"]],
        expected_contract_sha256=new_contract,
    )
    profile["created_at_utc"] = _utc_now()
    profile["claim_ceiling"] = "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
    profile["integrity_sha256"] = runner.canonical_sha(
        {key: value for key, value in profile.items() if key != "integrity_sha256"}
    )
    if _profile_scientific_projection(profile) != _profile_scientific_projection(
        state["old_profile"]
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_EXECUTION_PROFILE_DRIFT")
    _write_json(staged_profile, profile)
    _write_json(staged_legacy, profile)

    data_archive = (
        ROOT
        / "data"
        / "derived"
        / "method_x_brainiac"
        / "superseded"
        / f"plan017_postcalibration_source_migration_{nonce}"
    )
    result_archive = (
        runner.REGISTRY_ROOT
        / "superseded"
        / f"plan017_postcalibration_source_migration_{nonce}"
    )
    data_archive.mkdir(parents=True, exist_ok=False)
    result_archive.mkdir(parents=True, exist_ok=False)
    moves = [
        _move_plan_row(
            "archive_equivalence",
            "ARCHIVE_OLD",
            state["equivalence_root"],
            data_archive / "old_equivalence_v2",
        ),
        _move_plan_row(
            "archive_calibration",
            "ARCHIVE_OLD",
            runner.CALIBRATION_ROOT,
            data_archive / "old_calibration",
        ),
        _move_plan_row(
            "archive_threshold",
            "ARCHIVE_OLD",
            runner.THRESHOLD_MANIFEST_PATH,
            data_archive / "old_qc_threshold_manifest.json",
        ),
        _move_plan_row(
            "archive_profile",
            "ARCHIVE_OLD",
            runner.EXECUTION_PROFILE_PATH,
            result_archive / "old_execution_profile.json",
        ),
        _move_plan_row(
            "archive_dependency",
            "ARCHIVE_OLD",
            runner.DEPENDENCY_MANIFEST_PATH,
            result_archive / "old_dependency_manifest.json",
        ),
        _move_plan_row(
            "archive_legacy_profile",
            "ARCHIVE_OLD",
            LEGACY_EQUIVALENCE_REPORT,
            result_archive / "old_equivalence_report.json",
        ),
        _move_plan_row(
            "install_equivalence",
            "INSTALL_NEW",
            staged_equivalence,
            state["equivalence_root"],
        ),
        _move_plan_row(
            "install_calibration",
            "INSTALL_NEW",
            staged_calibration,
            runner.CALIBRATION_ROOT,
        ),
        _move_plan_row(
            "install_threshold",
            "INSTALL_NEW",
            staged_threshold,
            runner.THRESHOLD_MANIFEST_PATH,
        ),
        _move_plan_row(
            "install_profile",
            "INSTALL_NEW",
            staged_profile,
            runner.EXECUTION_PROFILE_PATH,
        ),
        _move_plan_row(
            "install_dependency",
            "INSTALL_NEW",
            staged_dependency,
            runner.DEPENDENCY_MANIFEST_PATH,
        ),
        _move_plan_row(
            "install_legacy_profile",
            "INSTALL_NEW",
            staged_legacy,
            LEGACY_EQUIVALENCE_REPORT,
        ),
    ]
    transaction = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PLAN017_POSTCALIBRATION_MIGRATION_TRANSACTION_V2",
        "status": "IN_PROGRESS",
        "phase": "PREPARED",
        "created_at_utc": _utc_now(),
        "nonce": nonce,
        "staging_root": _relative(staging),
        "data_archive": _relative(data_archive),
        "result_archive": _relative(result_archive),
        "old_dependency_closure_sha256": OLD_DEPENDENCY_CLOSURE_SHA256,
        "new_dependency_closure_sha256": state["new_dependency"]["closure_sha256"],
        "old_profile_integrity_sha256": OLD_PROFILE_INTEGRITY_SHA256,
        "new_profile_integrity_sha256": profile["integrity_sha256"],
        "old_threshold_file_sha256": OLD_THRESHOLD_FILE_SHA256,
        "new_threshold_file_sha256": runner.sha256_file(staged_threshold),
        "moves": moves,
        "primary_valid_n": 0,
        "preprocessing_reexecuted": False,
        "manual_override_allowed": False,
    }
    transaction = _seal_transaction(transaction)
    _write_json(TRANSACTION_PATH, transaction)
    return transaction


def _validate_new_live(transaction: Mapping[str, Any]) -> dict[str, Any]:
    dependency = _load_json(runner.DEPENDENCY_MANIFEST_PATH)
    if dependency != runner._dependency_manifest():
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_DEPENDENCY_INSTALL_INVALID")
    bindings = runner._equivalence_receipt_bindings()
    profile = _load_json(runner.EXECUTION_PROFILE_PATH)
    checked = runner.validate_execution_profile(
        profile,
        expected_contract_sha256=runner._current_equivalence_contract_sha256(),
        expected_receipt_bindings=bindings,
    )
    calibration = _load_root_receipts(
        runner.CALIBRATION_ROOT,
        expected_n=32,
        expected_bindings=bindings,
    )
    threshold = _load_json(runner.THRESHOLD_MANIFEST_PATH)
    validate_threshold_manifest(
        threshold,
        expected_sample_hashes=sorted(calibration),
        expected_receipt_integrity_sha256s={
            sample: str(receipt["integrity_sha256"])
            for sample, receipt in sorted(calibration.items())
        },
        expected_contract_bindings=runner._threshold_contract_bindings(bindings),
    )
    if runner.sha256_file(runner.THRESHOLD_MANIFEST_PATH) != transaction.get(
        "new_threshold_file_sha256"
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_THRESHOLD_INSTALL_INVALID")
    if checked.get("integrity_sha256") != transaction.get(
        "new_profile_integrity_sha256"
    ):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_PROFILE_INSTALL_INVALID")
    assert_safety_state(
        primary_valid_n=_primary_receipt_count(),
        candidate_exists=runner.ACCEPTANCE_CANDIDATE_PATH.exists()
        or runner.TRACEABILITY_MANIFEST_PATH.exists()
        or runner.READY_REPORT_PATH.exists(),
        exclusion=inspect_action_exclusion(),
    )
    return {
        "dependency": dependency,
        "bindings": bindings,
        "profile": checked,
        "threshold": threshold,
        "calibration": calibration,
    }


def _archive_prior_blocker() -> dict[str, Any] | None:
    if not BLOCKED_PATH.is_file():
        return None
    payload = _load_json(BLOCKED_PATH)
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if payload.get("status") != "BLOCKED" or recorded != runner.canonical_sha(body):
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_BLOCKER_INVALID")
    digest = runner.sha256_file(BLOCKED_PATH)
    target = (
        runner.REGISTRY_ROOT
        / "superseded"
        / "plan017_postcalibration_blockers"
        / f"sha256-{digest}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if runner.sha256_file(target) != digest:
            raise RuntimeError("BRAINIAC_POSTCALIBRATION_BLOCKER_ARCHIVE_COLLISION")
        raise RuntimeError("BRAINIAC_POSTCALIBRATION_BLOCKER_DUPLICATE_LIVE_FILE")
    os.replace(BLOCKED_PATH, target)
    return runner._artifact_binding(target)


def _final_report(
    transaction: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    superseded_blocker: Mapping[str, Any] | None,
) -> dict[str, Any]:
    data_archive = _resolve_relative(str(transaction["data_archive"]))
    result_archive = _resolve_relative(str(transaction["result_archive"]))
    profile = live["profile"]
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": MIGRATION_ARTIFACT,
        "status": "PASS_POSTCALIBRATION_SOURCE_ONLY_MIGRATION",
        "created_at_utc": _utc_now(),
        "transaction_nonce": transaction["nonce"],
        "parent_migration": runner._artifact_binding(
            runner.SOURCE_ONLY_RESUME_MIGRATION_PATH
        ),
        "old_dependency_closure_sha256": OLD_DEPENDENCY_CLOSURE_SHA256,
        "new_dependency_closure_sha256": live["dependency"]["closure_sha256"],
        "allowed_source_transitions": {
            RUNNER_RELATIVE_PATH: {
                "old_sha256": OLD_RUNNER_SHA256,
                "new_sha256": NEW_RUNNER_SHA256,
            }
        },
        "scientific_input_drift_count": 0,
        "output_byte_drift_count": 0,
        "transform_byte_drift_count": 0,
        "scientific_receipt_field_drift_count": 0,
        "threshold_numeric_drift_count": 0,
        "preprocessing_reexecuted": False,
        "outer_loader_invocation_count": 0,
        "migrated_equivalence_receipts": 36,
        "migrated_calibration_receipts": 32,
        "threshold_manifest_migrated": True,
        "old_profile_integrity_sha256": OLD_PROFILE_INTEGRITY_SHA256,
        "new_profile_integrity_sha256": profile["integrity_sha256"],
        "old_threshold_file_sha256": OLD_THRESHOLD_FILE_SHA256,
        "new_threshold_file_sha256": runner.sha256_file(
            runner.THRESHOLD_MANIFEST_PATH
        ),
        "new_live_bindings": {
            "equivalence": v1._tree_binding(v2.SMOKE_ROOT / "equivalence_v2"),
            "calibration": v1._tree_binding(runner.CALIBRATION_ROOT),
            "threshold": v1._file_binding(runner.THRESHOLD_MANIFEST_PATH),
            "execution_profile": v1._file_binding(runner.EXECUTION_PROFILE_PATH),
            "dependency_manifest": v1._file_binding(runner.DEPENDENCY_MANIFEST_PATH),
        },
        "archive_bindings": {
            "data": v1._tree_binding(data_archive),
            "results": v1._tree_binding(result_archive),
        },
        "execution_profile": {
            "status": profile["status"],
            "execution_mode": profile["execution_mode"],
            "exact_artifact_and_scientific_equivalence": profile[
                "exact_artifact_and_scientific_equivalence"
            ],
            "timing_fresh": profile["timing_fresh"],
            "median_speedup_fraction": profile["median_speedup_fraction"],
            "integrity_sha256": profile["integrity_sha256"],
        },
        "primary_valid_n": 0,
        "plans_005_006_007_allowed": False,
        "training_authorization": False,
        "manual_override_allowed": False,
        "human_approval_required": False,
        "unknown_is_pass": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    if superseded_blocker is not None:
        body["superseded_blocker"] = dict(superseded_blocker)
    body["integrity_sha256"] = runner.canonical_sha(body)
    return body


def migrate() -> dict[str, Any]:
    state = _preflight()
    if state["status"] == "ALREADY_COMPLETED":
        return dict(state["report"])
    if state["status"] == "PASS":
        transaction = _stage(state)
    else:
        transaction = dict(state["transaction"])
    transaction = _validate_transaction(transaction)
    transaction = _execute_move_phase(transaction, "ARCHIVE_OLD")
    transaction = _update_transaction(transaction, phase="ARCHIVED_OLD")
    transaction = _execute_move_phase(transaction, "INSTALL_NEW")
    transaction = _update_transaction(transaction, phase="INSTALLED_NEW")
    live = _validate_new_live(transaction)
    transaction = _update_transaction(transaction, phase="LIVE_VALIDATED")
    superseded_blocker = _archive_prior_blocker()
    report = _final_report(
        transaction, live, superseded_blocker=superseded_blocker
    )
    _write_json(REPORT_PATH, report)
    runner._validated_postcalibration_source_migration()
    _update_transaction(transaction, phase="SEALED", status="COMPLETED")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "migrate"), required=True)
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            state = _preflight()
            if state["status"] == "ALREADY_COMPLETED":
                report = dict(state["report"])
            elif state["status"] == "RESUME_TRANSACTION":
                report = {
                    "schema_version": 1,
                    "status": "PASS_RESUMABLE_TRANSACTION",
                    "phase": state["transaction"]["phase"],
                    "new_dependency_closure_sha256": state["transaction"][
                        "new_dependency_closure_sha256"
                    ],
                    "primary_valid_n": 0,
                    "manual_override_allowed": False,
                }
                report["integrity_sha256"] = runner.canonical_sha(report)
            else:
                report = {
                    "schema_version": 1,
                    "status": "PASS_POSTCALIBRATION_SOURCE_MIGRATION_PREFLIGHT",
                    "old_dependency_closure_sha256": state["old_dependency"][
                        "closure_sha256"
                    ],
                    "new_dependency_closure_sha256": state["new_dependency"][
                        "closure_sha256"
                    ],
                    "equivalence_receipt_count": sum(
                        len(rows) for rows in state["legs"].values()
                    ),
                    "calibration_receipt_count": len(state["calibration"]),
                    "threshold_manifest_valid": True,
                    "execution_profile_valid": True,
                    "scientific_input_drift_count": 0,
                    "primary_valid_n": 0,
                    "manual_override_allowed": False,
                }
                report["integrity_sha256"] = runner.canonical_sha(report)
        else:
            report = migrate()
    except Exception as exc:
        body: dict[str, Any] = {
            "schema_version": 1,
            "status": "BLOCKED",
            "reason_code": runner._sanitize_exception_message(exc),
            "created_at_utc": _utc_now(),
            "primary_valid_n": _primary_receipt_count(),
            "manual_override_allowed": False,
        }
        body["integrity_sha256"] = runner.canonical_sha(body)
        _write_json(BLOCKED_PATH, body)
        print(json.dumps(body, sort_keys=True))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
