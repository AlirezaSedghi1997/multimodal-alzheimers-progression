"""Audited source-only migration for the interrupted Plan-017 calibration run.

The scientific preprocessing bytes are never recomputed by this utility.  It
replays already committed output/transform bytes through the production
transaction writer so receipts and journals bind to the repaired orchestration
source closure.  The transition is deliberately one-shot and exact-hash
allow-listed; any other source, policy, config, scientific field, or byte drift
fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import prepare_brainiac_raw_t1_v2 as runner  # noqa: E402
from src.research import brainiac_preprocessing_v2 as v2  # noqa: E402
from src.research.brainiac_action_guard import inspect_action_exclusion  # noqa: E402


OLD_DEPENDENCY_CLOSURE_SHA256 = (
    "6de860893a2ab79f33974c338ad02df9d4d090be833b900b7555c30c14e8a371"
)
OLD_PROFILE_INTEGRITY_SHA256 = (
    "db5902e8a0814729f52c9998401dd9e3a0804f197c31fedaaf9a3379b76b42df"
)
ALLOWED_SOURCE_TRANSITIONS = {
    "scripts/research/prepare_brainiac_raw_t1_v2.py": (
        "2c5956ed9f6afe1a87fd6c56953aebd488bb43ad8a55ef8d55adfb71552cdc77",
        "e654ee103bcd42f99f5d9ea9683f31026acfccd7328162ea5c0279e2c897f1a1",
    ),
    "src/research/brainiac_preprocessing_v2.py": (
        "3c9e00c7bf111beccc24f1a2f66536fea6a341ef31828b4eaaf74d62f4cad3f4",
        "b25d9b717af9e6fc243fd140e3f8f634fbf8b5785b449127b81f3b41cee28d0f",
    ),
}
MIGRATION_REPORT = (
    runner.REGISTRY_ROOT / "BRAINIAC_PLAN017_SOURCE_ONLY_RESUME_MIGRATION_V1.json"
)
BLOCKED_REPORT = (
    runner.REGISTRY_ROOT / "BLOCKED_BRAINIAC_PLAN017_SOURCE_ONLY_RESUME_MIGRATION_V1.json"
)
PLACEHOLDER_THRESHOLD_SHA256 = hashlib.sha256(
    b"NOT_YET_CALIBRATED_PLAN_003"
).hexdigest()
NONSCIENTIFIC_RECEIPT_FIELDS = {
    "dependency_closure_sha256",
    "integrity_sha256",
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_JSON_INVALID")
    return payload


def _relative(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_PATH_OUTSIDE_PROJECT") from exc


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or v2._is_reparse(resolved):
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_FILE_UNSAFE")
    return {
        "path": _relative(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": runner.sha256_file(resolved),
    }


def _tree_binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), **runner._tree_evidence(path)}


def _receipt_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key not in NONSCIENTIFIC_RECEIPT_FIELDS
    }


def _binding_view(receipt: Mapping[str, Any], keys: set[str]) -> dict[str, str]:
    values = {key: str(receipt.get(key) or "") for key in sorted(keys)}
    if any(len(value) != 64 for value in values.values()):
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_RECEIPT_BINDING_INVALID")
    return values


def _load_root_receipts(
    root: Path,
    *,
    expected_n: int,
    expected_bindings: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    paths = sorted((root / "receipts").glob("*.json"))
    if len(paths) != expected_n:
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_RECEIPT_COUNT_INVALID")
    receipts: dict[str, dict[str, Any]] = {}
    for path in paths:
        receipt = v2.load_committed_receipt(
            path.stem,
            root,
            expected_bindings=expected_bindings,
        )
        if str(receipt["sample_hash"]) != path.stem or path.stem in receipts:
            raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_SAMPLE_SET_INVALID")
        receipts[path.stem] = receipt
    return receipts


def _assert_dependency_transition(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> None:
    if old.get("closure_sha256") != OLD_DEPENDENCY_CLOSURE_SHA256:
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_OLD_CLOSURE_INVALID")
    for role in ("dynamic_code_inputs", "static_inputs"):
        if old.get(role) != new.get(role):
            raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_SCIENTIFIC_INPUT_DRIFT")
    old_python = dict(old.get("python_closure") or {})
    new_python = dict(new.get("python_closure") or {})
    if set(old_python) != set(new_python):
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_CLOSURE_FILE_SET_DRIFT")
    changed = {
        path
        for path in old_python
        if str(old_python[path]).casefold() != str(new_python[path]).casefold()
    }
    if changed != set(ALLOWED_SOURCE_TRANSITIONS):
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_UNAPPROVED_SOURCE_DRIFT")
    for path, transition in ALLOWED_SOURCE_TRANSITIONS.items():
        if (old_python[path], new_python[path]) != transition:
            raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_SOURCE_SHA_MISMATCH")


def _copy_pipeline(
    source_root: Path, receipt: Mapping[str, Any]
):
    def pipeline(_item: Mapping[str, str], scratch: Path) -> dict[str, Any]:
        output = scratch / "output.nii.gz"
        transform = scratch / "transform.tfm"
        shutil.copy2(source_root / str(receipt["output_relative_path"]), output)
        shutil.copy2(source_root / str(receipt["transform_relative_path"]), transform)
        result = {
            "output_path": output,
            "transform_path": transform,
            "source_sha256": receipt["source_sha256"],
            "source_size": receipt["source_size"],
            "finite": receipt["finite"],
        }
        for field in (
            "affine_before",
            "affine_after",
            "spacing_before",
            "spacing_after",
            "origin_before",
            "origin_after",
            "direction_before",
            "direction_after",
            "com_coordinate_frame",
            "rigid_determinant",
            "orthogonality_error",
            "rotation_deg",
            "translation_norm_mm",
            "foreground_fraction",
            "normalized_nonzero_mean",
            "normalized_nonzero_std",
            "center_of_mass_before",
            "center_of_mass_after",
            "registration_similarity_before",
            "registration_similarity_after",
        ):
            result[field] = receipt[field]
        return result

    return pipeline


def migrate_receipt_root(
    source_root: Path,
    destination_root: Path,
    *,
    source_receipts: Mapping[str, Mapping[str, Any]],
    new_bindings: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Recommit exact bytes under new bindings; reject any scientific drift."""

    if destination_root.exists():
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_DESTINATION_EXISTS")
    migrated: dict[str, dict[str, Any]] = {}
    for sample_hash in sorted(source_receipts):
        old = dict(source_receipts[sample_hash])
        new = v2.run_sample_transaction(
            {"sample_hash": sample_hash, "t1_file_hash": "HASH_ONLY_MIGRATION"},
            destination_root,
            pipeline=_copy_pipeline(source_root, old),
            bindings=new_bindings,
            scratch_base=v2.LOCAL_SCRATCH_ROOT,
        )
        if _receipt_projection(old) != _receipt_projection(new):
            raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_SCIENTIFIC_DRIFT")
        if old["output_sha256"] != new["output_sha256"] or old[
            "transform_sha256"
        ] != new["transform_sha256"]:
            raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_BYTE_DRIFT")
        migrated[sample_hash] = new
    return migrated


def _preflight(calibration_source: Path) -> dict[str, Any]:
    if MIGRATION_REPORT.is_file():
        report = _load_json(MIGRATION_REPORT)
        body = dict(report)
        recorded = body.pop("integrity_sha256", None)
        if (
            report.get("status") != "PASS_SOURCE_ONLY_RUNTIME_RESUME_MIGRATION"
            or recorded != runner.canonical_sha(body)
        ):
            raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_EXISTING_REPORT_INVALID")
        return {"status": "ALREADY_COMPLETED", "report": report}
    v2.assert_no_project_gpu_writer()
    exclusion = inspect_action_exclusion()
    if exclusion.get("status") != "PASS" or exclusion.get("blockers"):
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_WRITER_EXCLUSION_BLOCKED")
    if not calibration_source.is_dir() or v2._is_reparse(calibration_source):
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_CALIBRATION_SOURCE_UNSAFE")
    expected_prefix = (
        ROOT / "data" / "derived" / "method_x_brainiac" / "superseded"
    ).resolve()
    try:
        calibration_source.resolve(strict=True).relative_to(expected_prefix)
    except ValueError as exc:
        raise RuntimeError(
            "BRAINIAC_SOURCE_MIGRATION_CALIBRATION_SOURCE_OUTSIDE_SUPERSEDED"
        ) from exc
    if runner.THRESHOLD_MANIFEST_PATH.exists() or runner.ACCEPTANCE_CANDIDATE_PATH.exists():
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_LATER_STAGE_ALREADY_EXISTS")
    primary_n = len(list((v2.PRIMARY_ROOT / "receipts").glob("*.json")))
    if primary_n != 0:
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_PRIMARY_ALREADY_STARTED")
    old_dependency = _load_json(runner.DEPENDENCY_MANIFEST_PATH)
    new_dependency = runner._dependency_manifest()
    _assert_dependency_transition(old_dependency, new_dependency)
    new_bindings = runner._base_bindings()
    if new_bindings["dependency_closure_sha256"] != new_dependency["closure_sha256"]:
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_NEW_CLOSURE_BINDING_INVALID")
    old_profile = _load_json(runner.EXECUTION_PROFILE_PATH)
    if old_profile.get("integrity_sha256") != OLD_PROFILE_INTEGRITY_SHA256:
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_OLD_PROFILE_INVALID")
    equivalence_root = v2.SMOKE_ROOT / "equivalence_v2"
    first_receipt_path = next(
        iter(sorted((equivalence_root / "sequential" / "receipts").glob("*.json"))),
        None,
    )
    if first_receipt_path is None:
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_EQUIVALENCE_MISSING")
    first_receipt = _load_json(first_receipt_path)
    old_bindings = _binding_view(first_receipt, set(new_bindings))
    if old_bindings["dependency_closure_sha256"] != OLD_DEPENDENCY_CLOSURE_SHA256:
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_OLD_RECEIPT_CLOSURE_INVALID")
    for key in new_bindings:
        if key != "dependency_closure_sha256" and old_bindings[key] != new_bindings[key]:
            raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_NON_SOURCE_BINDING_DRIFT")
    old_contract = runner.canonical_sha(runner._threshold_contract_bindings(old_bindings))
    runner.validate_execution_profile(
        old_profile,
        expected_contract_sha256=old_contract,
        expected_receipt_bindings=old_bindings,
    )
    legs = {
        role: _load_root_receipts(
            equivalence_root / role,
            expected_n=12,
            expected_bindings=old_bindings,
        )
        for role in ("sequential", "parallel_1", "parallel_2")
    }
    calibration_receipts = _load_root_receipts(
        calibration_source,
        expected_n=4,
        expected_bindings=old_bindings,
    )
    live_calibration = _load_root_receipts(
        runner.CALIBRATION_ROOT,
        expected_n=1,
        expected_bindings=old_bindings,
    )
    overlap = set(calibration_receipts) & set(live_calibration)
    if len(overlap) != 1 or any(
        _receipt_projection(calibration_receipts[sample])
        != _receipt_projection(live_calibration[sample])
        for sample in overlap
    ):
        raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_CALIBRATION_OVERLAP_DRIFT")
    return {
        "status": "PASS",
        "old_dependency": old_dependency,
        "new_dependency": new_dependency,
        "old_bindings": old_bindings,
        "new_bindings": new_bindings,
        "old_profile": old_profile,
        "equivalence_root": equivalence_root,
        "legs": legs,
        "calibration_receipts": calibration_receipts,
        "live_calibration": live_calibration,
        "overlap": sorted(overlap),
    }


def migrate(calibration_source: Path) -> dict[str, Any]:
    state = _preflight(calibration_source)
    if state["status"] == "ALREADY_COMPLETED":
        return dict(state["report"])
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    nonce = f"{stamp}_{time.time_ns()}"
    staging = (
        ROOT
        / "data"
        / "derived"
        / "method_x_brainiac"
        / f".plan017_source_migration_staging_{nonce}"
    )
    staging.mkdir(parents=True, exist_ok=False)
    staged_equivalence = staging / "equivalence_v2"
    staged_calibration = staging / "calibration"
    migrated_legs: dict[str, dict[str, dict[str, Any]]] = {}
    new_contract = runner.canonical_sha(
        runner._threshold_contract_bindings(state["new_bindings"])
    )
    try:
        for role in ("sequential", "parallel_1", "parallel_2"):
            source_root = state["equivalence_root"] / role
            destination = staged_equivalence / role
            migrated_legs[role] = migrate_receipt_root(
                source_root,
                destination,
                source_receipts=state["legs"][role],
                new_bindings=state["new_bindings"],
            )
            old_timing = _load_json(source_root / runner.EXECUTION_LEG_TIMING_FILENAME)
            runner.write_execution_leg_timing_receipt(
                destination,
                role=role,
                sample_hashes=sorted(migrated_legs[role]),
                contract_sha256=new_contract,
                started_at_utc=str(old_timing["started_at_utc"]),
                finished_at_utc=str(old_timing["finished_at_utc"]),
                elapsed_seconds=float(old_timing["elapsed_seconds"]),
                fresh_run=bool(old_timing["fresh_run"]),
                resume_event_count=int(old_timing["resume_event_count"]),
            )
        migrated_calibration = migrate_receipt_root(
            calibration_source,
            staged_calibration,
            source_receipts=state["calibration_receipts"],
            new_bindings=state["new_bindings"],
        )
        staged_bindings = {
            "equivalence": _tree_binding(staged_equivalence),
            "calibration": _tree_binding(staged_calibration),
        }
        data_archive = (
            ROOT
            / "data"
            / "derived"
            / "method_x_brainiac"
            / "superseded"
            / f"plan017_source_only_resume_migration_{nonce}"
        )
        result_archive = runner.REGISTRY_ROOT / "superseded" / (
            f"plan017_source_only_resume_migration_{nonce}"
        )
        data_archive.mkdir(parents=True, exist_ok=False)
        result_archive.mkdir(parents=True, exist_ok=False)
        old_live = {
            "equivalence": state["equivalence_root"],
            "calibration": runner.CALIBRATION_ROOT,
        }
        old_bindings = {role: _tree_binding(path) for role, path in old_live.items()}
        os.replace(old_live["equivalence"], data_archive / "old_equivalence_v2")
        os.replace(old_live["calibration"], data_archive / "old_live_calibration")
        os.replace(runner.EXECUTION_PROFILE_PATH, result_archive / "old_execution_profile.json")
        legacy_report = runner.REGISTRY_ROOT / "brainiac_preprocessing_v2_equivalence.json"
        if legacy_report.is_file():
            os.replace(legacy_report, result_archive / "old_equivalence_report.json")
        os.replace(runner.DEPENDENCY_MANIFEST_PATH, result_archive / "old_dependency_manifest.json")
        os.replace(staged_equivalence, state["equivalence_root"])
        os.replace(staged_calibration, runner.CALIBRATION_ROOT)
        runner._atomic_write_json(
            runner.DEPENDENCY_MANIFEST_PATH, dict(state["new_dependency"])
        )
        recomputed = {
            role: runner.recompute_execution_leg(
                role=role,
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
        runner._atomic_write_json(runner.EXECUTION_PROFILE_PATH, profile)
        runner._atomic_write_json(legacy_report, profile)
        checked = runner.validate_execution_profile(
            profile,
            expected_contract_sha256=new_contract,
            expected_receipt_bindings=state["new_bindings"],
        )
        migrated_calibration_checked = _load_root_receipts(
            runner.CALIBRATION_ROOT,
            expected_n=4,
            expected_bindings=state["new_bindings"],
        )
        if set(migrated_calibration_checked) != set(migrated_calibration):
            raise RuntimeError("BRAINIAC_SOURCE_MIGRATION_CALIBRATION_FINAL_INVALID")
        body: dict[str, Any] = {
            "schema_version": 1,
            "artifact": "BRAINIAC_PLAN017_SOURCE_ONLY_RESUME_MIGRATION_V1",
            "status": "PASS_SOURCE_ONLY_RUNTIME_RESUME_MIGRATION",
            "created_at_utc": _utc_now(),
            "old_dependency_closure_sha256": OLD_DEPENDENCY_CLOSURE_SHA256,
            "new_dependency_closure_sha256": state["new_dependency"]["closure_sha256"],
            "allowed_source_transitions": {
                path: {"old_sha256": old, "new_sha256": new}
                for path, (old, new) in sorted(ALLOWED_SOURCE_TRANSITIONS.items())
            },
            "scientific_input_drift_count": 0,
            "output_byte_drift_count": 0,
            "transform_byte_drift_count": 0,
            "scientific_receipt_field_drift_count": 0,
            "preprocessing_reexecuted": False,
            "outer_loader_invocation_count": 0,
            "migrated_equivalence_receipts": 36,
            "migrated_calibration_receipts": 4,
            "recovered_calibration_progress": "4_OF_32",
            "duplicate_recomputed_calibration_receipts": len(state["overlap"]),
            "old_live_bindings": old_bindings,
            "staged_bindings_before_commit": staged_bindings,
            "new_live_bindings": {
                "equivalence": _tree_binding(state["equivalence_root"]),
                "calibration": _tree_binding(runner.CALIBRATION_ROOT),
                "execution_profile": _file_binding(runner.EXECUTION_PROFILE_PATH),
                "dependency_manifest": _file_binding(runner.DEPENDENCY_MANIFEST_PATH),
            },
            "archive_bindings": {
                "data": _tree_binding(data_archive),
                "results": _tree_binding(result_archive),
                "calibration_source_retained": _tree_binding(calibration_source),
            },
            "execution_profile": {
                "status": checked["status"],
                "execution_mode": checked["execution_mode"],
                "exact_artifact_and_scientific_equivalence": checked[
                    "exact_artifact_and_scientific_equivalence"
                ],
                "timing_fresh": checked["timing_fresh"],
                "median_speedup_fraction": checked["median_speedup_fraction"],
                "integrity_sha256": checked["integrity_sha256"],
            },
            "primary_valid_n": 0,
            "plans_005_006_007_allowed": False,
            "manual_override_allowed": False,
            "human_approval_required": False,
            "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        }
        body["integrity_sha256"] = runner.canonical_sha(body)
        runner._atomic_write_json(MIGRATION_REPORT, body)
        return body
    finally:
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "migrate"), required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    args = parser.parse_args()
    source = args.calibration_source.resolve(strict=False)
    try:
        if args.mode == "preflight":
            state = _preflight(source)
            if state["status"] == "ALREADY_COMPLETED":
                report = dict(state["report"])
            else:
                report = {
                    "schema_version": 1,
                    "status": "PASS_SOURCE_ONLY_MIGRATION_PREFLIGHT",
                    "old_dependency_closure_sha256": state["old_dependency"][
                        "closure_sha256"
                    ],
                    "new_dependency_closure_sha256": state["new_dependency"][
                        "closure_sha256"
                    ],
                    "equivalence_receipt_count": sum(
                        len(receipts) for receipts in state["legs"].values()
                    ),
                    "calibration_receipt_count": len(
                        state["calibration_receipts"]
                    ),
                    "live_duplicate_receipt_count": len(state["overlap"]),
                    "calibration_source": _relative(source),
                    "scientific_input_drift_count": 0,
                    "primary_valid_n": 0,
                    "manual_override_allowed": False,
                }
                report["integrity_sha256"] = runner.canonical_sha(report)
        else:
            report = migrate(source)
    except Exception as exc:
        body = {
            "schema_version": 1,
            "status": "BLOCKED",
            "reason_code": runner._sanitize_exception_message(exc),
            "created_at_utc": _utc_now(),
            "manual_override_allowed": False,
        }
        body["integrity_sha256"] = runner.canonical_sha(body)
        runner._atomic_write_json(BLOCKED_REPORT, body)
        print(json.dumps(body, sort_keys=True))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
