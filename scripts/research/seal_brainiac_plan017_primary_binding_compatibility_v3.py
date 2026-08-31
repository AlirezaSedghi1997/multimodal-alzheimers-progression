"""Seal the exact source-only primary-binding repair after Plan 017.

The already-finalized equivalence, calibration, threshold, and smoke evidence
remains bound to the dependency closure that produced it.  This utility proves
that the next source revision changes only provenance routing/validation and
that all scientific preprocessing functions are AST-identical.  It archives
stale acceptance snapshots, preserves the previous dependency manifest, and
installs the current manifest without reprocessing any MRI.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import migrate_brainiac_plan017_resume_v1 as migration_v1  # noqa: E402
from scripts.research import prepare_brainiac_raw_t1_v2 as runner  # noqa: E402
from src.research import brainiac_preprocessing_v2 as v2  # noqa: E402
from src.research.brainiac_action_guard import inspect_action_exclusion  # noqa: E402
from src.research.brainiac_method_x_isolation import (  # noqa: E402
    verify_frozen_method_x_source_contract,
)
from src.research.brainiac_preprocessing_v2_schema import (  # noqa: E402
    validate_threshold_manifest,
)


OLD_DEPENDENCY_CLOSURE_SHA256 = (
    "93fbd6011cce4fd0f906c7e90f41fff44cffd30345ff57cfa877550d4d867ae0"
)
NEW_DEPENDENCY_CLOSURE_SHA256 = (
    "d82d4f0b4061d5f92a9435eb85bb6b0970919baad3808773ef096e7d3017e946"
)
RUNNER_RELATIVE_PATH = "scripts/research/prepare_brainiac_raw_t1_v2.py"
OLD_RUNNER_SHA256 = (
    "3084e52fccec3e19bcd77331c324ee0c7ea606f142d1bd5e9f45fc5984b26c63"
)
NEW_RUNNER_SHA256 = (
    "e3fbb930e5a1ced765bcd17b0ab97dec4d07c57f534c233bd5c96849570add60"
)
OLD_SOURCE_COMMIT = "eefdcf2b34603f63490534ee7333fb1b43f72806"
REPORT_PATH = runner.PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH
BLOCKED_PATH = (
    runner.REGISTRY_ROOT
    / "BLOCKED_BRAINIAC_PLAN017_PRIMARY_BINDING_SOURCE_COMPATIBILITY_V3.json"
)
ARCHIVE_ROOT = (
    runner.REGISTRY_ROOT
    / "superseded"
    / "plan017_primary_binding_source_compatibility_v3"
)

CRITICAL_PREPROCESSING_FUNCTIONS = frozenset(
    {
        "_pin_itk_threads",
        "_scratch_paths",
        "_validate_runtime_receipts",
        "_validate_template_assets",
        "_validate_hd_bet_weight",
        "_policy",
        "_config",
        "_lineage_binding",
        "_pipeline_contract_sha256",
        "_load_fixed_template",
        "_canonical_rigid_parameters",
        "_canonical_registration_metric",
        "_canonicalize_rigid_transform",
        "_registration",
        "_metric_value",
        "_transform_qc",
        "_image_geometry",
        "_foreground_com_physical",
        "_relative_com",
        "_serialized_registration_stage",
        "_run_hd_bet",
        "make_real_pipeline",
    }
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _load_json(path: Path) -> dict[str, Any]:
    return migration_v1._load_json(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_show_source(commit: str, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_OLD_SOURCE_UNAVAILABLE")
    raw = completed.stdout
    if _sha256_bytes(raw) != OLD_RUNNER_SHA256:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_OLD_SOURCE_SHA_INVALID")
    return raw.decode("utf-8")


def _function_ast_hashes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            canonical = ast.dump(node, annotate_fields=True, include_attributes=False)
            result[node.name] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def assert_critical_preprocessing_functions_unchanged(
    old_source: str, new_source: str
) -> dict[str, str]:
    old = _function_ast_hashes(old_source)
    new = _function_ast_hashes(new_source)
    comparable = CRITICAL_PREPROCESSING_FUNCTIONS & (set(old) | set(new))
    if not comparable:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_PREPROCESSING_FUNCTIONS_MISSING")
    if any(name not in old or name not in new or old[name] != new[name] for name in comparable):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_PREPROCESSING_FUNCTION_DRIFT")
    return {name: new[name] for name in sorted(comparable)}


def assert_dependency_transition(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> None:
    if old.get("closure_sha256") != OLD_DEPENDENCY_CLOSURE_SHA256:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_OLD_CLOSURE_INVALID")
    if new.get("closure_sha256") != NEW_DEPENDENCY_CLOSURE_SHA256:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_NEW_CLOSURE_INVALID")
    for role in ("dynamic_code_inputs", "static_inputs"):
        if old.get(role) != new.get(role):
            raise RuntimeError("BRAINIAC_PRIMARY_BINDING_SCIENTIFIC_INPUT_DRIFT")
    old_python = dict(old.get("python_closure") or {})
    new_python = dict(new.get("python_closure") or {})
    if set(old_python) != set(new_python):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_CLOSURE_FILE_SET_DRIFT")
    changed = {
        path
        for path in old_python
        if str(old_python[path]).casefold() != str(new_python[path]).casefold()
    }
    if changed != {RUNNER_RELATIVE_PATH}:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_UNAPPROVED_SOURCE_DRIFT")
    if (
        str(old_python[RUNNER_RELATIVE_PATH]).casefold() != OLD_RUNNER_SHA256
        or str(new_python[RUNNER_RELATIVE_PATH]).casefold() != NEW_RUNNER_SHA256
    ):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_SOURCE_SHA_MISMATCH")


def _assert_run_primary_uses_current_binding(source: str) -> None:
    tree = ast.parse(source)
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_primary"
        ),
        None,
    )
    if target is None:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_RUN_PRIMARY_MISSING")
    assignments: list[str] = []
    for node in ast.walk(target):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(item, ast.Name) and item.id == "bindings" for item in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            assignments.append(node.value.func.id)
    if assignments != ["_base_bindings"]:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_RUN_PRIMARY_INVALID")


def _primary_n() -> int:
    receipts = v2.PRIMARY_ROOT / "receipts"
    return len(list(receipts.glob("*.json"))) if receipts.is_dir() else 0


def _validate_live_evidence() -> dict[str, Any]:
    if runner.sha256_file(ROOT / RUNNER_RELATIVE_PATH) != NEW_RUNNER_SHA256:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_LIVE_RUNNER_SHA_INVALID")
    exclusion = inspect_action_exclusion()
    if exclusion.get("status") != "PASS" or exclusion.get("blockers"):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_WRITER_EXCLUSION_BLOCKED")
    if _primary_n() != 0:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_PRIMARY_ALREADY_STARTED")
    parent = runner._validated_postcalibration_source_migration()
    old_dependency = _load_json(runner.DEPENDENCY_MANIFEST_PATH)
    new_dependency = runner._dependency_manifest()
    assert_dependency_transition(old_dependency, new_dependency)

    old_source = _git_show_source(OLD_SOURCE_COMMIT, RUNNER_RELATIVE_PATH)
    new_source = (ROOT / RUNNER_RELATIVE_PATH).read_text(encoding="utf-8")
    critical_hashes = assert_critical_preprocessing_functions_unchanged(
        old_source, new_source
    )
    if set(critical_hashes) != set(CRITICAL_PREPROCESSING_FUNCTIONS):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_PREPROCESSING_FUNCTION_SET_DRIFT")
    _assert_run_primary_uses_current_binding(new_source)

    base = runner._base_bindings()
    if (
        base.get("dependency_closure_sha256") != NEW_DEPENDENCY_CLOSURE_SHA256
        or base.get("qc_threshold_manifest_sha256")
        != runner.sha256_file(runner.THRESHOLD_MANIFEST_PATH)
    ):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_CURRENT_BINDINGS_INVALID")
    equivalence_bindings = runner._equivalence_receipt_bindings()
    threshold_bindings = runner._threshold_evidence_bindings()
    if (
        equivalence_bindings.get("dependency_closure_sha256")
        != OLD_DEPENDENCY_CLOSURE_SHA256
        or threshold_bindings.get("dependency_closure_sha256")
        != OLD_DEPENDENCY_CLOSURE_SHA256
        or threshold_bindings.get("qc_threshold_manifest_sha256")
        != base.get("qc_threshold_manifest_sha256")
    ):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_HISTORICAL_BINDINGS_INVALID")
    profile = _load_json(runner.EXECUTION_PROFILE_PATH)
    checked_profile = runner.validate_execution_profile(
        profile,
        expected_contract_sha256=runner.canonical_sha(
            runner._threshold_contract_bindings(equivalence_bindings)
        ),
        expected_receipt_bindings=equivalence_bindings,
    )
    calibration = migration_v1._load_root_receipts(
        runner.CALIBRATION_ROOT,
        expected_n=32,
        expected_bindings=equivalence_bindings,
    )
    threshold = _load_json(runner.THRESHOLD_MANIFEST_PATH)
    validate_threshold_manifest(
        threshold,
        expected_sample_hashes=sorted(calibration),
        expected_receipt_integrity_sha256s={
            sample: str(row["integrity_sha256"])
            for sample, row in sorted(calibration.items())
        },
        expected_contract_bindings=runner._threshold_contract_bindings(
            threshold_bindings
        ),
    )
    source = verify_frozen_method_x_source_contract(
        expected_digest=runner.EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    if runner.sha256_file(runner.BATTERY_MANIFEST_PATH) != runner.EXPECTED_BATTERY_MANIFEST_SHA256:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_BATTERY_MANIFEST_DRIFT")
    return {
        "parent": parent,
        "old_dependency": old_dependency,
        "new_dependency": new_dependency,
        "critical_hashes": critical_hashes,
        "base_bindings": base,
        "equivalence_bindings": equivalence_bindings,
        "threshold_bindings": threshold_bindings,
        "profile": checked_profile,
        "threshold": threshold,
        "calibration": calibration,
        "method_x_source_digest": source["source_contract_digest"],
    }


def _validate_sealed_json(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if recorded != runner.canonical_sha(body):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_STALE_ARTIFACT_INVALID")
    return payload


def _archive_stale_artifact(path: Path, *, role: str) -> dict[str, Any] | None:
    role_root = ARCHIVE_ROOT / role
    if path.is_file():
        _validate_sealed_json(path)
        digest = runner.sha256_file(path)
        target = role_root / f"sha256-{digest}" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError("BRAINIAC_PRIMARY_BINDING_ARCHIVE_COLLISION")
        os.replace(path, target)
        reason: dict[str, Any] = {
            "schema_version": 1,
            "artifact": "BRAINIAC_PRIMARY_BINDING_STALE_ACCEPTANCE_SUPERSESSION",
            "status": "SUPERSEDED_APPEND_ONLY",
            "created_at_utc": _utc_now(),
            "role": role,
            "reason_code": "SOURCE_ONLY_PRIMARY_BINDING_CONTRACT_ADVANCED",
            "archived_artifact": runner._artifact_binding(target),
            "old_dependency_closure_sha256": OLD_DEPENDENCY_CLOSURE_SHA256,
            "new_dependency_closure_sha256": NEW_DEPENDENCY_CLOSURE_SHA256,
            "primary_valid_n": 0,
            "manual_override_allowed": False,
        }
        reason["integrity_sha256"] = runner.canonical_sha(reason)
        reason_path = target.parent / "reason.json"
        runner._atomic_write_json(reason_path, reason)
        return {
            "artifact": runner._artifact_binding(target),
            "reason": runner._artifact_binding(reason_path),
        }
    candidates = sorted(role_root.glob(f"sha256-*/{path.name}"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_ARCHIVE_AMBIGUOUS")
    target = candidates[0]
    reason_path = target.parent / "reason.json"
    _validate_sealed_json(target)
    _validate_sealed_json(reason_path)
    return {
        "artifact": runner._artifact_binding(target),
        "reason": runner._artifact_binding(reason_path),
    }


def _archive_old_dependency_and_install_current(
    current: Mapping[str, Any],
) -> dict[str, Any]:
    role_root = ARCHIVE_ROOT / "dependency_manifest"
    old_file_sha = None
    if runner.DEPENDENCY_MANIFEST_PATH.is_file():
        live = _load_json(runner.DEPENDENCY_MANIFEST_PATH)
        if live == dict(current):
            candidates = sorted(role_root.glob("sha256-*/old_dependency_manifest.json"))
            if len(candidates) != 1:
                raise RuntimeError("BRAINIAC_PRIMARY_BINDING_DEPENDENCY_ARCHIVE_MISSING")
            archived = candidates[0]
            return runner._artifact_binding(archived)
        if live.get("closure_sha256") != OLD_DEPENDENCY_CLOSURE_SHA256:
            raise RuntimeError("BRAINIAC_PRIMARY_BINDING_DEPENDENCY_LIVE_DRIFT")
        old_file_sha = runner.sha256_file(runner.DEPENDENCY_MANIFEST_PATH)
        archived = role_root / f"sha256-{old_file_sha}" / "old_dependency_manifest.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        if archived.exists():
            raise RuntimeError("BRAINIAC_PRIMARY_BINDING_DEPENDENCY_ARCHIVE_COLLISION")
        os.replace(runner.DEPENDENCY_MANIFEST_PATH, archived)
    else:
        candidates = sorted(role_root.glob("sha256-*/old_dependency_manifest.json"))
        if len(candidates) != 1:
            raise RuntimeError("BRAINIAC_PRIMARY_BINDING_DEPENDENCY_ARCHIVE_MISSING")
        archived = candidates[0]
        old_file_sha = runner.sha256_file(archived)
    if old_file_sha is None:
        old_file_sha = runner.sha256_file(archived)
    runner._atomic_write_json(runner.DEPENDENCY_MANIFEST_PATH, dict(current))
    if _load_json(runner.DEPENDENCY_MANIFEST_PATH) != dict(current):
        raise RuntimeError("BRAINIAC_PRIMARY_BINDING_DEPENDENCY_INSTALL_FAILED")
    return runner._artifact_binding(archived)


def seal() -> dict[str, Any]:
    if REPORT_PATH.is_file():
        return runner._validated_primary_binding_source_compatibility()
    state = _validate_live_evidence()
    archived = {
        "acceptance_candidate": _archive_stale_artifact(
            runner.ACCEPTANCE_CANDIDATE_PATH, role="acceptance_candidate"
        ),
        "acceptance_preflight_snapshot": _archive_stale_artifact(
            runner.ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH,
            role="acceptance_preflight_snapshot",
        ),
    }
    archived_dependency = _archive_old_dependency_and_install_current(
        state["new_dependency"]
    )
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PLAN017_PRIMARY_BINDING_SOURCE_COMPATIBILITY_V3",
        "status": "PASS_SOURCE_ONLY_PRIMARY_BINDING_COMPATIBILITY",
        "created_at_utc": _utc_now(),
        "parent_migration": runner._artifact_binding(
            runner.POSTCALIBRATION_SOURCE_MIGRATION_PATH
        ),
        "allowed_source_transition": {
            "path": RUNNER_RELATIVE_PATH,
            "old_sha256": OLD_RUNNER_SHA256,
            "new_sha256": NEW_RUNNER_SHA256,
        },
        "old_source_commit": OLD_SOURCE_COMMIT,
        "old_dependency_closure_sha256": OLD_DEPENDENCY_CLOSURE_SHA256,
        "new_dependency_closure_sha256": NEW_DEPENDENCY_CLOSURE_SHA256,
        "critical_preprocessing_function_sha256s": state["critical_hashes"],
        "scientific_preprocessing_function_drift_count": 0,
        "output_byte_drift_count": 0,
        "transform_byte_drift_count": 0,
        "threshold_numeric_drift_count": 0,
        "primary_binding_uses_current_dependency_closure": True,
        "primary_binding_uses_locked_threshold_sha256": True,
        "historical_evidence_uses_locked_dependency_closure": True,
        "historical_execution_profile_integrity_sha256": state["profile"][
            "integrity_sha256"
        ],
        "historical_threshold_manifest_sha256": runner.sha256_file(
            runner.THRESHOLD_MANIFEST_PATH
        ),
        "historical_calibration_receipt_count": len(state["calibration"]),
        "archived_stale_acceptance": archived,
        "archived_dependency_manifest": archived_dependency,
        "current_dependency_manifest": runner._artifact_binding(
            runner.DEPENDENCY_MANIFEST_PATH
        ),
        "method_x_source_digest": state["method_x_source_digest"],
        "battery_manifest_sha256": runner.EXPECTED_BATTERY_MANIFEST_SHA256,
        "preprocessing_reexecuted": False,
        "outer_loader_invocation_count": 0,
        "primary_valid_n": 0,
        "plans_005_006_007_allowed": False,
        "training_authorization": False,
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    body["integrity_sha256"] = runner.canonical_sha(body)
    runner._atomic_write_json(REPORT_PATH, body)
    runner._validated_primary_binding_source_compatibility()
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "seal"), required=True)
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            state = _validate_live_evidence()
            report: dict[str, Any] = {
                "schema_version": 1,
                "status": "PASS_PRIMARY_BINDING_SOURCE_COMPATIBILITY_PREFLIGHT",
                "old_dependency_closure_sha256": OLD_DEPENDENCY_CLOSURE_SHA256,
                "new_dependency_closure_sha256": state["new_dependency"][
                    "closure_sha256"
                ],
                "critical_preprocessing_function_count": len(
                    state["critical_hashes"]
                ),
                "scientific_preprocessing_function_drift_count": 0,
                "primary_valid_n": 0,
                "manual_override_allowed": False,
            }
            report["integrity_sha256"] = runner.canonical_sha(report)
        else:
            report = seal()
    except Exception as exc:
        blocked: dict[str, Any] = {
            "schema_version": 1,
            "status": "BLOCKED",
            "reason_code": runner._sanitize_exception_message(exc),
            "created_at_utc": _utc_now(),
            "primary_valid_n": _primary_n(),
            "manual_override_allowed": False,
        }
        blocked["integrity_sha256"] = runner.canonical_sha(blocked)
        runner._atomic_write_json(BLOCKED_PATH, blocked)
        print(json.dumps(blocked, sort_keys=True))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
