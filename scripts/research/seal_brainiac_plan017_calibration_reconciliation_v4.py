"""Repair the false Plan-017 calibration reconciliation without MRI reruns.

The V3 source compatibility change intentionally kept equivalence/calibration
evidence on its historical closure.  A legacy-generation probe accidentally
validated calibration receipts against the newer primary closure and moved the
byte-valid tree to the superseded area.  This one-shot, resumable transaction:

* proves the scientific preprocessing functions are unchanged;
* proves the archived tree is exactly the tree named by the reconciliation;
* restores that tree with an atomic same-volume move and validates all 32
  committed receipts plus the locked threshold;
* advances only the dependency manifest and archives stale readiness evidence;
* emits an append-only V4 compatibility receipt.

No MRI, outer loader, model trainer, or primary preprocessing path is invoked.
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
from scripts.research import (  # noqa: E402
    seal_brainiac_plan017_primary_binding_compatibility_v3 as compatibility_v3,
)
from src.research import brainiac_preprocessing_v2 as v2  # noqa: E402
from src.research.brainiac_action_guard import inspect_action_exclusion  # noqa: E402
from src.research.brainiac_method_x_isolation import (  # noqa: E402
    verify_frozen_method_x_source_contract,
)


OLD_SOURCE_COMMIT = "5472cdaf661b151f806adf95f3a2e1417b080731"
RUNNER_RELATIVE_PATH = "scripts/research/prepare_brainiac_raw_t1_v2.py"
OLD_RUNNER_SHA256 = runner.PLAN017_PRIMARY_BINDING_V3_RUNNER_SHA256
OLD_DEPENDENCY_CLOSURE_SHA256 = (
    runner.PLAN017_PRIMARY_BINDING_V3_DEPENDENCY_CLOSURE_SHA256
)
REPORT_PATH = runner.CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_PATH
BLOCKED_PATH = (
    runner.REGISTRY_ROOT
    / "BLOCKED_BRAINIAC_PLAN017_CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_V4.json"
)
ARCHIVE_ROOT = (
    runner.REGISTRY_ROOT
    / "superseded"
    / "plan017_calibration_reconciliation_source_compatibility_v4"
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
    if completed.returncode != 0 or _sha256_bytes(completed.stdout) != OLD_RUNNER_SHA256:
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_OLD_SOURCE_INVALID")
    return completed.stdout.decode("utf-8")


def _function_ast_hashes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            canonical = ast.dump(node, annotate_fields=True, include_attributes=False)
            result[node.name] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def assert_scientific_preprocessing_unchanged(
    old_source: str, new_source: str
) -> dict[str, str]:
    old = _function_ast_hashes(old_source)
    new = _function_ast_hashes(new_source)
    required = compatibility_v3.CRITICAL_PREPROCESSING_FUNCTIONS
    if any(name not in old or name not in new or old[name] != new[name] for name in required):
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_SCIENTIFIC_FUNCTION_DRIFT")
    return {name: new[name] for name in sorted(required)}


def assert_dependency_transition(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> None:
    if old.get("closure_sha256") != OLD_DEPENDENCY_CLOSURE_SHA256:
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_OLD_CLOSURE_INVALID")
    for role in ("dynamic_code_inputs", "static_inputs"):
        if old.get(role) != new.get(role):
            raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_SCIENTIFIC_INPUT_DRIFT")
    old_python = dict(old.get("python_closure") or {})
    new_python = dict(new.get("python_closure") or {})
    if set(old_python) != set(new_python):
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_CLOSURE_FILE_SET_DRIFT")
    changed = {
        path
        for path in old_python
        if str(old_python[path]).casefold() != str(new_python[path]).casefold()
    }
    if changed != {RUNNER_RELATIVE_PATH}:
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_UNAPPROVED_SOURCE_DRIFT")
    if str(old_python[RUNNER_RELATIVE_PATH]).casefold() != OLD_RUNNER_SHA256:
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_OLD_RUNNER_SHA_INVALID")
    if str(new_python[RUNNER_RELATIVE_PATH]).casefold() != runner.sha256_file(
        ROOT / RUNNER_RELATIVE_PATH
    ):
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_NEW_RUNNER_SHA_INVALID")


def _primary_n() -> int:
    receipts = v2.PRIMARY_ROOT / "receipts"
    return len(list(receipts.glob("*.json"))) if receipts.is_dir() else 0


def _validated_false_reconciliation() -> tuple[dict[str, Any], dict[str, Any], Path]:
    report = _load_json(runner.FALSE_CALIBRATION_RECONCILIATION_PATH)
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    rows = report.get("artifacts")
    if (
        recorded != runner.canonical_sha(body)
        or report.get("status") != "ARCHIVED_INCOMPATIBLE_PREPROCESSING_GENERATION"
        or report.get("artifact_count") != 1
        or not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("source_role") != "r1_primary_v2_calibration"
    ):
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_RECONCILIATION_INVALID")
    row = dict(rows[0])
    archive = ROOT / str(row.get("archive_relative_path") or "")
    return report, row, archive


def _validate_calibration_tree(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    evidence = runner._tree_evidence(root)
    if (
        evidence.get("tree_sha256") != row.get("tree_sha256")
        or evidence.get("file_count") != row.get("file_count")
        or evidence.get("bytes") != row.get("bytes")
    ):
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_TREE_MISMATCH")
    receipts = migration_v1._load_root_receipts(
        root,
        expected_n=32,
        expected_bindings=runner._equivalence_receipt_bindings(),
    )
    if len(receipts) != 32:
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_RECEIPT_COUNT_INVALID")
    return evidence


def _restore_calibration_tree(row: Mapping[str, Any], archive: Path) -> dict[str, Any]:
    live = runner.CALIBRATION_ROOT
    if archive.exists() and live.exists():
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_BOTH_LIVE_AND_ARCHIVE_PRESENT")
    if archive.exists():
        _validate_calibration_tree(archive, row)
        live.parent.mkdir(parents=True, exist_ok=True)
        os.replace(archive, live)
    elif not live.exists():
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_TREE_MISSING")
    evidence = _validate_calibration_tree(live, row)
    if archive.exists():
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_ARCHIVE_MOVE_INCOMPLETE")
    runner._load_bound_threshold_manifest(runner._threshold_evidence_bindings())
    return evidence


def _validate_sealed_json(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if recorded != runner.canonical_sha(body):
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_STALE_ARTIFACT_INVALID")
    return payload


def _archive_stale_artifact(path: Path, *, role: str) -> dict[str, Any] | None:
    role_root = ARCHIVE_ROOT / role
    if path.is_file():
        _validate_sealed_json(path)
        digest = runner.sha256_file(path)
        target = role_root / f"sha256-{digest}" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_ARCHIVE_COLLISION")
        os.replace(path, target)
        reason: dict[str, Any] = {
            "schema_version": 1,
            "artifact": "BRAINIAC_CALIBRATION_REPAIR_STALE_EVIDENCE_SUPERSESSION",
            "status": "SUPERSEDED_APPEND_ONLY",
            "created_at_utc": _utc_now(),
            "role": role,
            "reason_code": "CALIBRATION_RECONCILIATION_SOURCE_CONTRACT_ADVANCED",
            "archived_artifact": runner._artifact_binding(target),
            "old_dependency_closure_sha256": OLD_DEPENDENCY_CLOSURE_SHA256,
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
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_ARCHIVE_AMBIGUOUS")
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
    if runner.DEPENDENCY_MANIFEST_PATH.is_file():
        live = _load_json(runner.DEPENDENCY_MANIFEST_PATH)
        if live == dict(current):
            candidates = sorted(role_root.glob("sha256-*/old_dependency_manifest.json"))
            if len(candidates) != 1:
                raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_DEPENDENCY_ARCHIVE_MISSING")
            return runner._artifact_binding(candidates[0])
        if live.get("closure_sha256") != OLD_DEPENDENCY_CLOSURE_SHA256:
            raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_DEPENDENCY_LIVE_DRIFT")
        digest = runner.sha256_file(runner.DEPENDENCY_MANIFEST_PATH)
        archived = role_root / f"sha256-{digest}" / "old_dependency_manifest.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        if archived.exists():
            raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_DEPENDENCY_ARCHIVE_COLLISION")
        os.replace(runner.DEPENDENCY_MANIFEST_PATH, archived)
    else:
        candidates = sorted(role_root.glob("sha256-*/old_dependency_manifest.json"))
        if len(candidates) != 1:
            raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_DEPENDENCY_ARCHIVE_MISSING")
        archived = candidates[0]
    runner._atomic_write_json(runner.DEPENDENCY_MANIFEST_PATH, dict(current))
    if _load_json(runner.DEPENDENCY_MANIFEST_PATH) != dict(current):
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_DEPENDENCY_INSTALL_FAILED")
    return runner._artifact_binding(archived)


def _validate_live_state(*, restore: bool) -> dict[str, Any]:
    exclusion = inspect_action_exclusion()
    if exclusion.get("status") != "PASS" or exclusion.get("blockers"):
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_WRITER_EXCLUSION_BLOCKED")
    if _primary_n() != 0:
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_PRIMARY_ALREADY_STARTED")
    parent = runner._validated_primary_binding_source_compatibility()
    reconciliation, row, archive = _validated_false_reconciliation()
    old_dependency = _load_json(runner.DEPENDENCY_MANIFEST_PATH)
    new_dependency = runner._dependency_manifest()
    assert_dependency_transition(old_dependency, new_dependency)
    old_source = _git_show_source(OLD_SOURCE_COMMIT, RUNNER_RELATIVE_PATH)
    new_source = (ROOT / RUNNER_RELATIVE_PATH).read_text(encoding="utf-8")
    critical_hashes = assert_scientific_preprocessing_unchanged(old_source, new_source)
    if set(critical_hashes) != set(compatibility_v3.CRITICAL_PREPROCESSING_FUNCTIONS):
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_FUNCTION_SET_DRIFT")
    if restore:
        restored_tree = _restore_calibration_tree(row, archive)
    else:
        source = archive if archive.exists() else runner.CALIBRATION_ROOT
        if not source.exists():
            raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_TREE_MISSING")
        restored_tree = _validate_calibration_tree(source, row)
    source_contract = verify_frozen_method_x_source_contract(
        expected_digest=runner.EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    if runner.sha256_file(runner.BATTERY_MANIFEST_PATH) != runner.EXPECTED_BATTERY_MANIFEST_SHA256:
        raise RuntimeError("BRAINIAC_CALIBRATION_REPAIR_BATTERY_MANIFEST_DRIFT")
    return {
        "parent": parent,
        "reconciliation": reconciliation,
        "row": row,
        "archive": archive,
        "old_dependency": old_dependency,
        "new_dependency": new_dependency,
        "critical_hashes": critical_hashes,
        "restored_tree": restored_tree,
        "method_x_source_digest": source_contract["source_contract_digest"],
    }


def seal() -> dict[str, Any]:
    if REPORT_PATH.is_file():
        return runner._validated_calibration_reconciliation_source_compatibility()
    state = _validate_live_state(restore=True)
    stale_paths = {
        "blocked_attempt": BLOCKED_PATH,
        "acceptance_candidate": runner.ACCEPTANCE_CANDIDATE_PATH,
        "acceptance_preflight_snapshot": runner.ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH,
        "acceptance_test_manifest": runner.ACCEPTANCE_TEST_MANIFEST_PATH,
        "full_test_manifest": runner.FULL_TEST_MANIFEST_PATH,
        "traceability_manifest": runner.TRACEABILITY_MANIFEST_PATH,
        "final_readiness": runner.READY_REPORT_PATH,
    }
    archived_stale = {
        role: _archive_stale_artifact(path, role=role)
        for role, path in stale_paths.items()
    }
    archived_dependency = _archive_old_dependency_and_install_current(
        state["new_dependency"]
    )
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PLAN017_CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_V4",
        "status": "PASS_FALSE_CALIBRATION_RECONCILIATION_REPAIRED",
        "created_at_utc": _utc_now(),
        "parent_v3": runner._artifact_binding(
            runner.PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH
        ),
        "false_reconciliation": runner._artifact_binding(
            runner.FALSE_CALIBRATION_RECONCILIATION_PATH
        ),
        "allowed_source_transition": {
            "path": RUNNER_RELATIVE_PATH,
            "old_sha256": OLD_RUNNER_SHA256,
            "new_sha256": runner.sha256_file(ROOT / RUNNER_RELATIVE_PATH),
        },
        "old_source_commit": OLD_SOURCE_COMMIT,
        "old_dependency_closure_sha256": OLD_DEPENDENCY_CLOSURE_SHA256,
        "new_dependency_closure_sha256": state["new_dependency"]["closure_sha256"],
        "critical_preprocessing_function_sha256s": state["critical_hashes"],
        "scientific_preprocessing_function_drift_count": 0,
        "output_byte_drift_count": 0,
        "transform_byte_drift_count": 0,
        "threshold_numeric_drift_count": 0,
        "calibration_restored_from_exact_archive": True,
        "restored_calibration_tree": state["restored_tree"],
        "historical_calibration_receipt_count": 32,
        "archived_stale_artifacts": archived_stale,
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
    runner._validated_calibration_reconciliation_source_compatibility()
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "seal"), required=True)
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            state = _validate_live_state(restore=False)
            report: dict[str, Any] = {
                "schema_version": 1,
                "status": "PASS_CALIBRATION_RECONCILIATION_COMPATIBILITY_PREFLIGHT",
                "old_dependency_closure_sha256": OLD_DEPENDENCY_CLOSURE_SHA256,
                "new_dependency_closure_sha256": state["new_dependency"][
                    "closure_sha256"
                ],
                "critical_preprocessing_function_count": len(
                    state["critical_hashes"]
                ),
                "scientific_preprocessing_function_drift_count": 0,
                "calibration_tree_sha256": state["restored_tree"]["tree_sha256"],
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
