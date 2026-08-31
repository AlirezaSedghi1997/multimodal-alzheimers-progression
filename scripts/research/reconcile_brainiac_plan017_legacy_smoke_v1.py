"""One-shot, fail-closed reconciliation for the exact pre-Plan-17 smoke file.

The historical schema-2 smoke report predates the real 12-sample equivalence
profile.  It cannot be overwritten and it cannot satisfy the schema-3
cross-bind.  This utility accepts exactly the frozen historical payload, moves
it to an append-only superseded location, and emits a sealed receipt.  No MRI
preprocessing, outer evaluation, primary receipt, or training action occurs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import prepare_brainiac_raw_t1_v2 as runner  # noqa: E402
from src.research.brainiac_action_guard import inspect_action_exclusion  # noqa: E402
from src.research.brainiac_method_x_isolation import (  # noqa: E402
    verify_frozen_method_x_source_contract,
)


LEGACY_SMOKE_FILE_SHA256 = (
    "c132a7ccdfe88673d592d6f37128133d3a345a5df750bb2c3f64238bd30a01a3"
)
LEGACY_SMOKE_PAYLOAD: dict[str, Any] = {
    "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    "created_at_utc": "2026-08-23T13:00:00+00:00",
    "equivalence": "SKIPPED_SEQUENTIAL_ONLY_LOCKED",
    "integrity_sha256": "70bf84601b9f978248b605f04e8ecdd7ba0b3d814c898ff123b12126450266d8",
    "mode": "smoke",
    "rows": 12,
    "schema_version": 2,
    "sequential_status": "COMPLETED",
    "status": "PASS",
}
REPORT_PATH = (
    runner.REGISTRY_ROOT / "BRAINIAC_PLAN017_LEGACY_SMOKE_RECONCILIATION_V1.json"
)
ARCHIVE_ROOT = (
    runner.REGISTRY_ROOT / "superseded" / "legacy_smoke_reports"
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_EVIDENCE_NOT_REGULAR_FILE")
    stat = path.stat()
    if stat.st_size <= 0:
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_EVIDENCE_EMPTY")
    return {
        "path": _public_path(path),
        "bytes": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    try:
        with temp.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _validated_report(
    report_path: Path,
    *,
    expected_current_scientific: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    body = dict(report)
    recorded = body.pop("integrity_sha256", None)
    archived_binding = report.get("archived_smoke")
    reason_binding = report.get("reason_receipt")
    if (
        report.get("schema_version") != 1
        or report.get("artifact")
        != "BRAINIAC_PLAN017_LEGACY_SMOKE_RECONCILIATION"
        or report.get("status") != "PASS"
        or report.get("action") != "ARCHIVED_EXACT_LEGACY_SCHEMA2_SMOKE"
        or recorded != runner.canonical_sha(body)
        or report.get("legacy_payload_sha256")
        != runner.canonical_sha(LEGACY_SMOKE_PAYLOAD)
        or report.get("expected_current_scientific_sha256")
        != runner.canonical_sha(dict(expected_current_scientific))
        or int(report.get("primary_valid_n", -1))
        != int(evidence.get("primary_valid_n", -2))
        or int(report.get("calibration_valid_n", -1))
        != int(evidence.get("calibration_valid_n", -2))
        or not isinstance(archived_binding, Mapping)
        or not isinstance(reason_binding, Mapping)
    ):
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_RECONCILIATION_REPORT_INVALID")
    for binding in (archived_binding, reason_binding):
        path = ROOT / str(binding.get("path"))
        if _binding(path) != dict(binding):
            raise RuntimeError("BRAINIAC_LEGACY_SMOKE_RECONCILIATION_EVIDENCE_DRIFT")
    if report.get("preprocessing_reexecuted") is not False:
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_RECONCILIATION_REPORT_INVALID")
    return report


def reconcile_legacy_smoke(
    *,
    smoke_path: Path,
    archive_root: Path,
    report_path: Path,
    expected_current_scientific: Mapping[str, Any],
    evidence: Mapping[str, Any],
    require_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Archive only the exact historical payload; reject every other mismatch."""

    if (
        int(evidence.get("primary_valid_n", -1)) != 0
        or int(evidence.get("calibration_valid_n", -1)) != 32
        or expected_current_scientific.get("schema_version") != 3
    ):
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_SAFETY_GATE_BLOCKED")
    if report_path.is_file():
        if smoke_path.exists():
            raise RuntimeError("BRAINIAC_LEGACY_SMOKE_RECONCILIATION_DUPLICATE_LIVE_FILE")
        return _validated_report(
            report_path,
            expected_current_scientific=expected_current_scientific,
            evidence=evidence,
        )

    file_sha: str
    payload: dict[str, Any]
    if smoke_path.is_file():
        if smoke_path.is_symlink():
            raise RuntimeError("BRAINIAC_LEGACY_SMOKE_NOT_EXACT_ALLOWLIST")
        payload = json.loads(smoke_path.read_text(encoding="utf-8"))
        file_sha = _sha256_file(smoke_path)
        if payload != LEGACY_SMOKE_PAYLOAD or (
            require_file_sha256 is not None
            and file_sha.casefold() != require_file_sha256.casefold()
        ):
            raise RuntimeError("BRAINIAC_LEGACY_SMOKE_NOT_EXACT_ALLOWLIST")
    else:
        file_sha = require_file_sha256 or LEGACY_SMOKE_FILE_SHA256
        payload = dict(LEGACY_SMOKE_PAYLOAD)

    leaf = f"sha256-{file_sha.casefold()}"
    archive_dir = archive_root / leaf
    archived = archive_dir / smoke_path.name
    archive_dir.mkdir(parents=True, exist_ok=True)
    if archived.exists():
        if archived.is_symlink() or _sha256_file(archived) != file_sha:
            raise RuntimeError("BRAINIAC_LEGACY_SMOKE_ARCHIVE_COLLISION")
        if json.loads(archived.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("BRAINIAC_LEGACY_SMOKE_ARCHIVE_COLLISION")
        if smoke_path.exists():
            raise RuntimeError("BRAINIAC_LEGACY_SMOKE_ARCHIVE_COLLISION")
    else:
        if not smoke_path.is_file():
            raise RuntimeError("BRAINIAC_LEGACY_SMOKE_SOURCE_MISSING")
        os.replace(smoke_path, archived)

    reason_path = archive_dir / "reason.json"
    if not reason_path.exists():
        reason_body: dict[str, Any] = {
            "schema_version": 1,
            "artifact": "BRAINIAC_SUPERSEDED_LEGACY_SMOKE_RECEIPT",
            "status": "SUPERSEDED_APPEND_ONLY",
            "created_at_utc": _utc_now(),
            "reason_code": "PRE_PLAN017_SCHEMA2_SMOKE_NOT_REAL_EQUIVALENCE_CROSS_BIND",
            "legacy_file_sha256": file_sha,
            "legacy_payload_sha256": runner.canonical_sha(payload),
            "expected_current_scientific_sha256": runner.canonical_sha(
                dict(expected_current_scientific)
            ),
            "preprocessing_reexecuted": False,
            "primary_valid_n": 0,
            "outer_loader_invocation_count": 0,
            "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        }
        reason_body["integrity_sha256"] = runner.canonical_sha(reason_body)
        _atomic_write_json(reason_path, reason_body)

    archived_binding = _binding(archived)
    reason_binding = _binding(reason_path)
    body = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PLAN017_LEGACY_SMOKE_RECONCILIATION",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "action": "ARCHIVED_EXACT_LEGACY_SCHEMA2_SMOKE",
        "archive_leaf": leaf,
        "legacy_file_sha256": file_sha,
        "legacy_payload_sha256": runner.canonical_sha(payload),
        "expected_current_scientific_sha256": runner.canonical_sha(
            dict(expected_current_scientific)
        ),
        "archived_smoke": archived_binding,
        "reason_receipt": reason_binding,
        "primary_valid_n": 0,
        "calibration_valid_n": 32,
        "preprocessing_reexecuted": False,
        "outer_loader_invocation_count": 0,
        "plans_005_006_007_allowed": False,
        "training_authorization": False,
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    for key in (
        "dependency_closure_sha256",
        "execution_profile_integrity_sha256",
        "threshold_manifest_sha256",
        "method_x_source_digest",
        "battery_manifest_sha256",
    ):
        if key in evidence:
            body[key] = evidence[key]
    body["integrity_sha256"] = runner.canonical_sha(body)
    _atomic_write_json(report_path, body)
    return body


def _current_smoke_scientific(profile: Mapping[str, Any]) -> dict[str, Any]:
    legs = profile.get("legs")
    sequential = legs.get("sequential") if isinstance(legs, Mapping) else None
    if (
        not isinstance(sequential, Mapping)
        or len(sequential.get("sample_hashes") or ()) != 12
        or sequential.get("live_evidence_status")
        != "PASS_RECOMPUTED_FROM_LIVE_BYTES"
    ):
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_CURRENT_PROFILE_INVALID")
    return {
        "schema_version": 3,
        "artifact": "BRAINIAC_REAL_12_SAMPLE_SMOKE_CROSS_BIND",
        "status": "PASS",
        "rows": 12,
        "source_role": "EQUIVALENCE_SEQUENTIAL_LEG",
        "root_relative_path": sequential["root_relative_path"],
        "root_tree_sha256": sequential["root_tree_sha256"],
        "selection_sha256": sequential["selection_sha256"],
        "contract_sha256": profile["contract_sha256"],
        "execution_profile_integrity_sha256": profile["integrity_sha256"],
        "physical_qc_validated_from_live_receipts": True,
        "label_blind": True,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }


def _live_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    exclusion = inspect_action_exclusion()
    if exclusion.get("status") != "PASS" or exclusion.get("blockers"):
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_ACTIVE_WRITER_BLOCKED")
    status = runner.run_status()
    if (
        status.get("status") != "PASS"
        or int(status.get("expected_rows", -1)) != 1669
        or int(status.get("completed_rows", -1)) != 0
        or status.get("invalid_rows")
    ):
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_PRIMARY_STATE_BLOCKED")
    calibration = runner._calibrate_if_needed(device="cuda")
    if (
        calibration.get("status") not in {"PASS", "ALREADY_COMPLETED"}
        or int(calibration.get("selected_n", -1)) != 32
        or not runner.THRESHOLD_MANIFEST_PATH.is_file()
    ):
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_CALIBRATION_STATE_BLOCKED")
    profile_raw = json.loads(runner.EXECUTION_PROFILE_PATH.read_text(encoding="utf-8"))
    profile = runner.validate_execution_profile(
        profile_raw,
        expected_contract_sha256=runner._current_equivalence_contract_sha256(),
        expected_receipt_bindings=runner._equivalence_receipt_bindings(),
    )
    source = verify_frozen_method_x_source_contract(
        expected_digest=runner.EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    if runner.sha256_file(runner.BATTERY_MANIFEST_PATH) != runner.EXPECTED_BATTERY_MANIFEST_SHA256:
        raise RuntimeError("BRAINIAC_LEGACY_SMOKE_BATTERY_MANIFEST_DRIFT")
    evidence = {
        "primary_valid_n": 0,
        "calibration_valid_n": 32,
        "dependency_closure_sha256": runner._dependency_manifest()["closure_sha256"],
        "execution_profile_integrity_sha256": profile["integrity_sha256"],
        "threshold_manifest_sha256": runner.sha256_file(
            runner.THRESHOLD_MANIFEST_PATH
        ),
        "method_x_source_digest": source["source_contract_digest"],
        "battery_manifest_sha256": runner.EXPECTED_BATTERY_MANIFEST_SHA256,
    }
    return profile, evidence


def main() -> int:
    try:
        profile, evidence = _live_evidence()
        report = reconcile_legacy_smoke(
            smoke_path=runner.SMOKE_REPORT,
            archive_root=ARCHIVE_ROOT,
            report_path=REPORT_PATH,
            expected_current_scientific=_current_smoke_scientific(profile),
            evidence=evidence,
            require_file_sha256=LEGACY_SMOKE_FILE_SHA256,
        )
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": str(exc)}))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
