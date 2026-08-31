"""Supersede the transactionally stale first Plan-018 release append-only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import prepare_brainiac_raw_t1_v2 as runner  # noqa: E402
from src.research import brainiac_preprocessing_v2 as v2  # noqa: E402
from src.research.brainiac_action_guard import inspect_action_exclusion  # noqa: E402
from src.research.brainiac_method_x_isolation import (  # noqa: E402
    verify_frozen_method_x_source_contract,
)


OLD_GIT_HEAD = "4b15ecdc2ca3ddc81032808729be0da4672154c2"
EXPECTED_RUNNER_SHA256 = (
    "c440107631ebf835e145c279942d016b8e4df36d750040e2180c45aeb0d0dffd"
)
EXPECTED_DEPENDENCY_CLOSURE_SHA256 = (
    "9c2cca27c27931b65b43d1e55c39a31e15a2d2be5420d113596b0772cff20604"
)
PARENT_V6 = runner.REGISTRY_ROOT / "BRAINIAC_PLAN017_TEST_FIXTURE_SUPERSESSION_V6.json"
REPORT = runner.REGISTRY_ROOT / "BRAINIAC_PLAN018_LIVE_STATUS_SUPERSESSION_V7.json"
BLOCKED = runner.REGISTRY_ROOT / "BLOCKED_BRAINIAC_PLAN018_LIVE_STATUS_SUPERSESSION_V7.json"
ARCHIVE = (
    runner.REGISTRY_ROOT / "superseded" / "plan018_live_status_supersession_v7"
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError("BRAINIAC_PLAN018_V7_GIT_INSPECTION_FAILED")
    return completed.stdout.strip().casefold()


def _load_sealed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if recorded != runner.canonical_sha(body):
        raise RuntimeError("BRAINIAC_PLAN018_V7_SEALED_EVIDENCE_INVALID")
    return payload


def _archive_json(path: Path, role: str) -> dict[str, Any]:
    role_root = ARCHIVE / role
    if path.is_file():
        _load_sealed(path)
        digest = runner.sha256_file(path)
        target = role_root / f"sha256-{digest}" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError("BRAINIAC_PLAN018_V7_ARCHIVE_COLLISION")
        os.replace(path, target)
    else:
        candidates = sorted(role_root.glob(f"sha256-*/{path.name}"))
        if len(candidates) != 1:
            raise RuntimeError("BRAINIAC_PLAN018_V7_ARCHIVE_MISSING")
        target = candidates[0]
        _load_sealed(target)
    return runner._artifact_binding(target)


def _copy_bound_document(binding: Mapping[str, Any]) -> dict[str, Any]:
    source = runner._validate_artifact_binding(binding)
    relative = source.relative_to(ROOT)
    target = ARCHIVE / "documentation" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if runner.sha256_file(target) != binding.get("sha256"):
            raise RuntimeError("BRAINIAC_PLAN018_V7_DOCUMENT_ARCHIVE_COLLISION")
    else:
        shutil.copy2(source, target)
    archived = runner._artifact_binding(target)
    if (
        archived.get("sha256") != binding.get("sha256")
        or archived.get("bytes") != binding.get("bytes")
    ):
        raise RuntimeError("BRAINIAC_PLAN018_V7_DOCUMENT_ARCHIVE_MISMATCH")
    return {
        "original": dict(binding),
        "archived": archived,
    }


def _primary_n() -> int:
    receipts = v2.PRIMARY_ROOT / "receipts"
    return len(list(receipts.glob("*.json"))) if receipts.is_dir() else 0


def seal() -> dict[str, Any]:
    if REPORT.is_file():
        return _load_sealed(REPORT)
    parent = _load_sealed(PARENT_V6)
    if (
        parent.get("status") != "PASS_TEST_FIXTURE_CHAIN_SUPERSESSION"
        or parent.get("new_git_head") != OLD_GIT_HEAD
    ):
        raise RuntimeError("BRAINIAC_PLAN018_V7_PARENT_INVALID")
    exclusion = inspect_action_exclusion()
    if exclusion.get("status") != "PASS" or exclusion.get("blockers"):
        raise RuntimeError("BRAINIAC_PLAN018_V7_WRITER_EXCLUSION_BLOCKED")
    if _primary_n() != 0:
        raise RuntimeError("BRAINIAC_PLAN018_V7_PRIMARY_ALREADY_STARTED")
    head = _git_head()
    if head == OLD_GIT_HEAD:
        raise RuntimeError("BRAINIAC_PLAN018_V7_FIX_NOT_COMMITTED")
    runner_sha = runner.sha256_file(ROOT / "scripts/research/prepare_brainiac_raw_t1_v2.py")
    dependency = runner._dependency_manifest()
    if (
        runner_sha != EXPECTED_RUNNER_SHA256
        or dependency.get("closure_sha256") != EXPECTED_DEPENDENCY_CLOSURE_SHA256
    ):
        raise RuntimeError("BRAINIAC_PLAN018_V7_PRODUCTION_SOURCE_DRIFT")
    source_contract = verify_frozen_method_x_source_contract(
        expected_digest=runner.EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    if runner.sha256_file(runner.BATTERY_MANIFEST_PATH) != runner.EXPECTED_BATTERY_MANIFEST_SHA256:
        raise RuntimeError("BRAINIAC_PLAN018_V7_BATTERY_DRIFT")

    old_candidate = _load_sealed(runner.ACCEPTANCE_CANDIDATE_PATH)
    old_full_tests = _load_sealed(runner.FULL_TEST_MANIFEST_PATH)
    old_trace = _load_sealed(runner.TRACEABILITY_MANIFEST_PATH)
    old_ready = _load_sealed(runner.READY_REPORT_PATH)
    if (
        old_candidate.get("git_state", {}).get("head") != OLD_GIT_HEAD
        or old_full_tests.get("git_head") != OLD_GIT_HEAD
        or old_full_tests.get("summary", {}).get("failed") != 0
        or old_full_tests.get("summary", {}).get("errors") != 0
        or old_trace.get("facts", {}).get("plan_018")
        != "IN_PROGRESS_ISSUING_READINESS"
        or old_trace.get("facts", {}).get("plan_019")
        != "BLOCKED_PENDING_FINAL_READINESS"
        or old_ready.get("status")
        != "READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING"
        or old_ready.get("primary_valid_n_at_issuance") != 0
    ):
        raise RuntimeError("BRAINIAC_PLAN018_V7_PREDECESSOR_STATE_INVALID")

    documentation = old_trace.get("documentation_bindings")
    if not isinstance(documentation, list) or len(documentation) != 6:
        raise RuntimeError("BRAINIAC_PLAN018_V7_DOCUMENTATION_BINDINGS_INVALID")
    archived_documents = [_copy_bound_document(row) for row in documentation]
    archived = {
        "acceptance_candidate": _archive_json(
            runner.ACCEPTANCE_CANDIDATE_PATH, "acceptance_candidate"
        ),
        "acceptance_preflight_snapshot": _archive_json(
            runner.ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH,
            "acceptance_preflight_snapshot",
        ),
        "acceptance_test_manifest": _archive_json(
            runner.ACCEPTANCE_TEST_MANIFEST_PATH, "acceptance_test_manifest"
        ),
        "full_test_manifest": _archive_json(
            runner.FULL_TEST_MANIFEST_PATH, "full_test_manifest"
        ),
        "traceability_manifest": _archive_json(
            runner.TRACEABILITY_MANIFEST_PATH, "traceability_manifest"
        ),
        "readiness": _archive_json(runner.READY_REPORT_PATH, "readiness"),
    }
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PLAN018_LIVE_STATUS_SUPERSESSION_V7",
        "status": "PASS_PLAN018_LIVE_STATUS_SUPERSESSION",
        "created_at_utc": _utc_now(),
        "parent_v6": runner._artifact_binding(PARENT_V6),
        "old_git_head": OLD_GIT_HEAD,
        "new_git_head": head,
        "semantic_correction": {
            "removed_transactional_fact_keys": ["plan_018", "plan_019"],
            "removed_stale_text": "Plan 019 waits only for final preprocessing readiness",
            "replacement_text": "Plan 019 may start only when final readiness validates live",
        },
        "archived_predecessor_artifacts": archived,
        "archived_predecessor_documentation": archived_documents,
        "superseded_full_test_manifest": archived["full_test_manifest"],
        "production_runner_sha256": runner_sha,
        "dependency_closure_sha256": dependency["closure_sha256"],
        "method_x_source_digest": source_contract["source_contract_digest"],
        "battery_manifest_sha256": runner.EXPECTED_BATTERY_MANIFEST_SHA256,
        "preprocessing_reexecuted": False,
        "outer_loader_invocation_count": 0,
        "primary_valid_n": 0,
        "plans_005_006_007_allowed": False,
        "training_authorization": False,
        "manual_override_allowed": False,
        "human_approval_required": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    body["integrity_sha256"] = runner.canonical_sha(body)
    runner._atomic_write_json(REPORT, body)
    return body


def main() -> int:
    try:
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
        runner._atomic_write_json(BLOCKED, blocked)
        print(json.dumps(blocked, sort_keys=True))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
