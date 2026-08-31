"""Seal the test-fixture-only correction discovered by the full Plan-017 suite."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import prepare_brainiac_raw_t1_v2 as runner  # noqa: E402
from src.research import brainiac_preprocessing_v2 as v2  # noqa: E402
from src.research.brainiac_action_guard import inspect_action_exclusion  # noqa: E402
from src.research.brainiac_method_x_isolation import (  # noqa: E402
    verify_frozen_method_x_source_contract,
)


OLD_GIT_HEAD = "9b6ad39bf15827b03feb5a21e7ae360ee7afaeed"
EXPECTED_RUNNER_SHA256 = (
    "c440107631ebf835e145c279942d016b8e4df36d750040e2180c45aeb0d0dffd"
)
EXPECTED_DEPENDENCY_CLOSURE_SHA256 = (
    "9c2cca27c27931b65b43d1e55c39a31e15a2d2be5420d113596b0772cff20604"
)
FAILED_JUNIT = (
    runner.REGISTRY_ROOT
    / "test_runs"
    / "full"
    / "20260831T142702Z-e298d24aec44"
    / "junit.xml"
)
REPORT = runner.REGISTRY_ROOT / "BRAINIAC_PLAN017_TEST_FIXTURE_SUPERSESSION_V5.json"
BLOCKED = runner.REGISTRY_ROOT / "BLOCKED_BRAINIAC_PLAN017_TEST_FIXTURE_SUPERSESSION_V5.json"
ARCHIVE = (
    runner.REGISTRY_ROOT
    / "superseded"
    / "plan017_test_fixture_supersession_v5"
)
EXPECTED_FAILURES = {
    "tests.test_brainiac_preprocessing_speedup::test_acceptance_passes_same_stop_flag_to_calibration",
    "tests.test_brainiac_preprocessing_speedup::test_acceptance_candidate_cannot_create_final_readiness",
}


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
        raise RuntimeError("BRAINIAC_TEST_FIXTURE_GIT_INSPECTION_FAILED")
    return completed.stdout.strip().casefold()


def _load_sealed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if recorded != runner.canonical_sha(body):
        raise RuntimeError("BRAINIAC_TEST_FIXTURE_EVIDENCE_INVALID")
    return payload


def _failed_nodes() -> set[str]:
    root = ET.parse(FAILED_JUNIT).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    nodes: set[str] = set()
    failures = errors = tests = skipped = 0
    for suite in suites:
        tests += int(suite.get("tests", "0"))
        failures += int(suite.get("failures", "0"))
        errors += int(suite.get("errors", "0"))
        skipped += int(suite.get("skipped", "0"))
        for case in suite.findall("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                nodes.add(
                    f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
                )
    if (tests, failures, errors, skipped) != (595, 2, 0, 0) or nodes != EXPECTED_FAILURES:
        raise RuntimeError("BRAINIAC_TEST_FIXTURE_FAILED_JUNIT_UNEXPECTED")
    return nodes


def _archive(path: Path, role: str) -> dict[str, Any]:
    role_root = ARCHIVE / role
    if path.is_file():
        payload = _load_sealed(path)
        if role == "acceptance_candidate" and (
            payload.get("git_state", {}).get("head") != OLD_GIT_HEAD
            or payload.get("primary_valid_receipts") != 0
        ):
            raise RuntimeError("BRAINIAC_TEST_FIXTURE_OLD_CANDIDATE_INVALID")
        digest = runner.sha256_file(path)
        target = role_root / f"sha256-{digest}" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError("BRAINIAC_TEST_FIXTURE_ARCHIVE_COLLISION")
        os.replace(path, target)
    else:
        candidates = sorted(role_root.glob(f"sha256-*/{path.name}"))
        if len(candidates) != 1:
            raise RuntimeError("BRAINIAC_TEST_FIXTURE_ARCHIVE_MISSING")
        target = candidates[0]
        _load_sealed(target)
    return runner._artifact_binding(target)


def _primary_n() -> int:
    receipts = v2.PRIMARY_ROOT / "receipts"
    return len(list(receipts.glob("*.json"))) if receipts.is_dir() else 0


def seal() -> dict[str, Any]:
    if REPORT.is_file():
        return _load_sealed(REPORT)
    exclusion = inspect_action_exclusion()
    if exclusion.get("status") != "PASS" or exclusion.get("blockers"):
        raise RuntimeError("BRAINIAC_TEST_FIXTURE_WRITER_EXCLUSION_BLOCKED")
    if _primary_n() != 0:
        raise RuntimeError("BRAINIAC_TEST_FIXTURE_PRIMARY_ALREADY_STARTED")
    head = _git_head()
    if head == OLD_GIT_HEAD:
        raise RuntimeError("BRAINIAC_TEST_FIXTURE_FIX_NOT_COMMITTED")
    runner_sha = runner.sha256_file(ROOT / "scripts/research/prepare_brainiac_raw_t1_v2.py")
    dependency = runner._dependency_manifest()
    if (
        runner_sha != EXPECTED_RUNNER_SHA256
        or dependency.get("closure_sha256") != EXPECTED_DEPENDENCY_CLOSURE_SHA256
    ):
        raise RuntimeError("BRAINIAC_TEST_FIXTURE_PRODUCTION_SOURCE_DRIFT")
    source = verify_frozen_method_x_source_contract(
        expected_digest=runner.EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    if runner.sha256_file(runner.BATTERY_MANIFEST_PATH) != runner.EXPECTED_BATTERY_MANIFEST_SHA256:
        raise RuntimeError("BRAINIAC_TEST_FIXTURE_BATTERY_DRIFT")
    failures = _failed_nodes()
    archived = {
        "acceptance_candidate": _archive(
            runner.ACCEPTANCE_CANDIDATE_PATH, "acceptance_candidate"
        ),
        "acceptance_preflight_snapshot": _archive(
            runner.ACCEPTANCE_PREFLIGHT_SNAPSHOT_PATH,
            "acceptance_preflight_snapshot",
        ),
        "acceptance_test_manifest": _archive(
            runner.ACCEPTANCE_TEST_MANIFEST_PATH, "acceptance_test_manifest"
        ),
    }
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PLAN017_TEST_FIXTURE_SUPERSESSION_V5",
        "status": "PASS_TEST_FIXTURE_ONLY_SUPERSESSION",
        "created_at_utc": _utc_now(),
        "old_git_head": OLD_GIT_HEAD,
        "new_git_head": head,
        "failed_full_test_junit": runner._artifact_binding(FAILED_JUNIT),
        "failed_test_nodeids": sorted(failures),
        "fixture_corrections": [
            "mock_v4_compatibility_for_stop_propagation_unit_scope",
            "mock_contract_complete_equivalence_bindings_for_candidate_unit_scope",
        ],
        "archived_stale_acceptance_evidence": archived,
        "production_runner_sha256": runner_sha,
        "dependency_closure_sha256": dependency["closure_sha256"],
        "method_x_source_digest": source["source_contract_digest"],
        "battery_manifest_sha256": runner.EXPECTED_BATTERY_MANIFEST_SHA256,
        "preprocessing_reexecuted": False,
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
