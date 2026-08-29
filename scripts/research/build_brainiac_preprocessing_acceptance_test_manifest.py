"""Run and seal the real-process BrainIAC preprocessing acceptance checks.

This generator is intentionally separate from MRI preprocessing.  It invokes
the exact hermetic fault/signal suite, binds the JUnit bytes and every relevant
source/test input, and only then writes the authoritative schema-2 manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import prepare_brainiac_raw_t1_v2 as runner  # noqa: E402
from src.research.brainiac_dependency_closure import (  # noqa: E402
    verify_manifest_tracking,
)


CHECK_SELECTORS: dict[str, tuple[str, ...]] = {
    "transaction_fault_injection": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_pre_receipt_faults_archive_partial_state_and_reprocess",
    ),
    "sigint_sequential_pause_resume": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_real_subprocess_sigint_sequential_pause_resume",
    ),
    "sigint_parallel_pause_resume": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_real_subprocess_sigint_parallel_pause_resume",
    ),
    "sigterm_sequential_pause_resume": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_real_subprocess_sigterm_sequential_pause_resume",
    ),
    "sigterm_parallel_pause_resume": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_real_subprocess_sigterm_parallel_pause_resume",
    ),
    "hard_kill_pause_resume": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_real_process_kill_restart_matches_uninterrupted_transaction_bytes",
    ),
    "corrupt_receipt_rejected": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_corrupt_committed_receipt_and_truncated_journal_fail_closed",
    ),
    "corrupt_output_transform_rejected": (
        "tests/test_brainiac_preprocessing_v2.py::test_output_hash_or_size_mismatch_rejects",
        "tests/test_brainiac_preprocessing_final_repairs.py::test_corrupt_transform_is_rejected_from_committed_receipt",
    ),
    "journal_truncation_rejected": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_corrupt_committed_receipt_and_truncated_journal_fail_closed",
    ),
    "receipt_before_journal_recovered": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_receipt_without_final_journal_recovers_without_reprocessing",
    ),
    "reparse_lock_pid_reuse_rejected": (
        "tests/test_brainiac_preprocessing_v2.py::test_reparse_output_rejects",
        "tests/test_brainiac_preprocessing_final_repairs.py::test_sample_lock_pid_reuse_is_rejected_without_archiving",
        "tests/test_brainiac_preprocessing_final_repairs.py::test_proven_dead_sample_lock_is_archived_with_reason",
    ),
    "profile_live_byte_revalidation": (
        "tests/test_brainiac_preprocessing_final_repairs.py::test_execution_profile_recomputes_live_receipts_and_rejects_byte_drift",
    ),
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_GIT_INSPECTION_FAILED")
    return completed.stdout.strip()


def _manifest_input_paths(dependency: Mapping[str, Any]) -> list[Path]:
    relative: set[str] = set(map(str, dict(dependency["python_closure"]).keys()))
    for role in ("dynamic_code_inputs", "static_inputs"):
        for binding in dict(dependency.get(role) or {}).values():
            if isinstance(binding, Mapping):
                relative.add(str(binding["path"]))
    relative.add(Path(__file__).resolve().relative_to(ROOT).as_posix())
    relative.add("tests/fixtures/brainiac_transaction_worker.py")
    for selectors in CHECK_SELECTORS.values():
        relative.update(selector.split("::", 1)[0] for selector in selectors)
    return sorted((ROOT / item).resolve(strict=True) for item in relative)


def _assert_scoped_git_clean(paths: list[Path]) -> None:
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0 or completed.stdout.strip():
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_SCOPED_SOURCE_DIRTY")


def _junit_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_JUNIT_INVALID")
    return {
        "passed": sum(
            int(suite.attrib.get("tests", "0"))
            - int(suite.attrib.get("failures", "0"))
            - int(suite.attrib.get("errors", "0"))
            - int(suite.attrib.get("skipped", "0"))
            for suite in suites
        ),
        "failed": sum(int(suite.attrib.get("failures", "0")) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", "0")) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", "0")) for suite in suites),
        "collected": sum(int(suite.attrib.get("tests", "0")) for suite in suites),
    }


def _validate_manifest_self_hash(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    recorded = str(body.pop("integrity_sha256", ""))
    if len(recorded) != 64 or recorded != runner.canonical_sha(body):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_EXISTING_MANIFEST_CORRUPT")


def _binding_is_live(binding: Mapping[str, Any]) -> bool:
    try:
        path = (ROOT / str(binding["path"])).resolve(strict=True)
        path.relative_to(ROOT.resolve(strict=True))
        return (
            path.is_file()
            and int(binding["bytes"]) == path.stat().st_size
            and str(binding["sha256"]) == runner.sha256_file(path)
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _current_manifest_is_reusable(
    payload: Mapping[str, Any], dependency: Mapping[str, Any]
) -> bool:
    _validate_manifest_self_hash(payload)
    if (
        payload.get("schema_version") != 2
        or payload.get("artifact")
        != "BRAINIAC_PREPROCESSING_ACCEPTANCE_TEST_MANIFEST"
        or payload.get("status") != "PASS"
        or str(payload.get("git_head", "")).casefold()
        != _git("rev-parse", "HEAD").casefold()
        or payload.get("dependency_closure_sha256") != dependency.get("closure_sha256")
        or set(dict(payload.get("required_checks") or {})) != set(CHECK_SELECTORS)
    ):
        return False
    summary = payload.get("summary")
    evidence = payload.get("evidence_bindings")
    if (
        not isinstance(summary, Mapping)
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("errors", -1)) != 0
        or int(summary.get("skipped", -1)) != 0
        or not isinstance(evidence, Mapping)
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_EXISTING_MANIFEST_INVALID")
    bindings = [evidence.get("junit")]
    bindings.extend(list(evidence.get("test_files") or ()))
    bindings.extend(list(evidence.get("source_files") or ()))
    if not bindings or not all(
        isinstance(binding, Mapping) and _binding_is_live(binding)
        for binding in bindings
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_EXISTING_EVIDENCE_DRIFT")
    return True


def _archive_superseded_manifest(
    target: Path, existing: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Path:
    """Preserve a valid old manifest and an append-only supersession receipt."""

    _validate_manifest_self_hash(existing)
    if (
        existing.get("schema_version") != 2
        or existing.get("artifact")
        != "BRAINIAC_PREPROCESSING_ACCEPTANCE_TEST_MANIFEST"
        or existing.get("status") != "PASS"
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_EXISTING_MANIFEST_INVALID")
    old_sha = str(existing["integrity_sha256"])
    new_sha = str(candidate["integrity_sha256"])
    archive_dir = target.parent / "superseded" / "acceptance_test_manifests"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(existing.get("created_at_utc") or "unknown").replace(":", "").replace(
        "+", "_"
    )
    archived = archive_dir / f"{stamp}-{old_sha}.json"
    if archived.exists():
        archived_payload = json.loads(archived.read_text(encoding="utf-8"))
        if archived_payload != dict(existing):
            raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_SUPERSEDED_COLLISION")
        target.unlink()
    else:
        os.replace(target, archived)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_ACCEPTANCE_TEST_MANIFEST_SUPERSESSION",
        "status": "SUPERSEDED_APPEND_ONLY",
        "created_at_utc": _utc_now(),
        "reason_code": "SOURCE_POLICY_OR_DEPENDENCY_CONTRACT_ADVANCED",
        "superseded_manifest": runner._artifact_binding(archived),
        "superseded_integrity_sha256": old_sha,
        "replacement_integrity_sha256": new_sha,
        "human_approval_required": False,
        "manual_override_allowed": False,
    }
    receipt["integrity_sha256"] = runner.canonical_sha(receipt)
    receipt_path = archive_dir / f"{old_sha}-to-{new_sha}.supersession.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
            raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_SUPERSESSION_COLLISION")
    else:
        runner._atomic_write_json(receipt_path, receipt)
    return archived


def build_manifest() -> dict[str, Any]:
    if platform.system() != "Windows":
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_WINDOWS_REQUIRED")
    if set(CHECK_SELECTORS) != runner.ACCEPTANCE_REQUIRED_CHECKS:
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_CHECK_SET_DRIFT")
    dependency = runner._dependency_manifest()
    if verify_manifest_tracking(dependency):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_DEPENDENCY_UNTRACKED")
    inputs = _manifest_input_paths(dependency)
    _assert_scoped_git_clean(inputs)
    target = runner.ACCEPTANCE_TEST_MANIFEST_PATH
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if _current_manifest_is_reusable(existing, dependency):
            return existing

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
    run_root = runner.REGISTRY_ROOT / "test_runs" / "acceptance" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    junit = run_root / "junit.xml"
    selectors = sorted({node for rows in CHECK_SELECTORS.values() for node in rows})
    command = [
        sys.executable,
        "-m",
        "pytest",
        *selectors,
        "--junitxml",
        str(junit),
        "--tb=short",
        "-q",
    ]
    started = _utc_now()
    started_clock = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        env=dict(os.environ),
    )
    elapsed = time.perf_counter() - started_clock
    finished = _utc_now()
    (run_root / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if not junit.is_file():
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_JUNIT_MISSING")
    summary = _junit_summary(junit)
    if (
        completed.returncode != 0
        or summary["failed"] != 0
        or summary["errors"] != 0
        or summary["skipped"] != 0
    ):
        raise RuntimeError("BRAINIAC_ACCEPTANCE_TEST_SUITE_FAILED")

    test_paths = sorted(
        {
            (ROOT / selector.split("::", 1)[0]).resolve(strict=True)
            for selectors_for_check in CHECK_SELECTORS.values()
            for selector in selectors_for_check
        }
    )
    source_paths = [path for path in inputs if path not in set(test_paths)]
    body: dict[str, Any] = {
        "schema_version": 2,
        "artifact": "BRAINIAC_PREPROCESSING_ACCEPTANCE_TEST_MANIFEST",
        "status": "PASS",
        "created_at_utc": finished,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "elapsed_seconds": round(elapsed, 6),
        "platform": platform.system(),
        "python_executable": str(Path(sys.executable).resolve()),
        "command": command,
        "git_head": _git("rev-parse", "HEAD").casefold(),
        "dependency_closure_sha256": dependency["closure_sha256"],
        "exit_code": int(completed.returncode),
        "summary": summary,
        "required_checks": {
            name: {"status": "PASS", "nodeids": list(nodeids)}
            for name, nodeids in sorted(CHECK_SELECTORS.items())
        },
        "evidence_bindings": {
            "junit": runner._artifact_binding(junit),
            "test_files": [runner._artifact_binding(path) for path in test_paths],
            "source_files": [runner._artifact_binding(path) for path in source_paths],
        },
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
    }
    body["integrity_sha256"] = runner.canonical_sha(body)
    runner._atomic_write_json(run_root / "manifest.json", body)

    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing == body:
            return existing
        _archive_superseded_manifest(target, existing, body)
    runner._atomic_write_json(target, body)
    runner._validated_acceptance_test_manifest()
    return body


def main() -> int:
    try:
        report = build_manifest()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": str(exc)}))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
