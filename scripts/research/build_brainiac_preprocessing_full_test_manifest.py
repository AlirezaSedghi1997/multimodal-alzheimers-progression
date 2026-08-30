"""Run the complete BrainIAC regression set and seal exact test evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
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
from src.research.brainiac_dependency_closure import verify_manifest_tracking  # noqa: E402


OUTPUT = runner.REGISTRY_ROOT / "brainiac_preprocessing_test_manifest_v1.json"
# Cross-method source isolation is exercised by
# ``test_brainiac_completion_gates.py`` inside the BrainIAC set.  Do not bind
# unrelated, dirty Method X/Y test files into this preprocessing release.
SHARED_TESTS: tuple[Path, ...] = ()


def _scientific_source_paths() -> set[Path]:
    """Return the complete Git-clean BrainIAC code/contract surface.

    AST closure from the preprocessing entrypoint is necessary but not
    sufficient for the full regression suite: several tests intentionally
    exercise later-stage scaffolds and JSON/YAML contracts through dynamic
    imports and file reads.  Bind that complete, narrowly named surface so a
    green suite can never certify untracked or mutable BrainIAC code.
    """

    paths: set[Path] = set()
    patterns = (
        "scripts/build_brainiac*.py",
        "scripts/research/*brainiac*.py",
        "src/research/brainiac*.py",
        "configs/research/brainiac*.yaml",
        "governance/brainiac*.json",
    )
    for pattern in patterns:
        paths.update(path.resolve() for path in ROOT.glob(pattern) if path.is_file())
    for fixture in (ROOT / "tests" / "fixtures").glob("brainiac*"):
        if fixture.is_file():
            paths.add(fixture.resolve())
        elif fixture.is_dir():
            paths.update(path.resolve() for path in fixture.rglob("*") if path.is_file())
    for explicit in (
        ROOT / ".gitattributes",
        ROOT / "src" / "research" / "brainiac_adapter.py",
    ):
        if explicit.is_file():
            paths.add(explicit.resolve())
    return paths


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
        raise RuntimeError("BRAINIAC_FULL_TEST_GIT_INSPECTION_FAILED")
    return completed.stdout.strip().casefold()


def _validate_existing_manifest(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    recorded = str(body.pop("integrity_sha256", ""))
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact") != "BRAINIAC_PREPROCESSING_FULL_TEST_MANIFEST"
        or payload.get("status") != "PASS"
        or len(recorded) != 64
        or recorded != runner.canonical_sha(body)
    ):
        raise RuntimeError("BRAINIAC_FULL_TEST_EXISTING_MANIFEST_INVALID")


def _archive_superseded_manifest(
    target: Path, existing: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Path:
    """Move a valid prior full-suite manifest with a sealed lineage receipt."""

    _validate_existing_manifest(existing)
    old_sha = str(existing["integrity_sha256"])
    new_sha = str(candidate["integrity_sha256"])
    archive_dir = target.parent / "superseded" / "full_test_manifests"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(existing.get("created_at_utc") or "unknown").replace(":", "").replace(
        "+", "_"
    )
    archived = archive_dir / f"{stamp}-{old_sha}.json"
    if archived.exists():
        if json.loads(archived.read_text(encoding="utf-8")) != dict(existing):
            raise RuntimeError("BRAINIAC_FULL_TEST_SUPERSEDED_COLLISION")
        target.unlink()
    else:
        os.replace(target, archived)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_FULL_TEST_MANIFEST_SUPERSESSION",
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
            raise RuntimeError("BRAINIAC_FULL_TEST_SUPERSESSION_COLLISION")
    else:
        runner._atomic_write_json(receipt_path, receipt)
    return archived


def _summary(junit: Path, stdout: str) -> dict[str, int]:
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise RuntimeError("BRAINIAC_FULL_TEST_JUNIT_INVALID")
    tests = sum(int(row.attrib.get("tests", "0")) for row in suites)
    failed = sum(int(row.attrib.get("failures", "0")) for row in suites)
    errors = sum(int(row.attrib.get("errors", "0")) for row in suites)
    skipped = sum(int(row.attrib.get("skipped", "0")) for row in suites)
    warning_matches = re.findall(r"(\d+) warnings?", stdout)
    warnings = int(warning_matches[-1]) if warning_matches else 0
    return {
        "passed": tests - failed - errors - skipped,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "warnings": warnings,
        "collected": tests,
    }


def _dependency_paths(manifest: Mapping[str, Any]) -> set[Path]:
    relative = set(map(str, dict(manifest["python_closure"]).keys()))
    for role in ("dynamic_code_inputs", "static_inputs"):
        for binding in dict(manifest.get(role) or {}).values():
            if isinstance(binding, Mapping):
                relative.add(str(binding["path"]))
    return {(ROOT / item).resolve(strict=True) for item in relative}


def _assert_scoped_clean(paths: set[Path]) -> None:
    relative = sorted(path.relative_to(ROOT).as_posix() for path in paths)
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0 or completed.stdout.strip():
        raise RuntimeError("BRAINIAC_FULL_TEST_SCOPED_SOURCE_DIRTY")


def build_manifest() -> dict[str, Any]:
    if platform.system() != "Windows":
        raise RuntimeError("BRAINIAC_FULL_TEST_WINDOWS_REQUIRED")
    tests = sorted((ROOT / "tests").glob("test_brainiac*.py"))
    tests.extend(path for path in SHARED_TESTS if path.is_file())
    tests = sorted(set(path.resolve(strict=True) for path in tests))
    if len(tests) < 20:
        raise RuntimeError("BRAINIAC_FULL_TEST_FILE_SET_INCOMPLETE")
    dependency = runner._dependency_manifest()
    if verify_manifest_tracking(dependency):
        raise RuntimeError("BRAINIAC_FULL_TEST_DEPENDENCY_UNTRACKED")
    source_paths = _dependency_paths(dependency) | _scientific_source_paths()
    source_paths.update(
        {
            Path(__file__).resolve(),
            (ROOT / "scripts" / "build_brainiac_p08_status.py").resolve(strict=True),
            (
                ROOT
                / "scripts"
                / "research"
                / "build_brainiac_novelty_matrix_v3.py"
            ).resolve(strict=True),
            (
                ROOT
                / "scripts"
                / "research"
                / "build_brainiac_preprocessing_acceptance_test_manifest.py"
            ).resolve(strict=True),
            (
                ROOT
                / "scripts"
                / "research"
                / "build_brainiac_preprocessing_release.py"
            ).resolve(strict=True),
        }
    )
    _assert_scoped_clean(source_paths | set(tests))

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
    run_root = runner.REGISTRY_ROOT / "test_runs" / "full" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    junit = run_root / "junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *[str(path) for path in tests],
        "--junitxml",
        str(junit),
        "--tb=short",
        "-q",
    ]
    started = _utc_now()
    clock = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
        env=dict(os.environ),
    )
    elapsed = time.perf_counter() - clock
    finished = _utc_now()
    (run_root / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if not junit.is_file():
        raise RuntimeError("BRAINIAC_FULL_TEST_JUNIT_MISSING")
    summary = _summary(junit, completed.stdout)
    if completed.returncode != 0 or summary["failed"] or summary["errors"]:
        raise RuntimeError("BRAINIAC_FULL_TEST_SUITE_FAILED")

    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PREPROCESSING_FULL_TEST_MANIFEST",
        "status": "PASS",
        "created_at_utc": finished,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "elapsed_seconds": round(elapsed, 6),
        "platform": platform.system(),
        "python_executable": str(Path(sys.executable).resolve()),
        "command": command,
        "git_head": _git_head(),
        "exit_code": int(completed.returncode),
        "summary": summary,
        "dependency_closure_sha256": dependency["closure_sha256"],
        "test_files": [runner._artifact_binding(path) for path in tests],
        "source_files": [
            runner._artifact_binding(path) for path in sorted(source_paths)
        ],
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
    }
    body["integrity_sha256"] = runner.canonical_sha(body)
    runner._atomic_write_json(run_root / "manifest.json", body)
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if existing == body:
            return existing
        _archive_superseded_manifest(OUTPUT, existing, body)
    runner._atomic_write_json(OUTPUT, body)
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
