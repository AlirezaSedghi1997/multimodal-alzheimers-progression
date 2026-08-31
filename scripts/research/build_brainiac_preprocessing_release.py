"""Plan 018: derive traceability and issue preprocessing-only readiness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_brainiac_p08_status as p08_builder  # noqa: E402
from scripts.research import prepare_brainiac_raw_t1_v2 as runner  # noqa: E402
from scripts.research.build_brainiac_novelty_matrix_v3 import (  # noqa: E402
    materialize as materialize_novelty,
)
from src.research.brainiac_action_guard import inspect_action_exclusion  # noqa: E402
from src.research.brainiac_method_x_isolation import (  # noqa: E402
    verify_frozen_method_x_source_contract,
)
from src.utils.io_helpers import atomic_write_json  # noqa: E402


ROADMAP_JSON = ROOT / "results" / "method_x_registry" / "project_roadmap.json"
ROADMAP_MD = ROOT / "docs" / "plans" / "method_x_project_roadmap.md"
MEMORY_INDEX = ROOT / "docs" / "PROJECT_MEMORY_INDEX.md"
SDD_PROGRESS = (
    ROOT
    / ".superpowers"
    / "sdd"
    / "brainiac_caft_training_readiness_20260822"
    / "progress.md",
    ROOT
    / ".superpowers"
    / "sdd"
    / "method_x_brainiac_caft_execution_tasks_20260808"
    / "progress.md",
)
ADVISOR_README = ROOT / "advisor-plans" / "README.md"
SECTION_START = "<!-- BRAINIAC_PREPROCESSING_PLAN018_LIVE_START -->"
SECTION_END = "<!-- BRAINIAC_PREPROCESSING_PLAN018_LIVE_END -->"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded = text.encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _write_generated_section(path: Path, content: str) -> None:
    original = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = f"{SECTION_START}\n{content.rstrip()}\n{SECTION_END}"
    has_start = SECTION_START in original
    has_end = SECTION_END in original
    if has_start != has_end:
        raise RuntimeError("BRAINIAC_TRACEABILITY_DOCUMENT_MARKER_CORRUPT")
    if has_start:
        prefix, remainder = original.split(SECTION_START, 1)
        _, suffix = remainder.split(SECTION_END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = original.rstrip() + "\n\n" + block + "\n"
    if updated != original:
        _atomic_write_text(path, updated)


def _validate_sealed(path: Path, *, expected_status: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    recorded = body.pop("integrity_sha256", None)
    if payload.get("status") != expected_status or recorded != runner.canonical_sha(body):
        raise RuntimeError("BRAINIAC_PLAN018_SEALED_ARTIFACT_INVALID")
    return payload


def _validate_full_test_fixture_lineage(full_tests: Mapping[str, Any]) -> None:
    expected = (
        runner.REGISTRY_ROOT / "BRAINIAC_PLAN017_TEST_FIXTURE_SUPERSESSION_V5.json"
    )
    binding = full_tests.get("test_fixture_supersession")
    if (
        not isinstance(binding, Mapping)
        or runner._validate_artifact_binding(binding) != expected.resolve(strict=True)
    ):
        raise RuntimeError("BRAINIAC_PLAN018_TEST_FIXTURE_LINEAGE_INVALID")
    _validate_sealed(
        expected, expected_status="PASS_TEST_FIXTURE_ONLY_SUPERSESSION"
    )


def _materialize_p08() -> dict[str, Any]:
    report = p08_builder.build_status()
    target = p08_builder.OUTPUT
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError("BRAINIAC_P08_STATUS_IMMUTABLE_MISMATCH")
        return existing
    atomic_write_json(target, report)
    return report


def _preservation_report() -> dict[str, Any]:
    freeze = json.loads(runner.PRE_REPAIR_LIVE_FREEZE_PATH.read_text(encoding="utf-8"))
    body = dict(freeze)
    recorded = body.pop("integrity_sha256", None)
    snapshot = freeze.get("snapshot")
    if (
        freeze.get("schema_version") != 2
        or not isinstance(snapshot, Mapping)
        or recorded != runner.canonical_sha(body)
        or freeze.get("snapshot_sha256") != runner.canonical_sha(snapshot)
    ):
        raise RuntimeError("BRAINIAC_PLAN018_FREEZE_INVALID")
    frozen_inventory = snapshot.get("battery_finalized_inventory")
    if not isinstance(frozen_inventory, Mapping):
        raise RuntimeError("BRAINIAC_PLAN018_FREEZE_INVALID")
    live_inventory = runner.build_file_inventory(
        runner._battery_preservation_paths(), project_root=ROOT
    )
    if live_inventory != frozen_inventory:
        raise RuntimeError("BRAINIAC_PLAN018_FULL_BATTERY_TREE_DRIFT")
    source = verify_frozen_method_x_source_contract(
        expected_digest=runner.EXPECTED_METHOD_X_SOURCE_DIGEST
    )
    if runner.sha256_file(runner.BATTERY_MANIFEST_PATH) != runner.EXPECTED_BATTERY_MANIFEST_SHA256:
        raise RuntimeError("BRAINIAC_PLAN018_BATTERY_MANIFEST_DRIFT")
    return {
        "status": "PASS",
        "method_x_source_digest": source["source_contract_digest"],
        "battery_manifest_sha256": runner.EXPECTED_BATTERY_MANIFEST_SHA256,
        "battery_inventory_sha256": live_inventory["inventory_sha256"],
        "battery_file_count": live_inventory["file_count"],
        "battery_total_bytes": live_inventory["total_bytes"],
        "outer_loader_invocation_count": 0,
        "finalized_artifact_rewrite_count": 0,
    }


def _facts(full_tests: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "r0": "R0_COMPLETE",
        "citation_corroboration": "CITATION_CORROBORATION_PARTIAL",
        "novelty": "NOVELTY_CANDIDATE",
        "battery": "VERIFIED_TERMINAL",
        "plan_002": "ACCEPTED_READY_FOR_PRIMARY",
        "plan_003": "NOT_STARTED_PRIMARY_0_OF_1669",
        "plan_014": "DONE",
        "plan_015": "DONE",
        "plan_016": "DONE",
        "plan_017": "PASS_ACCEPTANCE_PRIMARY_NOT_STARTED",
        "plan_018": "IN_PROGRESS_ISSUING_READINESS",
        "plan_019": "BLOCKED_PENDING_FINAL_READINESS",
        "plans_005_006_007_allowed": False,
        "plan_005": "NOT_EXECUTED",
        "plan_006": "CODE_ONLY_UNACCEPTED",
        "plan_007": "NOT_EXECUTED",
        "external": "OPTIONAL_UNAVAILABLE_NONBLOCKING",
        "external_evidence_label": "ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION",
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "test_passed": int(full_tests["summary"]["passed"]),
        "test_failed": int(full_tests["summary"]["failed"]),
        "test_errors": int(full_tests["summary"]["errors"]),
    }


def _update_human_traceability(facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    section = f"""## BrainIAC preprocessing — generated live status (Plan 018)

- R0: `{facts['r0']}`; citation: `{facts['citation_corroboration']}`; novelty: `{facts['novelty']}`.
- Five-CNN battery: `{facts['battery']}`.
- Plan 002: `{facts['plan_002']}`; Plan 003: `{facts['plan_003']}`.
- Plans 014–017: `DONE/DONE/DONE/PASS`; Plan 019 waits only for final preprocessing readiness.
- Plans 005–007: `NOT_EXECUTED / CODE_ONLY_UNACCEPTED / NOT_EXECUTED` and remain hard-disabled until `1669/1669`.
- Tests: `{facts['test_passed']} passed`, `{facts['test_failed']} failed`, `{facts['test_errors']} errors` (live manifest-bound run).
- External data: `{facts['external']}`; internal substitute is `{facts['external_evidence_label']}`.
- Claim ceiling: `{facts['claim_ceiling']}`.
- Runtime counts must be read from `prepare_brainiac_raw_t1_v2.py --mode status`; no PID/count in this section is treated as timeless.
"""
    for path in (ROADMAP_MD, MEMORY_INDEX, *SDD_PROGRESS, ADVISOR_README):
        _write_generated_section(path, section)

    roadmap = json.loads(ROADMAP_JSON.read_text(encoding="utf-8"))
    roadmap["brainiac_preprocessing"] = dict(facts)
    roadmap["brainiac_preprocessing"]["status"] = "ACCEPTED_READY_FOR_PRIMARY"
    roadmap["brainiac_preprocessing"]["primary_expected_n"] = 1669
    roadmap["brainiac_preprocessing"]["primary_valid_n"] = 0
    roadmap["brainiac_preprocessing"]["next_command"] = (
        "H:\\New project\\.venvs\\brainiac\\Scripts\\python.exe "
        "H:\\New project\\scripts\\research\\prepare_brainiac_raw_t1_v2.py "
        "--mode full --device cuda"
    )
    atomic_write_json(ROADMAP_JSON, roadmap)
    return [
        runner._artifact_binding(path)
        for path in (ROADMAP_JSON, ROADMAP_MD, MEMORY_INDEX, *SDD_PROGRESS, ADVISOR_README)
    ]


def build_traceability() -> dict[str, Any]:
    candidate = runner.validate_acceptance_candidate()
    full_tests = runner._validated_full_test_manifest()
    _validate_full_test_fixture_lineage(full_tests)
    novelty = _validate_sealed(
        runner.NOVELTY_MATRIX_V3_PATH,
        expected_status="R0_COMPLETE_NOVELTY_CANDIDATE",
    )
    p08 = _validate_sealed(
        runner.P08_STATUS_PATH,
        expected_status="R0_COMPLETE_PREPARATION_PARTIAL_NONBLOCKING_FOR_MAIN_CNN",
    )
    if (
        p08.get("brainiac_training_allowed") is not False
        or p08.get("r0", {}).get("novelty_status") != "NOVELTY_CANDIDATE"
        or novelty.get("registry_sha256")
        != "e397b9dd03653bf7d4c3b80e470b8ffa9389b40e04a2208530bb5eb5c0a7cd8f"
    ):
        raise RuntimeError("BRAINIAC_PLAN018_CLAIM_OR_LINKAGE_DRIFT")
    status = runner.run_status()
    if (
        status.get("status") != "PASS"
        or int(status.get("expected_rows", -1)) != 1669
        or int(status.get("completed_rows", -1)) != 0
        or status.get("invalid_rows")
    ):
        raise RuntimeError("BRAINIAC_PLAN018_PRIMARY_NOT_ZERO_OR_INVALID")
    # Do not mutate any human-facing status document until the complete frozen
    # battery tree and Method X source contract have been revalidated.  This is
    # intentionally repeated in ``issue_readiness`` so a mutation between the
    # traceability and readiness transactions also fails closed.
    preservation = _preservation_report()
    facts = _facts(full_tests)
    documents = _update_human_traceability(facts)
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PREPROCESSING_TRACEABILITY_MANIFEST",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "facts": facts,
        "evidence_bindings": {
            "acceptance_candidate": runner._artifact_binding(
                runner.ACCEPTANCE_CANDIDATE_PATH
            ),
            "full_test_manifest": runner._artifact_binding(
                runner.FULL_TEST_MANIFEST_PATH
            ),
            "p08_status": runner._artifact_binding(runner.P08_STATUS_PATH),
            "novelty_matrix": runner._artifact_binding(
                runner.NOVELTY_MATRIX_V3_PATH
            ),
            "dependency_manifest": runner._artifact_binding(
                runner.DEPENDENCY_MANIFEST_PATH
            ),
            "source_only_resume_migration": runner._artifact_binding(
                runner.SOURCE_ONLY_RESUME_MIGRATION_PATH
            ),
            "postcalibration_source_migration": runner._artifact_binding(
                runner.POSTCALIBRATION_SOURCE_MIGRATION_PATH
            ),
            "primary_binding_source_compatibility": runner._artifact_binding(
                runner.PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH
            ),
            "calibration_reconciliation_source_compatibility": runner._artifact_binding(
                runner.CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_PATH
            ),
            "legacy_smoke_reconciliation": runner._artifact_binding(
                runner.LEGACY_SMOKE_RECONCILIATION_PATH
            ),
        },
        "documentation_bindings": documents,
        "full_battery_tree_preservation_pre_traceability": preservation,
        "candidate_integrity_sha256": candidate["integrity_sha256"],
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
    }
    body["integrity_sha256"] = runner.canonical_sha(body)
    target = runner.TRACEABILITY_MANIFEST_PATH
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        existing_semantic = dict(existing)
        candidate_semantic = dict(body)
        for row in (existing_semantic, candidate_semantic):
            row.pop("integrity_sha256", None)
            row.pop("created_at_utc", None)
        if existing_semantic != candidate_semantic:
            raise RuntimeError("BRAINIAC_TRACEABILITY_IMMUTABLE_MISMATCH")
        return existing
    runner._atomic_write_json(target, body)
    return body


def issue_readiness() -> dict[str, Any]:
    candidate = runner.validate_acceptance_candidate()
    full_tests = runner._validated_full_test_manifest()
    _validate_full_test_fixture_lineage(full_tests)
    traceability = runner._validated_traceability_manifest()
    exclusion = inspect_action_exclusion()
    if exclusion.get("status") != "PASS" or exclusion.get("blockers"):
        raise RuntimeError("BRAINIAC_PLAN018_WRITER_EXCLUSION_BLOCKED")
    status = runner.run_status()
    if (
        status.get("status") != "PASS"
        or int(status.get("expected_rows", -1)) != 1669
        or int(status.get("completed_rows", -1)) != 0
        or status.get("invalid_rows")
    ):
        raise RuntimeError("BRAINIAC_PLAN018_PRIMARY_NOT_ZERO_OR_INVALID")
    preservation = _preservation_report()
    evidence = {
        "acceptance_candidate": runner._artifact_binding(
            runner.ACCEPTANCE_CANDIDATE_PATH
        ),
        "full_test_manifest": runner._artifact_binding(
            runner.FULL_TEST_MANIFEST_PATH
        ),
        "traceability_manifest": runner._artifact_binding(
            runner.TRACEABILITY_MANIFEST_PATH
        ),
        "p08_status": runner._artifact_binding(runner.P08_STATUS_PATH),
        "novelty_matrix": runner._artifact_binding(
            runner.NOVELTY_MATRIX_V3_PATH
        ),
        "dependency_manifest": runner._artifact_binding(
            runner.DEPENDENCY_MANIFEST_PATH
        ),
        "source_only_resume_migration": runner._artifact_binding(
            runner.SOURCE_ONLY_RESUME_MIGRATION_PATH
        ),
        "postcalibration_source_migration": runner._artifact_binding(
            runner.POSTCALIBRATION_SOURCE_MIGRATION_PATH
        ),
        "primary_binding_source_compatibility": runner._artifact_binding(
            runner.PRIMARY_BINDING_SOURCE_COMPATIBILITY_PATH
        ),
        "calibration_reconciliation_source_compatibility": runner._artifact_binding(
            runner.CALIBRATION_RECONCILIATION_SOURCE_COMPATIBILITY_PATH
        ),
        "legacy_smoke_reconciliation": runner._artifact_binding(
            runner.LEGACY_SMOKE_RECONCILIATION_PATH
        ),
        "preflight": candidate["evidence_bindings"]["preflight"],
        "execution_profile": candidate["evidence_bindings"]["execution_profile"],
        "threshold_manifest": candidate["evidence_bindings"]["threshold_manifest"],
        "smoke_report": candidate["evidence_bindings"]["smoke_report"],
        "pre_repair_freeze": candidate["evidence_bindings"]["pre_repair_freeze"],
        "battery_manifest": runner._artifact_binding(runner.BATTERY_MANIFEST_PATH),
    }
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING",
        "status": "READY_TO_START_OR_RESUME_BRAINIAC_PREPROCESSING",
        "created_at_utc": _utc_now(),
        "completion": False,
        "training_authorization": False,
        "primary_expected_n": 1669,
        "primary_valid_n_at_issuance": 0,
        "evidence_bindings": evidence,
        "acceptance_candidate": evidence["acceptance_candidate"],
        "method_x_source_digest": runner.EXPECTED_METHOD_X_SOURCE_DIGEST,
        "battery_manifest_sha256": runner.EXPECTED_BATTERY_MANIFEST_SHA256,
        "full_battery_tree_preservation": preservation,
        "plans_005_006_007_allowed": False,
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "external_status": "ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION",
        "test_summary": dict(full_tests["summary"]),
        "traceability_integrity_sha256": traceability["integrity_sha256"],
    }
    body["integrity_sha256"] = runner.canonical_sha(body)
    target = runner.READY_REPORT_PATH
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        existing_semantic = dict(existing)
        candidate_semantic = dict(body)
        for row in (existing_semantic, candidate_semantic):
            row.pop("integrity_sha256", None)
            row.pop("created_at_utc", None)
        if existing_semantic != candidate_semantic:
            raise RuntimeError("BRAINIAC_FINAL_READINESS_IMMUTABLE_MISMATCH")
        runner.validate_final_preprocessing_readiness(target)
        return existing
    runner._atomic_write_json(target, body)
    runner.validate_final_preprocessing_readiness(target)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("traceability", "readiness", "all"), required=True
    )
    args = parser.parse_args()
    try:
        if args.mode == "all":
            materialize_novelty()
            _materialize_p08()
            build_traceability()
            report = issue_readiness()
        elif args.mode == "traceability":
            materialize_novelty()
            _materialize_p08()
            report = build_traceability()
        else:
            report = issue_readiness()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": str(exc)}))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
