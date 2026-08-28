"""Build deterministic, append-only BrainIAC R0 and R1 schema-4 status artifacts.

This command performs governance/evidence checks only.  It never opens study
images, labels, model checkpoints, embeddings, or outer-test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.research.brainiac_readiness import (
    DEFAULT_R0_REGISTRY,
    PROJECT_ROOT,
    audit_r0_registry,
    bind_file,
    build_live_readiness,
    validate_materialized_readiness,
    write_readiness_artifact,
)
from src.research.brainiac_readiness_anchor import validate_production_anchor
from src.research.brainiac_readiness_ledger import (
    PINNED_B1_GENESIS_EVENT_SHA256,
    PRODUCTION_READINESS_LEDGER,
    append_anchored_readiness_transition,
    readiness_transition_projection,
    validate_readiness_ledger,
)


REGISTRY_ROOT = PROJECT_ROOT / "research" / "brainiac_caft_20260808"
R0_STATUS_PATH = REGISTRY_ROOT / "r0_derived_status_20260822.json"
R0_STATUS_SNAPSHOTS = REGISTRY_ROOT / "r0_derived_status_snapshots"
R1_SCHEMA4_PATH = REGISTRY_ROOT / "r1_readiness_status_schema4_20260822.json"
READINESS_LEDGER = PRODUCTION_READINESS_LEDGER
READINESS_SNAPSHOTS = REGISTRY_ROOT / "r1_readiness_snapshots"


def build(*, write: bool) -> dict[str, object]:
    r0 = audit_r0_registry(DEFAULT_R0_REGISTRY, project_root=PROJECT_ROOT)
    readiness = build_live_readiness(project_root=PROJECT_ROOT)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "VALIDATED_NOT_MATERIALIZED" if not write else "MATERIALIZED",
        "r0": r0,
        "readiness": {
            "status": readiness["status"],
            "preparation_ready": readiness["preparation_ready"],
            "training_allowed": readiness["training_allowed"],
            "authorized_actions": readiness["authorized_actions"],
            "sole_runtime_blocker": readiness["sole_runtime_blocker"],
            "actions": readiness["actions"],
        },
        "training_started": False,
        "outer_test_touched": False,
        "main_method_x_battery_affected": False,
    }
    if not write:
        return report

    r0_payload = {
        "schema_version": 1,
        "artifact_type": "BRAINIAC_R0_DERIVED_LOCK",
        **r0,
        "training_started": False,
        "outer_test_touched": False,
        "supersedes": [
            {
                **bind_file(R0_STATUS_PATH, project_root=PROJECT_ROOT),
                "status": "SUPERSEDED_BY_APPEND_ONLY_R0_SNAPSHOT",
            }
        ] if R0_STATUS_PATH.exists() else [],
    }
    r0_target = R0_STATUS_SNAPSHOTS / f"{r0['registry_binding']['sha256']}.json"
    r0_receipt = write_readiness_artifact(
        r0_target, r0_payload, append_only=True
    )
    event = append_anchored_readiness_transition(
        READINESS_LEDGER,
        readiness_transition_projection(readiness),
        project_root=PROJECT_ROOT,
        expected_genesis_sha256=PINNED_B1_GENESIS_EVENT_SHA256,
    )
    snapshot_path = READINESS_SNAPSHOTS / f"{event['event_sha256']}.json"
    readiness_receipt = write_readiness_artifact(
        snapshot_path, readiness, append_only=True
    )
    validate_materialized_readiness(
        snapshot_path,
        project_root=PROJECT_ROOT,
        expected_sha256=str(readiness_receipt["sha256"]),
    )
    ledger_report = validate_readiness_ledger(
        READINESS_LEDGER,
        project_root=PROJECT_ROOT,
        expected_genesis_sha256=PINNED_B1_GENESIS_EVENT_SHA256,
    )
    anchor_report = validate_production_anchor(
        ledger_dir=READINESS_LEDGER, ledger_report=ledger_report
    )
    report["artifacts"] = {
        "r0": r0_receipt,
        "schema4_readiness": readiness_receipt,
        "readiness_transition": event,
        "readiness_ledger": ledger_report,
        "readiness_os_anchor": anchor_report,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("status", "materialize"), default="status")
    args = parser.parse_args(argv)
    print(json.dumps(build(write=args.mode == "materialize"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
