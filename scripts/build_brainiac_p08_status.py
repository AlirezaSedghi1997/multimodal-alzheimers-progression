"""Build P08 status from the live byte-bound BrainIAC B1 authorities."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.brainiac_execution_gate import (  # noqa: E402
    DEFAULT_READINESS_LEDGER,
    assess_brainiac_execution_gate,
)
from src.research.brainiac_readiness import (  # noqa: E402
    ACTIONS,
    DEFAULT_R0_REGISTRY,
    DEFAULT_SCHEMA3_READINESS,
    audit_r0_registry,
    build_live_readiness,
    validate_materialized_readiness,
)
from src.research.brainiac_readiness_ledger import (  # noqa: E402
    readiness_transition_projection,
    validate_readiness_ledger,
)
from src.research.brainiac_preprocessing_v2_schema import canonical_sha  # noqa: E402
from src.utils.io_helpers import atomic_write_json  # noqa: E402


R0_ROOT = PROJECT_ROOT / "research" / "brainiac_caft_20260808"
NOVELTY = R0_ROOT / "novelty_matrix_v3.json"
LEGACY_REGISTRY = R0_ROOT / "evidence_registry_v2.json"
INVALIDATION = (
    PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "BRAINIAC_PREEXECUTION_INVALIDATION_20260821.json"
)
LEGACY_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "BRAINIAC_P08_STATUS_20260822.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "BRAINIAC_P08_STATUS_20260828_V3.json"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"BRAINIAC_P08_EVIDENCE_INVALID:{path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"BRAINIAC_P08_EVIDENCE_INVALID:{path}")
    return payload, raw


def _binding(path: Path, raw: bytes | None = None) -> dict[str, Any]:
    material = path.read_bytes() if raw is None else raw
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": len(material),
        "sha256": _sha256(material),
    }


def _validate_invalidation(payload: dict[str, Any]) -> None:
    if (
        payload.get("status") != "INVALID_NON_SCIENTIFIC_ARTIFACTS_RETAINED"
        or payload.get("training_performed") is not False
        or payload.get("outer_test_touched") is not False
    ):
        raise RuntimeError("BRAINIAC_P08_INVALIDATION_RECEIPT_INVALID")
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or len(rows) != 3:
        raise RuntimeError("BRAINIAC_P08_INVALIDATION_RECEIPT_INVALID")
    for row in rows:
        if not isinstance(row, dict) or row.get("scientific_use_allowed") is not False:
            raise RuntimeError("BRAINIAC_P08_INVALIDATION_RECEIPT_INVALID")
        path = PROJECT_ROOT / str(row.get("path", ""))
        raw = path.read_bytes()
        if len(raw) != int(row.get("bytes", -1)) or _sha256(raw) != row.get("sha256"):
            raise RuntimeError("BRAINIAC_P08_INVALIDATED_ARTIFACT_DRIFT")


def build_status() -> dict[str, Any]:
    r0 = audit_r0_registry(DEFAULT_R0_REGISTRY, project_root=PROJECT_ROOT)
    registry_raw = DEFAULT_R0_REGISTRY.read_bytes()
    novelty, novelty_raw = _load(NOVELTY)
    novelty_body = dict(novelty)
    novelty_integrity = novelty_body.pop("integrity_sha256", None)
    if (
        novelty.get("schema_version") != 3
        or novelty.get("status") != "R0_COMPLETE_NOVELTY_CANDIDATE"
        or novelty_integrity != canonical_sha(novelty_body)
        or novelty.get("claim_ceiling") != "NOVELTY_CANDIDATE"
        or novelty.get("prior_art_exact_match_found") is not False
        or novelty.get("registry_path")
        != DEFAULT_R0_REGISTRY.relative_to(PROJECT_ROOT).as_posix()
        or novelty.get("registry_sha256") != _sha256(registry_raw)
        or novelty.get("r0_registry_status") != "R0_COMPLETE"
        or novelty.get("citation_corroboration")
        != "CITATION_CORROBORATION_PARTIAL"
        or novelty.get("registry_binding") != _binding(DEFAULT_R0_REGISTRY, registry_raw)
    ):
        raise RuntimeError("BRAINIAC_P08_NOVELTY_LINKAGE_INVALID")

    readiness = build_live_readiness(project_root=PROJECT_ROOT)
    ledger = validate_readiness_ledger(
        DEFAULT_READINESS_LEDGER, project_root=PROJECT_ROOT
    )
    if ledger.get("head_payload") != readiness_transition_projection(readiness):
        raise RuntimeError("BRAINIAC_P08_READINESS_LEDGER_DRIFT")
    readiness_path = (
        R0_ROOT / "r1_readiness_snapshots" / f"{ledger['head_sha256']}.json"
    )
    readiness_raw = readiness_path.read_bytes()
    validate_materialized_readiness(readiness_path, project_root=PROJECT_ROOT)
    gates = {
        action: assess_brainiac_execution_gate(action) for action in ACTIONS
    }
    if gates["CAFT_TRAINING"]["allowed"] is not False:
        raise RuntimeError("BRAINIAC_P08_TRAINING_GATE_FAIL_OPEN")

    invalidation, invalidation_raw = _load(INVALIDATION)
    _validate_invalidation(invalidation)
    report = {
        "schema_version": 3,
        "as_of": "2026-08-28",
        "status": "R0_COMPLETE_PREPARATION_PARTIAL_NONBLOCKING_FOR_MAIN_CNN",
        "battery_facing_preexecution_gate": "PASS",
        "main_cnn_battery_blocked_by_brainiac": False,
        "brainiac_execution_allowed": any(row["allowed"] for row in gates.values()),
        "brainiac_training_allowed": gates["CAFT_TRAINING"]["allowed"],
        "r0": {
            **r0,
            "blocks_main_cnn_battery": False,
            "blocks_label_free_preparation": False,
        },
        "r1": {
            "status": readiness["status"],
            "preparation_ready": readiness["preparation_ready"],
            "training_allowed": readiness["training_allowed"],
            "authorized_actions": readiness["authorized_actions"],
            "actions": gates,
            "pretraining_overlap_status": readiness.get(
                "pretraining_overlap_status"
            ),
            "lora_unlocked": False,
        },
        "evidence": {
            "current_r0_registry": _binding(DEFAULT_R0_REGISTRY, registry_raw),
            "novelty_matrix": _binding(NOVELTY, novelty_raw),
            "current_r1_readiness": _binding(readiness_path, readiness_raw),
            "readiness_ledger_head_sha256": ledger["head_sha256"],
            "schema3_readiness_predecessor": _binding(DEFAULT_SCHEMA3_READINESS),
            "invalidation_receipt": _binding(INVALIDATION, invalidation_raw),
            "superseded_p08_status": _binding(LEGACY_OUTPUT),
        },
        "supersedes": [
            {
                **_binding(LEGACY_OUTPUT),
                "status": "SUPERSEDED_BY_LIVE_BYTE_SCHEMA3",
            }
        ],
        "superseded_historical_inputs": [_binding(LEGACY_REGISTRY)],
        "invalidated_non_scientific_artifact_count": len(invalidation["artifacts"]),
        "training_performed": False,
        "outer_test_touched": False,
        "next_brainiac_action": "IMPLEMENT_AND_RUN_B2_SCHEMA_V2_PREPARATION",
    }
    report["integrity_sha256"] = canonical_sha(report)
    return report


def main() -> int:
    report = build_status()
    atomic_write_json(OUTPUT, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
