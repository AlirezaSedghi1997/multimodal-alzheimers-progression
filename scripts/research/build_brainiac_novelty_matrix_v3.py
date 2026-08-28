"""Rebuild the BrainIAC novelty matrix binding without widening its claim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.brainiac_readiness import audit_r0_registry  # noqa: E402
from src.research.brainiac_preprocessing_v2_schema import canonical_sha  # noqa: E402
from src.utils.io_helpers import atomic_write_json  # noqa: E402


R0_ROOT = ROOT / "research" / "brainiac_caft_20260808"
REGISTRY = R0_ROOT / "evidence_registry_v3.json"
PREDECESSOR = R0_ROOT / "novelty_matrix.json"
OUTPUT = R0_ROOT / "novelty_matrix_v3.json"


def _binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_predecessor_semantics(
    matrix: Mapping[str, Any], registry: Mapping[str, Any]
) -> None:
    rows = matrix.get("rows")
    sources = registry.get("sources")
    if not isinstance(rows, list) or not isinstance(sources, list):
        raise RuntimeError("BRAINIAC_NOVELTY_SEMANTIC_INPUT_INVALID")
    source_by_id = {
        str(source.get("id")): source
        for source in sources
        if isinstance(source, Mapping) and source.get("id")
    }
    row_by_id = {
        str(row.get("source_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("source_id")
    }
    if len(source_by_id) != 29 or set(row_by_id) != set(source_by_id):
        raise RuntimeError("BRAINIAC_NOVELTY_SOURCE_COVERAGE_DRIFT")
    dimensions = (
        "has_3d_foundation_model",
        "has_local_128_cnn_fusion",
        "has_pet_free_amyloid_target",
        "has_scanner_confound_audit",
        "has_calibrated_conformal_abstention",
    )
    for source_id, row in row_by_id.items():
        source = source_by_id[source_id]
        if row.get("year") != source.get("year") or row.get("tier") != source.get("tier"):
            raise RuntimeError("BRAINIAC_NOVELTY_SOURCE_METADATA_DRIFT")
        overlap = sum(bool(row.get(field)) for field in dimensions)
        if int(row.get("overlap_dimensions_count", -1)) != overlap:
            raise RuntimeError("BRAINIAC_NOVELTY_DIMENSION_COUNT_INVALID")
        if row.get("simultaneous_5_dimension_overlap") is not (overlap == 5):
            raise RuntimeError("BRAINIAC_NOVELTY_EXACT_MATCH_FLAG_INVALID")
    exact = any(bool(row["simultaneous_5_dimension_overlap"]) for row in rows)
    if (
        exact
        or matrix.get("prior_art_exact_match_found") is not False
        or matrix.get("claim_ceiling") != "NOVELTY_CANDIDATE"
        or matrix.get("claim_status") != "NOVELTY_CANDIDATE"
        or str(matrix.get("closest_prior_art", {}).get("source_id")) not in source_by_id
    ):
        raise RuntimeError("BRAINIAC_NOVELTY_CLAIM_SEMANTIC_DRIFT")


def build_matrix() -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    _validate_predecessor_semantics(predecessor, registry)
    audit = audit_r0_registry(REGISTRY, project_root=ROOT)
    if (
        audit.get("r0_status") != "R0_COMPLETE"
        or audit.get("novelty_status") != "NOVELTY_CANDIDATE"
        or audit.get("citation_corroboration") != "CITATION_CORROBORATION_PARTIAL"
        or int(audit.get("total_in_window", -1)) != 29
        or int(audit.get("peer_reviewed", -1)) != 13
        or int(audit.get("full_text_reviewed", -1)) != 28
    ):
        raise RuntimeError("BRAINIAC_NOVELTY_R0_AUDIT_DRIFT")

    body = {
        key: value
        for key, value in predecessor.items()
        if key not in {"schema_version", "registry_sha256", "r0_registry_status"}
    }
    body.update(
        {
            "schema_version": 3,
            "artifact": "BRAINIAC_CAFT_NOVELTY_MATRIX",
            "status": "R0_COMPLETE_NOVELTY_CANDIDATE",
            "r0_registry_status": "R0_COMPLETE",
            "citation_corroboration": "CITATION_CORROBORATION_PARTIAL",
            "registry_path": REGISTRY.relative_to(ROOT).as_posix(),
            "registry_sha256": _binding(REGISTRY)["sha256"],
            "registry_binding": _binding(REGISTRY),
            "predecessor_binding": _binding(PREDECESSOR),
            "supersedes": [
                {
                    **_binding(PREDECESSOR),
                    "status": "SUPERSEDED_BY_BYTE_BOUND_SCHEMA3",
                }
            ],
            "total_sources_evaluated": int(audit["total_in_window"]),
            "peer_reviewed_sources_count": int(audit["peer_reviewed"]),
            "full_text_reviewed_count": int(audit["full_text_reviewed"]),
            "prior_art_exact_match_found": False,
            "claim_ceiling": "NOVELTY_CANDIDATE",
            "claim_status": "NOVELTY_CANDIDATE",
        }
    )
    body["integrity_sha256"] = canonical_sha(body)
    return body


def materialize() -> dict[str, Any]:
    candidate = build_matrix()
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if existing != candidate:
            raise RuntimeError("BRAINIAC_NOVELTY_MATRIX_IMMUTABLE_MISMATCH")
        return existing
    atomic_write_json(OUTPUT, candidate)
    return candidate


def main() -> int:
    try:
        report = materialize()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": str(exc)}))
        return 3
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
