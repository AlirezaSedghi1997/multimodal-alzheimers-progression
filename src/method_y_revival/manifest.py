from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .constants import TRACER_TO_BRANCH


class ManifestError(ValueError):
    """Raised when a scan row cannot satisfy the locked manifest contract."""


@dataclass
class ScanManifestRow:
    subject_key: str
    session_key: str
    scan_key: str
    cohort: str
    branch: str
    tracer: str
    raw_path: str
    raw_sha256: str
    source_subject_id: str
    manufacturer: str
    metadata_linkage_method: str
    metadata_linkage_confidence: str
    training_eligible: bool
    sidecar_path: str | None = None
    sidecar_sha256: str | None = None
    visit_date: str | None = None
    acquisition_datetime: str | None = None
    injection_datetime: str | None = None
    frame_start_seconds: list[float] | None = None
    frame_duration_seconds: list[float] | None = None
    model_name: str = "UNKNOWN"
    station_name: str = "UNKNOWN"
    site: str = "UNKNOWN"
    software_version: str = "UNKNOWN"
    image_shape: list[int] = field(default_factory=list)
    voxel_spacing_mm: list[float] = field(default_factory=list)
    affine_sha256: str | None = None
    t1_scan_key: str | None = None
    exclusion_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected_branch = TRACER_TO_BRANCH.get(self.tracer)
        if expected_branch is None:
            raise ManifestError(f"UNSUPPORTED_TRACER:{self.tracer}")
        if self.branch != expected_branch:
            raise ManifestError(
                f"TRACER_BRANCH_MISMATCH:{self.tracer}:expected={expected_branch}:actual={self.branch}"
            )
        if self.metadata_linkage_confidence not in {"EXACT", "AMBIGUOUS", "MISSING"}:
            raise ManifestError(
                f"INVALID_METADATA_LINKAGE_CONFIDENCE:{self.metadata_linkage_confidence}"
            )
        if self.training_eligible and self.metadata_linkage_confidence != "EXACT":
            raise ManifestError("NON_EXACT_METADATA_CANNOT_TRAIN")
        if len(self.raw_sha256) != 64:
            raise ManifestError("RAW_SHA256_REQUIRED")
        if not self.manufacturer:
            self.manufacturer = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_digest(rows: Iterable[dict[str, Any]]) -> str:
    normalized = sorted(
        (json.loads(json.dumps(row, sort_keys=True, default=str)) for row in rows),
        key=lambda row: (
            str(row.get("subject_key", "")),
            str(row.get("session_key", "")),
            str(row.get("scan_key", "")),
        ),
    )
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_manifest(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    materialized = list(rows)
    subjects = {str(row["subject_key"]) for row in materialized}
    fbp_subjects = {
        str(row["subject_key"]) for row in materialized if row.get("tracer") == "FBP"
    }
    ftp_subjects = {
        str(row["subject_key"]) for row in materialized if row.get("tracer") == "FTP"
    }
    return {
        "row_count": len(materialized),
        "unique_subject_count": len(subjects),
        "fbp_subject_count": len(fbp_subjects),
        "ftp_subject_count": len(ftp_subjects),
        "unknown_scanner_count": sum(
            str(row.get("manufacturer", "UNKNOWN")).upper() == "UNKNOWN"
            for row in materialized
        ),
    }
