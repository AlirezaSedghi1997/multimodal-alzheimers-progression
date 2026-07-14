from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TracerDecision:
    tracer: str | None
    status: str
    reason: str
    evidence_source: str | None = None


def _normalize_tracer(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().casefold().replace("-", "").replace("_", "")
    if normalized in {"fbp", "florbetapir", "av45", "18fflorbetapir"}:
        return "FBP"
    if normalized in {"ftp", "flortaucipir", "av1451", "18fflortaucipir"}:
        return "FTP"
    return None


def resolve_tracer(
    *,
    metadata_tracer: str | None,
    sidecar_tracer: str | None,
    table_tracer: str | None,
    filename: str,
) -> TracerDecision:
    exact_evidence = [
        ("metadata", _normalize_tracer(metadata_tracer)),
        ("sidecar", _normalize_tracer(sidecar_tracer)),
        ("source_table", _normalize_tracer(table_tracer)),
    ]
    exact_values = {value for _, value in exact_evidence if value is not None}
    if len(exact_values) > 1:
        return TracerDecision(None, "QUARANTINE", "CONFLICTING_EXACT_TRACER_EVIDENCE")
    if len(exact_values) == 1:
        tracer = next(iter(exact_values))
        source = next(source for source, value in exact_evidence if value == tracer)
        return TracerDecision(tracer, "RESOLVED", "EXACT_EVIDENCE", source)

    filename_value = _normalize_tracer(filename.split(".")[0].split("_")[-1])
    if filename_value is not None:
        return TracerDecision(filename_value, "RESOLVED_LOW_CONFIDENCE", "FILENAME_ONLY", "filename")
    return TracerDecision(None, "QUARANTINE", "UNKNOWN_OR_UNSUPPORTED_TRACER")
