from __future__ import annotations

from typing import Any, Mapping


PRIVATE_MANIFEST_FIELDS = frozenset(
    {"raw_path", "sidecar_path", "source_subject_id", "station_name", "scan_key"}
)


def public_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in PRIVATE_MANIFEST_FIELDS}


def find_private_fields(rows: list[Mapping[str, Any]]) -> list[str]:
    offenders: set[str] = set()
    for row in rows:
        offenders.update(PRIVATE_MANIFEST_FIELDS.intersection(row))
    return sorted(offenders)
