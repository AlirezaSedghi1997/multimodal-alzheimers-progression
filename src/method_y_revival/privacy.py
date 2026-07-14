from __future__ import annotations

from typing import Any, Mapping

import re
from pathlib import Path


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


def scan_public_bundle(root: str | Path) -> list[dict[str, str]]:
    rules = {
        "RAW_DRIVE_PATH": re.compile(r"(?i)(?:[A-Z]:[\\/]+|A4[ _]Learn)"),
        "PRIVATE_CHECKPOINT_REFERENCE": re.compile(r"(?i)(?:\.ckpt|\.pt|\.h5|checkpoint)"),
        "RAW_IDENTIFIER_FIELD": re.compile(r"(?i)(?:source_subject_id|raw_path|sidecar_path)"),
    }
    findings: list[dict[str, str]] = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for rule, pattern in rules.items():
            if pattern.search(text):
                findings.append({"rule": rule, "path": path.relative_to(root).as_posix()})
    return findings
