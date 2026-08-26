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
        "PSEUDONYMOUS_SUBJECT_IDENTIFIER": re.compile(
            r"(?i)\b(?:pt|sub|scan)_[a-z0-9][a-z0-9_-]*\b"
        ),
    }
    private_file_suffixes = {
        ".ckpt",
        ".h5",
        ".hdf5",
        ".joblib",
        ".nii",
        ".onnx",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
    }
    findings: list[dict[str, str]] = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        # The scanner's own prior receipt can legitimately contain the names of
        # rules that failed.  Re-scanning that receipt makes a corrected bundle
        # permanently fail because it reports its historical finding back to
        # itself.  The receipt is regenerated after every scan, so exclude only
        # this fixed bookkeeping filename; all release payloads remain covered.
        if path.name == "privacy_scan.json":
            continue
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if suffixes.intersection(private_file_suffixes):
            findings.append(
                {
                    "rule": "PRIVATE_MODEL_OR_DATA_FILE",
                    "path": path.relative_to(root).as_posix(),
                }
            )
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for rule, pattern in rules.items():
            if pattern.search(text):
                findings.append({"rule": rule, "path": path.relative_to(root).as_posix()})
    return findings
