from __future__ import annotations

from typing import Any, Mapping

import re
import csv
import io
import json
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
    forbidden_keys = {"event", "time_days", "original_event", "original_time_days",
                      "patient_hash", "subject_key", "censoring_reference",
                      "oof_predictions", "predictions", "embeddings", "outcomes",
                      "risk_score", "source_subject_id", "raw_path", "sidecar_path"}
    def has_rows(value):
        if isinstance(value, dict):
            return any(str(k).lower() in forbidden_keys or has_rows(v) for k, v in value.items())
        return isinstance(value, list) and any(has_rows(v) for v in value)
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
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                safe = (path.parent == Path(root) and isinstance(receipt, dict)
                        and set(receipt) <= {"status", "findings", "facts_sha256", "bundle_manifest_sha256"}
                        and receipt.get("status") in {"PASS", "FAIL"}
                        and isinstance(receipt.get("findings"), list)
                        and all(isinstance(f, dict) and set(f) <= {"rule", "path"}
                                and all(isinstance(v, str) for v in f.values())
                                for f in receipt["findings"]))
            except (OSError, ValueError):
                safe = False
            if safe:
                continue
            findings.append({"rule": "INVALID_PRIVACY_RECEIPT", "path": path.relative_to(root).as_posix()})
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
            findings.append({"rule": "UNINSPECTABLE_PUBLIC_PAYLOAD", "path": path.relative_to(root).as_posix()})
            continue
        try:
            if path.suffix.lower() == ".json":
                if has_rows(json.loads(text)):
                    findings.append({"rule": "ROW_LEVEL_DATA", "path": path.relative_to(root).as_posix()})
            elif path.suffix.lower() == ".csv":
                header = next(csv.reader(io.StringIO(text)), [])
                if any(k.lower() in forbidden_keys for k in header):
                    findings.append({"rule": "ROW_LEVEL_DATA", "path": path.relative_to(root).as_posix()})
        except (ValueError, csv.Error):
            findings.append({"rule": "UNPARSEABLE_PUBLIC_PAYLOAD", "path": path.relative_to(root).as_posix()})
        for rule, pattern in rules.items():
            if pattern.search(text):
                findings.append({"rule": rule, "path": path.relative_to(root).as_posix()})
    return findings
