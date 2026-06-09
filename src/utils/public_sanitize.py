from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RAW_ID_REPLACEMENTS = {
    "PREVBID": "<RAW_ID_FIELD>",
    "BID": "<RAW_ID_FIELD>",
    "patient_id": "<RAW_ID_FIELD>",
    "sample_id": "<RAW_ID_FIELD>",
}

LOCAL_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\[^\s\"'`,\]\)]+"),
    re.compile(r"/mnt/[a-zA-Z]/[^\s\"'`,\]\)]+"),
    re.compile(r"[A-Za-z]:/[^\s\"'`,\]\)]+"),
)


def redact_raw_identifiers(text: str) -> str:
    out = text
    for old, new in RAW_ID_REPLACEMENTS.items():
        out = out.replace(old, new)
    return out


def redact_local_paths(text: str) -> str:
    out = text
    for pattern in LOCAL_PATH_PATTERNS:
        out = pattern.sub("<LOCAL_PATH_REDACTED>", out)
    return out


def sanitize_public_text(text: str) -> str:
    return redact_local_paths(redact_raw_identifiers(text))


def scan_for_local_paths(text: str) -> bool:
    return any(p.search(text) for p in LOCAL_PATH_PATTERNS)


def scan_json_for_subject_vectors(obj: Any) -> bool:
    text = json.dumps(obj, ensure_ascii=False)
    return "y_prob" in text or "y_true" in text


def refresh_mri_smoke_summary_if_superseded(
    smoke_path: Path = Path("results/tables/mri_volume_cache_smoke_summary.json"),
    full_path: Path = Path("results/tables/mri_volume_cache_summary.json"),
) -> dict[str, Any] | None:
    if not full_path.exists():
        return None
    full = json.loads(full_path.read_text(encoding="utf-8"))
    if int(full.get("ok") or 0) < 1600 or not full.get("published_cache"):
        return None
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else {}
    smoke.update(
        {
            "status": "SUPERSEDED_BY_FULL_CACHE",
            "purpose": "historical_dependency_smoke_only",
            "training_usable": False,
            "published_cache": False,
            "superseded_by": str(full_path).replace("\\", "/"),
            "full_cache_rows": full.get("rows"),
            "full_cache_ok": full.get("ok"),
            "note": "Full 96³ T1 volume cache exists; smoke artifact retained for audit trail only.",
        }
    )
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    md_path = smoke_path.with_suffix(".md")
    md_path.write_text(
        "# MRI Volume Cache Smoke Summary\n\n"
        f"- status: `{smoke['status']}`\n"
        f"- superseded_by: `{smoke['superseded_by']}`\n"
        f"- full_cache_rows: `{smoke.get('full_cache_rows')}`\n"
        f"- note: {smoke['note']}\n",
        encoding="utf-8",
    )
    return smoke
