from __future__ import annotations

import hashlib
import re
from pathlib import Path


SECRET_SALT = "method-x-a4learn-local"
ID_PATTERNS = [
    re.compile(r"\bB\d{5,}\b", re.IGNORECASE),
    re.compile(r"\bPREVBID\b", re.IGNORECASE),
    re.compile(r"\bLRN_?\d{4,}\b", re.IGNORECASE),
]


def stable_hash(value: object, *, prefix: str = "h") -> str:
    text = "" if value is None else str(value)
    digest = hashlib.sha256((SECRET_SALT + "|" + text).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def redact_path(path: str | Path) -> str:
    p = str(path)
    p = p.replace("\\", "/")
    if "A4 Learn" in p:
        suffix = p.split("A4 Learn", 1)[1].lstrip("/")
        return "<DATA_ROOT>/" + suffix
    return "<PATH:" + stable_hash(p, prefix="path") + ">"


def contains_raw_identifier(text: str) -> bool:
    return any(p.search(text or "") for p in ID_PATTERNS)


def scrub_identifier_text(text: str) -> str:
    out = text
    for pattern in ID_PATTERNS:
        out = pattern.sub("<RAW_ID_FIELD>", out)
    return out

