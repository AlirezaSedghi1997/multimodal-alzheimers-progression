from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io_helpers import atomic_write_json, read_json


VALID_STATUSES = {"PENDING", "RUNNING", "DONE", "FAILED", "BLOCKED", "SKIPPED"}
RAW_ID_PATTERN = re.compile(r"\bB\d{3,}\b", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_digest(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def assert_safe_resume_key(key: str) -> None:
    if not key or any(part in key for part in ("/", "\\", ":", "..")):
        raise ValueError("Resume item keys must be opaque hashes, not paths.")
    if RAW_ID_PATTERN.search(key):
        raise ValueError("Resume item keys must not contain raw subject identifiers.")


class ResumableManifest:
    """Crash-safe JSON manifest for long per-item stages.

    Item keys must be hashes generated elsewhere in the pipeline. The manifest is
    intentionally small and text-only so interrupted stages can be inspected and
    resumed without scanning raw data again.
    """

    def __init__(self, path: str | Path, *, stage: str, config: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.stage = stage
        self.config = config or {}
        self.config_hash = config_digest(self.config)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": 1,
                "stage": self.stage,
                "config_hash": self.config_hash,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "items": {},
            }
        obj = read_json(self.path)
        if obj.get("stage") != self.stage:
            raise ValueError(f"Manifest stage mismatch: expected {self.stage}, got {obj.get('stage')}")
        if obj.get("config_hash") != self.config_hash:
            raise ValueError("Manifest config hash mismatch; use a new manifest or explicit reset.")
        obj.setdefault("items", {})
        return obj

    def write(self, obj: dict[str, Any]) -> None:
        obj["updated_at"] = utc_now()
        atomic_write_json(self.path, obj)

    def item_done(self, key: str) -> bool:
        assert_safe_resume_key(key)
        item = self.load()["items"].get(key)
        return bool(item and item.get("status") == "DONE")

    def set_item(
        self,
        key: str,
        status: str,
        *,
        artifact: str | None = None,
        message: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        assert_safe_resume_key(key)
        if status not in VALID_STATUSES:
            raise ValueError(f"Unknown resumable status: {status}")
        obj = self.load()
        entry: dict[str, Any] = {"status": status, "updated_at": utc_now()}
        if artifact:
            entry["artifact"] = artifact
        if message:
            entry["message"] = message[:300]
        if extra:
            entry["extra"] = extra
        obj["items"][key] = entry
        self.write(obj)

    def summary(self) -> dict[str, int]:
        counts = Counter(item.get("status", "PENDING") for item in self.load()["items"].values())
        return {status: int(counts.get(status, 0)) for status in sorted(VALID_STATUSES)}
