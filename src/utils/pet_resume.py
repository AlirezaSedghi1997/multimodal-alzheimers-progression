from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .pet_spatial_qc import atomic_write_json, sha256_text


class PetResumeManifest:
    """Tiny resumable manifest. Saves after each item to survive crashes."""
    def __init__(self, path: str | Path, stage: str, config: Dict[str, Any]):
        self.path = Path(path)
        self.stage = stage
        self.config = config
        self.config_digest = sha256_text(json.dumps(config, sort_keys=True))
        self.data = {
            "stage": stage,
            "config_digest": self.config_digest,
            "config": config,
            "items": {},
        }

    def load(self) -> None:
        if not self.path.exists():
            return
        obj = json.loads(self.path.read_text(encoding="utf-8"))
        if obj.get("config_digest") != self.config_digest:
            raise RuntimeError(
                f"Resume manifest config digest mismatch for {self.path}; use --overwrite_resume or change output."
            )
        self.data = obj

    def is_done(self, key: str) -> bool:
        return self.data.get("items", {}).get(key, {}).get("status") in {"PASS", "SKIPPED_EXISTS"}

    def mark(self, key: str, status: str, info: Dict[str, Any] | None = None) -> None:
        self.data.setdefault("items", {})[key] = {"status": status, "info": info or {}}

    def save(self) -> None:
        atomic_write_json(self.path, self.data)
