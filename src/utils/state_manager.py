from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io_helpers import atomic_write_text


PROGRESS_PATH = Path(".agent/PROGRESS.yaml")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StateManager:
    """Small YAML-like state writer that avoids requiring PyYAML for bootstrap."""

    def __init__(self, path: str | Path = PROGRESS_PATH) -> None:
        self.path = Path(path)

    def append_event(self, stage: str, status: str, message: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        event = (
            "\n"
            f"# event {utc_now()}\n"
            f"last_stage: {stage}\n"
            f"last_status: {status}\n"
            f"last_message: {message}\n"
        )
        atomic_write_text(self.path, existing.rstrip() + event)

