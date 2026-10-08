from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from src.utils.io_helpers import atomic_write_json
from src.utils.privacy import redact_path


def write_run_manifest(path: str | Path, *, stage: str, data_root: str | None, extra: dict[str, Any] | None = None) -> Path:
    obj: dict[str, Any] = {
        "project": "amyloid-tau-prediction-a4learn",
        "method": "X",
        "stage": stage,
        "data_root_redacted": redact_path(data_root or ""),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "privacy": {
            "raw_ids_logged": False,
            "raw_data_copied": False,
            "pet_as_input": False,
        },
    }
    if extra:
        obj.update(extra)
    return atomic_write_json(path, obj)

