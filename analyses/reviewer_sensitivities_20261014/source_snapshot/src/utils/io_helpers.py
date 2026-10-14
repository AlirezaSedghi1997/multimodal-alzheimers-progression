from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    p = ensure_parent(path)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    return atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

