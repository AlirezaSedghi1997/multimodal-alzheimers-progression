from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"status": "MISSING", "_path": str(p)}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "JSON_READ_FAILED", "_path": str(p), "error": str(exc)}


def nested(obj: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def pick_nested(obj: dict[str, Any] | None, *paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        value = nested(obj, *path, default=None)
        if value is not None:
            return value
    return default


def validate_required_keys(
    obj: dict[str, Any] | None,
    required_paths: list[tuple[str, ...]],
    *,
    context: str,
) -> list[str]:
    warnings: list[str] = []
    if not isinstance(obj, dict) or obj.get("status") in {"MISSING", "JSON_READ_FAILED"}:
        warnings.append(f"{context}: source missing or unreadable")
        return warnings
    for path in required_paths:
        if nested(obj, *path, default=None) is None:
            warnings.append(f"{context}: missing key {'/'.join(path)}")
    return warnings


def fmt_ci(point: Any, ci: dict[str, Any] | None, ndigits: int = 4) -> str:
    if point is None:
        return "NA"
    if not ci or ci.get("status") != "OK":
        return f"{float(point):.{ndigits}f}"
    lo = ci.get("ci_low", ci.get("ci_lower"))
    hi = ci.get("ci_high", ci.get("ci_upper"))
    if lo is None or hi is None:
        return f"{float(point):.{ndigits}f}"
    return f"{float(point):.{ndigits}f} [{float(lo):.{ndigits}f}, {float(hi):.{ndigits}f}]"


def fmt(value: object, ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def best_centiloid_spearman(subset_block: dict[str, Any] | None) -> float | None:
    if not isinstance(subset_block, dict):
        return None
    models = subset_block.get("models") or {}
    best: float | None = None
    for model in models.values():
        if not isinstance(model, dict):
            continue
        r = model.get("spearman_r")
        if r is None:
            continue
        r_f = float(r)
        if best is None or r_f > best:
            best = r_f
    return best


def ci_overlap(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool | None:
    if not a or not b or a.get("status") != "OK" or b.get("status") != "OK":
        return None
    a_lo, a_hi = a.get("ci_low", a.get("ci_lower")), a.get("ci_high", a.get("ci_upper"))
    b_lo, b_hi = b.get("ci_low", b.get("ci_lower")), b.get("ci_high", b.get("ci_upper"))
    if None in (a_lo, a_hi, b_lo, b_hi):
        return None
    return not (float(a_hi) < float(b_lo) or float(b_hi) < float(a_lo))
