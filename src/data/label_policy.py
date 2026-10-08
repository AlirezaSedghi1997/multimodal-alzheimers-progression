from __future__ import annotations

import json
from pathlib import Path

CENTILOID_FROM_SUVRCER_SLOPE = 183.07
CENTILOID_FROM_SUVRCER_INTERCEPT = -177.26
DEFAULT_LABEL_FREEZE = Path("results/governance/label_freeze.json")
PLACEHOLDER_CUTOFF = "[INSERT PRESPECIFIED CUTOFF BEFORE MODELING]"


def load_frozen_centiloid_cutoff(path: Path | None = None) -> float:
    freeze_path = path or DEFAULT_LABEL_FREEZE
    if freeze_path.exists():
        obj = json.loads(freeze_path.read_text(encoding="utf-8"))
        cutoff = obj.get("cutoff_value", obj.get("centiloid_threshold"))
        if cutoff not in {None, "", PLACEHOLDER_CUTOFF}:
            return float(cutoff)
    return 33.2705


def centiloid_from_pmod_suvr(pmod_suvr: float) -> float:
    return CENTILOID_FROM_SUVRCER_SLOPE * float(pmod_suvr) + CENTILOID_FROM_SUVRCER_INTERCEPT


def binary_label_from_centiloid(centiloid: float, cutoff: float) -> int:
    return 1 if float(centiloid) >= float(cutoff) else 0


def binary_label_from_va(overall_score: str | None) -> int | None:
    if overall_score is None:
        return None
    value = overall_score.strip().lower()
    if value in {"1", "pos", "positive", "ab_pos", "a+", "elevated"}:
        return 1
    if value in {"0", "neg", "negative", "ab_neg", "a-", "not elevated", "nonelevated"}:
        return 0
    if "pos" in value or "elev" in value:
        return 1
    if "neg" in value or "non" in value:
        return 0
    return None
