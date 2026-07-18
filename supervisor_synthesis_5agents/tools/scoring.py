"""Deterministic hard-gate and 100-point candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


WEIGHTS = {
    "novelty": 18,
    "clinical": 16,
    "data": 15,
    "reuse": 15,
    "time": 12,
    "compute": 8,
    "evaluation": 8,
    "story": 5,
    "interpretability": 3,
}
REQUIRED_GATES = ("G0", "G1", "G2", "G3", "G4", "G5", "G7", "G8")


@dataclass
class Candidate:
    idea_id: str
    normalized_family: str
    gates: dict[str, str]
    scores: dict[str, float]
    penalties: list[tuple[str, float]] = field(default_factory=list)
    scientific_from_scratch: float | None = None
    practical_continuation: float | None = None
    confidence: str = "B"
    metadata: dict[str, Any] = field(default_factory=dict)


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, round(value, 2)))


def score_candidate(candidate: Candidate) -> dict[str, Any]:
    missing = [key for key in WEIGHTS if key not in candidate.scores]
    if missing:
        raise ValueError(f"Missing score criteria: {missing}")
    weighted = sum(min(WEIGHTS[key], max(0.0, float(candidate.scores[key]))) for key in WEIGHTS)
    penalties = sum(float(value) for _, value in candidate.penalties)
    practical = _clamp(weighted - penalties)
    # The from-scratch score deliberately discounts reuse and rewards novelty.
    from_scratch = _clamp(
        weighted
        - float(candidate.scores.get("reuse", 0)) * 0.30
        + float(candidate.scores.get("novelty", 0)) * 0.20
    )
    hard_fail = [gate for gate in REQUIRED_GATES if candidate.gates.get(gate) == "FAIL"]
    if hard_fail:
        verdict = "REJECTED_HARD_GATE"
    elif candidate.gates.get("G6") == "FAIL":
        verdict = "REJECTED_NOVELTY_GATE"
    else:
        verdict = "ELIGIBLE"
    return {
        "idea_id": candidate.idea_id,
        "normalized_family": candidate.normalized_family,
        "gates": dict(candidate.gates),
        "weighted_score": round(weighted, 2),
        "penalties": [{"reason": reason, "points": points} for reason, points in candidate.penalties],
        "scientific_from_scratch_score": from_scratch,
        "practical_continuation_score": practical,
        "cannot_be_primary": bool(hard_fail or candidate.gates.get("G6") == "FAIL"),
        "hard_fail_gates": hard_fail,
        "verdict": verdict,
        "confidence": candidate.confidence,
        "metadata": candidate.metadata,
    }


def choose_primary(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in results if not row["cannot_be_primary"]]
    if not eligible:
        return max(results, key=lambda row: row["practical_continuation_score"])
    return sorted(
        eligible,
        key=lambda row: (
            row["practical_continuation_score"],
            row["scientific_from_scratch_score"],
            -len(row["penalties"]),
            row["idea_id"],
        ),
        reverse=True,
    )[0]
