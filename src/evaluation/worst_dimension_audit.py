"""Worst-dimension fusion audit via augmented Chebyshev scalarization (arXiv:2606.07801).

Adapted for amyloid fusion evaluation: non-compensatory scoring over
{AUC, calibration, cohort-probe, missing-modality robustness}.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


MetricFn = Callable[[Sequence[int], Sequence[float]], float]


@dataclass
class WorstDimensionConfig:
    rho: float = 0.05
    dimensions: tuple[str, ...] = ("roc_auc", "calibration", "cohort_probe", "missing_modality")
    higher_is_better: Mapping[str, bool] = field(
        default_factory=lambda: {
            "roc_auc": True,
            "calibration": True,
            "cohort_probe": True,
            "missing_modality": True,
        }
    )


def normalize_score(raw: float, *, higher_is_better: bool) -> float:
    x = float(raw)
    if not higher_is_better:
        x = 1.0 - x
    return max(0.0, min(1.0, x))


def augmented_chebyshev(
    scores: Mapping[str, float],
    ideal: Mapping[str, float],
    *,
    rho: float = 0.05,
    higher_is_better: Mapping[str, bool] | None = None,
) -> float:
    """Augmented Chebyshev: focus on worst dimension (MMS-PRM Eq. 8)."""
    hib = higher_is_better or {}
    gaps = []
    for key, ideal_val in ideal.items():
        raw = scores.get(key, 0.0)
        norm = normalize_score(raw, higher_is_better=hib.get(key, True))
        ideal_norm = normalize_score(ideal_val, higher_is_better=hib.get(key, True))
        gaps.append(ideal_norm - norm)
    cheby = -max(gaps) if gaps else 0.0
    aug = cheby - rho * sum(gaps)
    return float(aug)


def worst_dimension_report(
    variant_scores: Mapping[str, Mapping[str, float]],
    *,
    ideal: Mapping[str, float] | None = None,
    config: WorstDimensionConfig | None = None,
) -> dict[str, object]:
    cfg = config or WorstDimensionConfig()
    ideal = ideal or {dim: 1.0 for dim in cfg.dimensions}
    ranked = []
    for name, scores in variant_scores.items():
        scalar = augmented_chebyshev(scores, ideal, rho=cfg.rho, higher_is_better=cfg.higher_is_better)
        gaps = {
            dim: normalize_score(scores.get(dim, 0.0), higher_is_better=cfg.higher_is_better.get(dim, True))
            - normalize_score(ideal.get(dim, 1.0), higher_is_better=cfg.higher_is_better.get(dim, True))
            for dim in cfg.dimensions
        }
        worst_dim = min(gaps, key=gaps.get) if gaps else None
        ranked.append({"variant": name, "chebyshev_aug": scalar, "worst_dimension": worst_dim, "gaps": gaps, "raw": dict(scores)})
    ranked.sort(key=lambda row: row["chebyshev_aug"], reverse=True)
    return {
        "method": "MMS-PRM_adapted_worst_dimension_audit",
        "literature_anchor": "LIT016 arXiv:2606.07801",
        "ideal_point": dict(ideal),
        "rho": cfg.rho,
        "ranked_variants": ranked,
        "best_variant": ranked[0]["variant"] if ranked else None,
    }


def calibration_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """1 - Brier as calibration proxy (higher is better)."""
    if not y_true:
        return 0.0
    brier = sum((float(y) - float(p)) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)
    return float(max(0.0, 1.0 - brier))
