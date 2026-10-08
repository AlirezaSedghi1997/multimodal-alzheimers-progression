from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Any

from src.models.backbones.backbone_factory import AVAILABLE_BACKBONES


class MatchedBenchmarkError(ValueError):
    pass


def paired_bootstrap_metric_delta(
    y_true: list[int],
    metric_a: list[float],
    metric_b: list[float],
    metric_fn,
    *,
    resamples: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    """Compute a paired subject-level bootstrap delta without exporting vectors.

    The caller should keep predictions in memory and persist only this summary.
    Resamples with a single class are skipped because ROC/PR metrics are
    undefined there; Brier-like metrics remain valid.
    """

    if not (len(y_true) == len(metric_a) == len(metric_b)) or not y_true:
        raise MatchedBenchmarkError("PAIRED_BOOTSTRAP_INPUT_LENGTH_MISMATCH")
    rng = random.Random(int(seed))
    deltas: list[float] = []
    indices = list(range(len(y_true)))
    attempts = 0
    max_attempts = max(int(resamples) * 4, int(resamples))
    while len(deltas) < int(resamples) and attempts < max_attempts:
        attempts += 1
        sample = [rng.choice(indices) for _ in indices]
        y = [int(y_true[index]) for index in sample]
        try:
            value_a = float(metric_fn(y, [float(metric_a[index]) for index in sample]))
            value_b = float(metric_fn(y, [float(metric_b[index]) for index in sample]))
        except (ValueError, ZeroDivisionError):
            continue
        delta = value_b - value_a
        if math.isfinite(delta):
            deltas.append(delta)
    if not deltas:
        raise MatchedBenchmarkError("PAIRED_BOOTSTRAP_NO_VALID_RESAMPLES")
    deltas.sort()
    low = deltas[max(int(0.025 * len(deltas)), 0)]
    high = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    return {
        "status": "COMPLETED",
        "resamples_requested": int(resamples),
        "resamples_completed": len(deltas),
        "seed": int(seed),
        "delta_definition": "metric_b_minus_metric_a",
        "delta_mean": sum(deltas) / len(deltas),
        "ci95": [low, high],
        "subject_level_vectors_exported": False,
    }


def validate_matched_backbone_grid(config: dict[str, Any]) -> dict[str, Any]:
    backbones = list(config.get("backbones", []))
    common = deepcopy(config.get("common", {}))
    if len(backbones) < 2 or len(set(backbones)) != len(backbones):
        raise MatchedBenchmarkError("AT_LEAST_TWO_UNIQUE_BACKBONES_REQUIRED")
    forbidden = {"backbone", "model_name", "architecture"}
    if forbidden.intersection(common):
        raise MatchedBenchmarkError("BACKBONE_MUST_BE_THE_ONLY_ARCHITECTURE_DIFFERENCE")
    if config.get("folds") != 5:
        raise MatchedBenchmarkError("METHOD_X_BENCHMARK_REQUIRES_LOCKED_FIVE_FOLDS")
    unknown = sorted(set(backbones).difference(AVAILABLE_BACKBONES))
    if unknown:
        raise MatchedBenchmarkError(f"UNIMPLEMENTED_BACKBONES:{unknown}")
    if config.get("equal_hpo_budget") is not True:
        raise MatchedBenchmarkError("EQUAL_HPO_BUDGET_MUST_BE_EXPLICIT")
    if int(config.get("trials_per_family", 0)) < 1:
        raise MatchedBenchmarkError("TRIALS_PER_FAMILY_REQUIRED")
    required_common = {
        "fusion", "task_loss", "optimizer", "learning_rate", "scheduler",
        "dropout", "augmentation", "ema", "pretrained", "batch_size",
        "outer_test_access",
    }
    missing_common = sorted(required_common.difference(common))
    if missing_common:
        raise MatchedBenchmarkError(f"MATCHED_COMMON_PARAMETERS_MISSING:{missing_common}")
    return {
        "status": "PASS",
        "backbone_count": len(backbones),
        "backbones": backbones,
        "common_parameter_digest_input": common,
        "equal_hpo_budget": True,
        "trials_per_family": int(config["trials_per_family"]),
        "claim_status": "INTERNAL_ONLY_TECHNICAL_BENCHMARK",
    }
