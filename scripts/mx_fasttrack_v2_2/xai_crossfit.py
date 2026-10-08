"""Cross-fitted, class-targeted XAI checks; never a disease-localization test."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def crossfit_plan(n: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if n != 8:
        raise ValueError("XAI_REQUIRES_EIGHT_PREDECLARED_SAMPLES")
    return [(tuple(range(4)), tuple(range(4, 8))),
            (tuple(range(4, 8)), tuple(range(4)))]


def choose_size_matched_controls(atlas: np.ndarray, top_id: int) -> list[int]:
    """Three atlas regions within 1.5x the top ROI volume; deterministic ties."""
    labels, counts = np.unique(atlas, return_counts=True)
    volumes = {int(label): int(count) for label, count in zip(labels, counts, strict=True)
               if int(label) > 0}
    if top_id not in volumes:
        raise ValueError("XAI_TOP_REGION_NOT_IN_ATLAS")
    top_n = volumes[top_id]
    ranked = sorted(((abs(math.log(volume / top_n)), int(label))
                     for label, volume in volumes.items() if label != top_id
                     and 1 / 1.5 <= volume / top_n <= 1.5))
    if len(ranked) < 3:
        raise ValueError("XAI_THREE_SIZE_MATCHED_CONTROLS_UNAVAILABLE")
    return [label for _, label in ranked[:3]]


def paired_target_effect(original: float, top_occluded: float,
                         controls_occluded: Sequence[float]) -> dict:
    values = [original, top_occluded, *controls_occluded]
    if (len(controls_occluded) != 3 or any(not math.isfinite(float(v))
        or not 0 <= float(v) <= 1 for v in values)):
        raise ValueError("XAI_PROBABILITY_OR_CONTROL_COUNT_INVALID")
    target = int(original >= 0.5)
    sign = 1.0 if target else -1.0
    top_drop = sign * (original - top_occluded)
    control_drops = [sign * (original - value) for value in controls_occluded]
    median = float(np.median(control_drops))
    return {"target_class": target, "top_target_drop": float(top_drop),
            "control_target_drop_median": median,
            "top_minus_control_drop": float(top_drop - median)}


def randomization_check(original: np.ndarray, randomized: np.ndarray,
                        *, maximum_absolute_correlation: float = 0.9) -> dict:
    a = np.asarray(original, dtype=float).ravel()
    b = np.asarray(randomized, dtype=float).ravel()
    if (a.shape != b.shape or a.size < 2 or not np.isfinite(a).all()
        or not np.isfinite(b).all()):
        raise ValueError("XAI_RANDOMIZATION_MAP_INVALID")
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return {"status": "FAIL", "reason": "DEGENERATE_MAP", "absolute_correlation": None}
    correlation = abs(float(np.corrcoef(a, b)[0, 1]))
    return {"status": "PASS" if correlation < maximum_absolute_correlation else "FAIL",
            "absolute_correlation": correlation,
            "threshold": maximum_absolute_correlation}


def run_fold_crossfit(*, model, batches: Sequence, atlas_data: np.ndarray,
                      region_names: Mapping[int, str], randomization_seeds: Sequence[int],
                      frozen_original_probabilities: Sequence[float] | None = None) -> dict:
    """Assess each image only under a ROI chosen on the opposite four images.

    Batches are fixed inner-validation records. The same eight records train no
    new model here, but they were used during model selection: this remains an
    internal exploratory check, not held-out independent validation.
    """
    import torch
    from src.evaluation.method_x_xai_forward import (
        _aggregate_regions, _randomized_copy, input_gradient_attribution)

    crossfit_plan(len(batches))
    if list(randomization_seeds) != [13, 29, 47]:
        raise ValueError("XAI_RANDOMIZATION_SEEDS_DRIFT")
    device = next(model.parameters()).device
    atlas = np.asarray(atlas_data)
    if atlas.ndim != 3:
        raise ValueError("XAI_ATLAS_GEOMETRY_INVALID")
    if set(int(i) for i in np.unique(atlas) if i > 0) != set(region_names):
        raise ValueError("XAI_ATLAS_LABEL_COVERAGE_MISMATCH")

    def one(index: int):
        image, tabular, presence = batches[index]
        return (image.to(device), tabular.to(device),
                presence.to(device) if presence is not None else None)

    # Freeze the explanation target for every assessment image before any
    # attribution, ROI choice, or perturbation is computed. The vector stays
    # in memory; only its aggregate class counts are released.
    if frozen_original_probabilities is None:
        original_probabilities = []
        with torch.no_grad():
            for index in range(8):
                image, tabular, presence = one(index)
                original_probabilities.append(float(torch.sigmoid(
                    model(image, tabular, tab_presence=presence).reshape(-1))[0].item()))
    else:
        original_probabilities = [float(value) for value in frozen_original_probabilities]
        if (len(original_probabilities) != 8
            or any(not math.isfinite(value) or not 0 <= value <= 1
                   for value in original_probabilities)):
            raise ValueError("XAI_PRECOMMITTED_TARGET_VECTOR_INVALID")
    frozen_target_counts = {"negative": sum(value < 0.5 for value in original_probabilities),
                            "positive": sum(value >= 0.5 for value in original_probabilities)}

    maps = []
    for index in range(8):
        image, tabular, presence = one(index)
        map_array, _ = input_gradient_attribution(model, image, tabular, presence)
        if map_array.shape != (1, *atlas.shape):
            raise ValueError("XAI_ATTRIBUTION_ATLAS_GEOMETRY_MISMATCH")
        maps.append(np.asarray(map_array[0], dtype=np.float32))
    directions = []
    all_effects = []
    all_top_drops = []
    for select, assess in crossfit_plan(8):
        selection_map = np.mean([maps[index] for index in select], axis=0)
        regions = _aggregate_regions(selection_map, atlas, region_names)
        top_id = int(regions[0]["atlas_id"])
        controls = choose_size_matched_controls(atlas, top_id)
        effects = []
        top_drops = []
        with torch.no_grad():
            for index in assess:
                image, tabular, presence = one(index)

                def probability(roi: int | None) -> float:
                    x = image
                    if roi is not None:
                        mask = torch.as_tensor(atlas == roi, device=device, dtype=torch.bool)
                        while mask.ndim < x.ndim:
                            mask = mask.unsqueeze(0)
                        x = x.clone().masked_fill(mask, 0.0)
                    return float(torch.sigmoid(model(x, tabular, tab_presence=presence)
                                               .reshape(-1))[0].item())

                effect = paired_target_effect(original_probabilities[index], probability(top_id),
                                              [probability(roi) for roi in controls])
                effects.append(effect["top_minus_control_drop"])
                top_drops.append(effect["top_target_drop"])
        all_effects.extend(effects)
        all_top_drops.extend(top_drops)
        directions.append({"top_region_id": top_id, "top_region": regions[0]["region"],
                           "top_volume": int(np.count_nonzero(atlas == top_id)),
                           "controls": [{"atlas_id": roi,
                                         "volume": int(np.count_nonzero(atlas == roi))}
                                        for roi in controls],
                           "selection_n": len(select), "assessment_n": len(assess),
                           "assessment_mean_top_target_drop": float(np.mean(top_drops)),
                           "assessment_mean_paired_effect": float(np.mean(effects))})

    original = np.mean(maps, axis=0)
    sanity = []
    for seed in randomization_seeds:
        randomized = _randomized_copy(model, int(seed))
        random_maps = []
        for index in range(8):
            image, tabular, presence = one(index)
            map_array, _ = input_gradient_attribution(randomized, image, tabular, presence)
            random_maps.append(np.asarray(map_array[0], dtype=np.float32))
        sanity.append({"seed": int(seed),
                       **randomization_check(original, np.mean(random_maps, axis=0))})
        del randomized
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean_effect = float(np.mean(all_effects))
    mean_top_drop = float(np.mean(all_top_drops))
    return {"sample_count": 8, "crossfit_directions": directions,
            "target_definition": "FROZEN_UNOCCLUDED_PREDICTED_CLASS_BEFORE_ROI_SELECTION",
            "frozen_target_class_counts": frozen_target_counts,
            "paired_mean_effect": mean_effect,
            "paired_mean_top_target_drop": mean_top_drop,
            "paired_positive_count": int(sum(value > 0 for value in all_effects)),
            "randomization_sanity": sanity,
            "gate_status": ("PASS_EXPLORATORY" if mean_effect > 0 and mean_top_drop > 0
                            and all(row["status"] == "PASS" for row in sanity)
                            else "BLOCKED_EXPLORATORY_GATE"),
            "occlusion_fill": "zero_in_normalized_input_out_of_distribution_possible",
            "model_selection_overlap": True,
            "aggregate_only": True, "subject_level_outputs_written": False}


def aggregate_five_folds(folds: Sequence[dict], *, resamples: int = 1000) -> dict:
    if (len(folds) != 5 or [row.get("fold") for row in folds] != list(range(5))
        or any(row.get("sample_count") != 8 for row in folds)):
        raise ValueError("XAI_FIVE_FOLDS_REQUIRED")
    values = np.asarray([row["paired_mean_effect"] for row in folds], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("XAI_NONFINITE_FOLD_EFFECT")
    rng = np.random.default_rng(13)
    means = np.asarray([rng.choice(values, size=5, replace=True).mean()
                        for _ in range(resamples)])
    return {"fold_mean_effect": float(np.mean(values)),
            "fold_bootstrap_ci95": [float(np.quantile(means, 0.025)),
                                    float(np.quantile(means, 0.975))],
            "resampling_unit": "fold_not_image", "n_folds": 5,
            "gate_status": ("COMPLETED_EXPLORATORY_CROSSFIT"
                            if all(row.get("gate_status") == "PASS_EXPLORATORY" for row in folds)
                            else "BLOCKED_EXPLORATORY_GATE")}
