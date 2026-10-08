"""Forward-based, aggregate-only attribution primitives for Method X.

The implementation uses input-gradient saliency rather than pretending that
every backbone exposes a comparable convolutional Grad-CAM layer.  It is a
valid attribution method for the registered classifier, is deterministic under
the locked seeds, and is deliberately kept separate from claim wording: a
saliency region is never a disease or causal-region assertion.
"""

from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.evaluation.xai_regions import compute_region_attribution
from src.evaluation.xai_validation import (
    XAIValidationError,
    evaluate_occlusion_fidelity,
    evaluate_randomization_sanity,
)


def _state_dict_from_checkpoint(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    for key in ("best_averaged_state_dict", "best_model_state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value:
            return dict(value), key
    raise XAIValidationError("XAI_MODEL_STATE_MISSING")


def build_model_from_checkpoint(checkpoint: Mapping[str, Any], *, device: torch.device) -> tuple[torch.nn.Module, str]:
    """Construct the exact classifier declared by a finalized checkpoint."""

    from src.models.backbones.lightweight_3d import MethodX3DClassifier

    state, state_source = _state_dict_from_checkpoint(checkpoint)
    hidden = checkpoint.get("fusion_hidden_dim")
    hidden = None if hidden in (None, 0, "0") else int(hidden)
    constructor_values = {
        "in_channels": 1,
        "latent_dim": int(checkpoint.get("latent_dim", 128)),
        "tab_hidden_dim": int(checkpoint.get("tab_hidden_dim", 32)),
        "tab_dim": 5,
        "fusion": str(checkpoint.get("fusion", "cross_attention")),
        "dropout": float(checkpoint.get("dropout", 0.1)),
        "backbone": str(checkpoint.get("backbone", "tiny")),
        "align_dim": int(checkpoint["align_dim"]) if checkpoint.get("align_dim") is not None else None,
        "bottleneck_dim": int(checkpoint.get("bottleneck_dim", 32)),
        "fusion_hidden_dim": hidden,
        "gradient_checkpointing": False,
        "attention": str(checkpoint.get("attention", "none")),
        "alignment_head": str(checkpoint.get("alignment_head", "linear")),
        "alignment_head_mode": checkpoint.get("alignment_head_mode"),
        "tab_encoder": str(checkpoint.get("tab_encoder", "linear")),
        "causal_adjustment": str(checkpoint.get("causal_adjustment", "none")),
        "n_experts": int(checkpoint.get("n_experts", 4)),
        "fusion_num_heads": int(checkpoint.get("fusion_num_heads", 2)),
        "moe_load_balance_lambda": float(checkpoint.get("moe_load_balance_lambda", 0.0)),
        "channel_gate": str(checkpoint.get("q2_channel_gate", checkpoint.get("channel_gate", "none"))),
        "hybrid_variant": str(checkpoint.get("hybrid_variant", "baseline")),
        "hybrid_attention_heads": int(checkpoint.get("hybrid_attention_heads", 2)),
        "hybrid_attention_window_size": int(checkpoint.get("hybrid_attention_window_size", 4)),
    }
    parameters = inspect.signature(MethodX3DClassifier).parameters
    unsupported_hybrid = {
        "hybrid_variant": constructor_values["hybrid_variant"] != "baseline",
        "hybrid_attention_heads": constructor_values["hybrid_attention_heads"] != 2,
        "hybrid_attention_window_size": constructor_values["hybrid_attention_window_size"] != 4,
    }
    if any(value and key not in parameters for key, value in unsupported_hybrid.items()):
        raise XAIValidationError("XAI_MODEL_ARCHITECTURE_UNSUPPORTED")
    constructor_values = {key: value for key, value in constructor_values.items() if key in parameters}
    try:
        model = MethodX3DClassifier(**constructor_values)
    except (TypeError, ValueError) as exc:
        raise XAIValidationError("XAI_MODEL_CONSTRUCTION_INCOMPATIBLE") from exc
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise XAIValidationError("XAI_MODEL_STATE_INCOMPATIBLE") from exc
    model.to(device)
    model.eval()
    return model, state_source


def input_gradient_attribution(
    model: torch.nn.Module,
    image: torch.Tensor,
    tabular: torch.Tensor,
    tab_presence: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized absolute input gradients and probabilities."""

    image = image.detach().clone().requires_grad_(True)
    tabular = tabular.detach()
    presence = tab_presence.detach() if tab_presence is not None else None
    model.zero_grad(set_to_none=True)
    logits = model(image, tabular, tab_presence=presence)
    score = logits.reshape(-1).sum()
    gradients = torch.autograd.grad(score, image, retain_graph=False, create_graph=False)[0]
    saliency = gradients.detach().abs()
    if saliency.ndim == 5:
        saliency = saliency[:, 0]
    flat = saliency.reshape(saliency.shape[0], -1)
    saliency = saliency / flat.amax(dim=1).clamp_min(1e-12).reshape(-1, 1, 1, 1)
    probabilities = torch.sigmoid(logits.detach().reshape(-1))
    return saliency.cpu().numpy(), probabilities.cpu().numpy()


def _randomized_copy(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    randomized = copy.deepcopy(model)
    generator = torch.Generator(device=next(randomized.parameters()).device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for parameter in randomized.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, device=parameter.device, dtype=parameter.dtype) * 0.02)
    randomized.eval()
    return randomized


def _aggregate_regions(attribution_map: np.ndarray, atlas_data: np.ndarray, region_names: Mapping[int, str]) -> list[dict[str, Any]]:
    atlas_ids = {int(value) for value in np.unique(atlas_data) if int(value) > 0}
    label_ids = {int(value) for value in region_names}
    if not atlas_ids or atlas_ids != label_ids:
        raise XAIValidationError("XAI_ATLAS_LABEL_COVERAGE_MISMATCH")
    rows = compute_region_attribution(attribution_map, atlas_data, dict(region_names))
    return [
        {
            "atlas_id": int(row.region_id),
            "region": str(row.region_name),
            "mean_attribution": float(row.mean_attribution),
            "absolute_attribution": float(row.abs_mean_attribution),
            "voxel_count": int(row.voxel_count),
        }
        for row in rows
    ]


def _region_bootstrap(values: Sequence[float], *, seed: int, resamples: int = 1000) -> dict[str, float | int | None]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"n_folds": 0, "mean": None, "median": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(int(seed))
    samples = np.asarray(values, dtype=float)
    means = np.asarray([rng.choice(samples, size=len(samples), replace=True).mean() for _ in range(int(resamples))])
    return {
        "n_folds": len(values),
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def run_fold_attribution(
    *,
    model: torch.nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
    atlas_data: np.ndarray,
    region_names: Mapping[int, str],
    randomization_seeds: Sequence[int] = (13, 29, 47),
    bootstrap_resamples: int = 1000,
) -> dict[str, Any]:
    """Compute one fold's attribution and faithfulness gates.

    ``batches`` is already restricted to a fold-local locked split and carries
    no identifiers.  Only aggregate maps and gate statistics leave this
    function.
    """

    if not batches:
        raise XAIValidationError("XAI_FOLD_INPUTS_EMPTY")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise XAIValidationError("XAI_MODEL_HAS_NO_PARAMETERS") from exc

    def _device_batch(
        image: torch.Tensor,
        tabular: torch.Tensor,
        presence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return (
            image.to(device),
            tabular.to(device),
            presence.to(device) if presence is not None else None,
        )

    map_sum: np.ndarray | None = None
    sample_count = 0
    for image, tabular, presence in batches:
        image, tabular, presence = _device_batch(image, tabular, presence)
        maps, probs = input_gradient_attribution(model, image, tabular, presence)
        del probs
        maps = np.asarray(maps, dtype=np.float32)
        batch_sum = maps.sum(axis=0, dtype=np.float32)
        map_sum = batch_sum if map_sum is None else map_sum + batch_sum
        sample_count += int(maps.shape[0])
    if map_sum is None or sample_count <= 0:
        raise XAIValidationError("XAI_FOLD_INPUTS_EMPTY")
    mean_map = np.asarray(map_sum / float(sample_count), dtype=np.float32)
    if tuple(mean_map.shape) != tuple(atlas_data.shape):
        raise XAIValidationError("XAI_ATLAS_ATTRIBUTION_GEOMETRY_MISMATCH")
    regions = _aggregate_regions(mean_map, atlas_data, region_names)
    if not regions:
        raise XAIValidationError("XAI_ATLAS_REGION_COVERAGE_EMPTY")
    top_id = int(regions[0]["atlas_id"])
    region_ids = [int(value) for value in np.unique(atlas_data) if int(value) > 0]
    random_ids = [value for value in region_ids if value != top_id][:3]
    if not random_ids:
        raise XAIValidationError("XAI_RANDOM_REGION_CONTROLS_EMPTY")

    # Faithfulness must be estimated over every held-out sample, not the first
    # sample only.  A first-sample shortcut can make an occlusion gate pass by
    # accident when that sample is atypical.
    original_probabilities: list[float] = []
    top_probabilities: list[float] = []
    random_probabilities_by_region: dict[int, list[float]] = {region_id: [] for region_id in random_ids}
    with torch.no_grad():
        for image, tabular, presence in batches:
            image, tabular, presence = _device_batch(image, tabular, presence)
            original_probabilities.extend(
                float(value)
                for value in torch.sigmoid(model(image, tabular, tab_presence=presence).reshape(-1)).detach().cpu()
            )
            atlas_mask = torch.as_tensor(atlas_data == top_id, device=device, dtype=torch.bool)
            while atlas_mask.ndim < image.ndim:
                atlas_mask = atlas_mask.unsqueeze(0)
            top_image = image.clone().masked_fill(atlas_mask, 0.0)
            top_probabilities.extend(
                float(value)
                for value in torch.sigmoid(model(top_image, tabular, tab_presence=presence).reshape(-1)).detach().cpu()
            )
            for region_id in random_ids:
                random_image = image.clone()
                random_mask = torch.as_tensor(atlas_data == region_id, device=device, dtype=torch.bool)
                while random_mask.ndim < random_image.ndim:
                    random_mask = random_mask.unsqueeze(0)
                random_image = random_image.masked_fill(random_mask, 0.0)
                random_probabilities_by_region[region_id].extend(
                    float(value)
                    for value in torch.sigmoid(
                        model(random_image, tab_presence=presence, tabular=tabular).reshape(-1)
                    ).detach().cpu()
                )

    randomization_checks: dict[int, dict[str, Any]] = {}
    for seed in randomization_seeds:
        randomized_model = _randomized_copy(model, int(seed))
        randomized_sum: np.ndarray | None = None
        randomized_count = 0
        for image, tabular, presence in batches:
            image, tabular, presence = _device_batch(image, tabular, presence)
            maps, _ = input_gradient_attribution(randomized_model, image, tabular, presence)
            maps = np.asarray(maps, dtype=np.float32)
            batch_sum = maps.sum(axis=0, dtype=np.float32)
            randomized_sum = batch_sum if randomized_sum is None else randomized_sum + batch_sum
            randomized_count += int(maps.shape[0])
        if randomized_sum is None or randomized_count != sample_count:
            raise XAIValidationError("XAI_RANDOMIZATION_SAMPLE_COUNT_MISMATCH")
        randomized_mean = np.asarray(randomized_sum / float(randomized_count), dtype=np.float32)
        randomization_checks[int(seed)] = evaluate_randomization_sanity(mean_map, randomized_mean)
        del randomized_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gate = {
        "occlusion_fidelity": evaluate_occlusion_fidelity(
            original_prediction=sum(original_probabilities) / len(original_probabilities),
            top_roi_prediction=sum(top_probabilities) / len(top_probabilities),
            random_roi_predictions=[
                value for values in random_probabilities_by_region.values() for value in values
            ],
        ),
        "randomization_sanity": [
            {"seed": int(seed), **randomization_checks[int(seed)]} for seed in sorted(randomization_checks)
        ],
    }
    return {
        "region_attribution": regions,
        "top_region_id": top_id,
        "top_region": regions[0]["region"],
        "gate": gate,
        "sample_count": sample_count,
        "bootstrap_resamples": int(bootstrap_resamples),
        "randomization_seeds": [int(seed) for seed in randomization_seeds],
        "aggregate_only": True,
        "contains_subject_level_outputs": False,
    }


def aggregate_fold_regions(fold_results: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int = 1000) -> list[dict[str, Any]]:
    """Combine fold-level region values with a fold bootstrap CI."""

    by_region: dict[int, dict[str, Any]] = {}
    for fold in fold_results:
        fold_index = int(fold.get("fold", 0))
        for row in fold.get("region_attribution", []) if isinstance(fold.get("region_attribution"), list) else []:
            region_id = int(row.get("atlas_id", -1))
            if region_id <= 0:
                continue
            bucket = by_region.setdefault(region_id, {"region": row.get("region"), "values": [], "folds": []})
            bucket["values"].append(float(row.get("mean_attribution", 0.0)))
            bucket["folds"].append(fold_index)
    rows: list[dict[str, Any]] = []
    for region_id, bucket in by_region.items():
        ci = _region_bootstrap(bucket["values"], seed=13 + region_id, resamples=bootstrap_resamples)
        rows.append({"atlas_id": region_id, "region": bucket["region"], **ci, "folds": sorted(set(bucket["folds"]))})
    return sorted(rows, key=lambda row: abs(float(row.get("mean") or 0.0)), reverse=True)
