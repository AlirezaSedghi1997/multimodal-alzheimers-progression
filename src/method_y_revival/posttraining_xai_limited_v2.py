"""Anatomical-mask repair for frozen Method Y PET attribution maps.

The correction is map * mask after attribution. This is algebraically equal
to the existing attribution functions' foreground_mask argument. It does not
change model input, prediction, gradient, or any outer performance metric.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from .pet_attribution_v4 import risk_from_logits
from .image_modeling_v3 import survival_at_times
from .posttraining_correction_v4_2 import CorrectionV42Error
from .posttraining_xai_limited_v1 import map_agreement


def load_locked_brain_mask(mask_path: Path, reference_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Nearest-neighbour, affine-aware MNI mask in the exact model grid."""
    if not mask_path.is_file() or not reference_path.is_file():
        raise CorrectionV42Error("XAI_BRAIN_MASK_OR_REFERENCE_MISSING")
    lock_path = reference_path.with_name("reference_128_1p5mm.lock.json")
    if not lock_path.is_file():
        raise CorrectionV42Error("XAI_REFERENCE_LOCK_MISSING")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if (lock.get("status") != "REFERENCE_TEMPLATE_LOCKED"
            or lock.get("source_brain_mask_sha256") != mask_sha
            or lock.get("output_template_sha256") != reference_sha):
        raise CorrectionV42Error("XAI_REFERENCE_ASSET_HASH_MISMATCH")
    source = nib.load(str(mask_path))
    reference = nib.load(str(reference_path))
    if source.ndim != 3 or reference.shape != (128, 128, 128):
        raise CorrectionV42Error("XAI_BRAIN_MASK_GEOMETRY_INVALID")
    if not np.isfinite(source.affine).all() or not np.isfinite(reference.affine).all():
        raise CorrectionV42Error("XAI_BRAIN_MASK_AFFINE_INVALID")
    adapted = resample_from_to(source, reference, order=0)
    if not np.allclose(adapted.affine, reference.affine, rtol=0, atol=1e-6):
        raise CorrectionV42Error("XAI_BRAIN_MASK_REFERENCE_MISMATCH")
    mask = np.asarray(adapted.dataobj) > 0.5
    fraction = float(mask.mean())
    if not 0.1 < fraction < 0.5:
        raise CorrectionV42Error("XAI_BRAIN_MASK_IMPLAUSIBLE_COVERAGE")
    digest = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
    return mask, {"source_file_sha256": mask_sha,
                  "reference_file_sha256": reference_sha,
                  "reference_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                  "resampled_mask_sha256": digest, "brain_fraction": fraction,
                  "source_space": "MNI152NLin2009cAsym_res-01",
                  "model_space": "MNI152NLin2009cAsym_128_1p5mm",
                  "resampling": "AFFINE_AWARE_NEAREST_NEIGHBOR",
                  "role": "ANATOMICAL_DISPLAY_AND_ATTRIBUTION_QC_NOT_QUANTITATIVE_PET_ROI"}


def mask_maps_and_qc(maps: dict[str, np.ndarray], mask: np.ndarray) -> tuple[dict, dict]:
    expected = {"layercam_primary", "layercam_sensitivity", "gradcam_primary",
                "scorecam_primary"}
    if set(maps) != expected or mask.shape != (128, 128, 128):
        raise CorrectionV42Error("XAI_MASKED_MAP_SCHEMA_MISMATCH")
    corrected: dict[str, np.ndarray] = {}
    qc: dict[str, dict[str, Any]] = {}
    for name, original in maps.items():
        array = np.asarray(original, dtype=np.float32)
        if array.shape != mask.shape or not np.isfinite(array).all():
            raise CorrectionV42Error("XAI_MASKED_MAP_INVALID:" + name)
        absolute = np.abs(array)
        total = float(absolute.sum(dtype=np.float64))
        outside = float(absolute[~mask].sum(dtype=np.float64))
        trimmed = np.where(mask, array, 0.0).astype(np.float32)
        corrected[name] = trimmed
        qc[name] = {
            "outside_brain_absolute_mass_fraction_before_mask":
                (outside / total if total > 0 else None),
            "inside_brain_nonzero": bool(np.max(np.abs(trimmed)) > 1e-12),
            "outside_brain_exact_zero_after_mask": bool(np.all(trimmed[~mask] == 0)),
            "status": "ANATOMICALLY_MASKED" if np.max(np.abs(trimmed)) > 1e-12
                      else "NOT_ESTIMABLE_ZERO_MAP",
        }
    return corrected, qc


def corrected_agreement(maps: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "risk_target": map_agreement(maps["layercam_primary"], maps["layercam_sensitivity"]),
        "layer_vs_score": map_agreement(maps["layercam_primary"], maps["scorecam_primary"]),
        "layer_vs_grad": map_agreement(maps["layercam_primary"], maps["gradcam_primary"]),
    }


def brain_outside_perturbation(model, image: torch.Tensor, tabular: torch.Tensor,
                               mask: np.ndarray, *, bin_edges: list[float],
                               horizon_days: float) -> dict[str, Any]:
    """Outcome-blind OOD diagnostic; never a causal localization test.

    The tabular vector is held fixed. Zeroing either image region creates an
    out-of-distribution input, so the magnitude only flags potential reliance
    on nonbrain voxels; a small outside effect does not validate localization.
    """
    if tuple(image.shape) != (1, 128, 128, 128) or mask.shape != (128, 128, 128):
        raise CorrectionV42Error("XAI_PERTURBATION_GEOMETRY_INVALID")
    brain = torch.from_numpy(mask.copy()).to(dtype=image.dtype).unsqueeze(0)
    volume, table = image.unsqueeze(0), tabular.unsqueeze(0)
    brain = brain.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        original = model(volume, table)
        brain_only = model(volume * brain, table)
        outside_only = model(volume * (1.0 - brain), table)
    scores = {}
    if not (len(bin_edges) == original.shape[1] + 1
            and bin_edges[0] < horizon_days < bin_edges[-1]):
        raise CorrectionV42Error("XAI_PERTURBATION_HORIZON_CONTRACT_INVALID")
    for mode in ("logit_sum", "mortality_last", "mortality_at_locked_horizon"):
        def value(logits: torch.Tensor) -> float:
            if mode == "mortality_at_locked_horizon":
                return float((1.0 - survival_at_times(
                    logits, bin_edges, [horizon_days])[0, 0]).item())
            return float(risk_from_logits(logits, mode=mode).item())
        baseline = value(original)
        remove_outside = value(brain_only)
        remove_brain = value(outside_only)
        if not np.isfinite([baseline, remove_outside, remove_brain]).all():
            raise CorrectionV42Error("XAI_PERTURBATION_NONFINITE")
        scores[mode] = {
            "original": baseline,
            "outside_removed": remove_outside,
            "brain_removed": remove_brain,
            "absolute_change_remove_outside": abs(baseline - remove_outside),
            "absolute_change_remove_brain": abs(baseline - remove_brain),
            "outside_change_ge_brain_change":
                abs(baseline - remove_outside) >= abs(baseline - remove_brain),
        }
    return {"status": "COMPUTED_OOD_DIAGNOSTIC_ONLY", "risk_modes": scores,
            "locked_horizon_days": horizon_days,
            "input_region": "AFFINE_AWARE_MNI_BRAIN_MASK_VS_COMPLEMENT",
            "tabular_held_fixed": True,
            "interpretation": "NONBRAIN_RELIANCE_FLAG_NOT_CAUSAL_OR_VALIDATION_PROOF"}
