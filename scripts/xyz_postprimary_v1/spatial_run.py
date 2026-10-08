"""Per-case/per-baseline commits for bounded, frozen-model spatial diagnostics."""
from __future__ import annotations

import gc
import json
import time

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from src.evaluation.signed_spatial_validation_v1 import (
    integrated_gradients, perturbation_controls, reset_model_in_memory, signed_similarity)
from scripts.xyz_fusion_v1.runtime import gpu_guard
from .context import bound_job, output_hashes, write_array
from .rendering import native_maps, overlay, save_nifti
from .spatial import fusion_coefficient, load_case


def spatial_case(ctx, store, case, policy):
    name = case["case_id"]
    if (store.root / "private/jobs" / (name + ".json")).exists():
        return bound_job(store, name)
    gpu_guard()
    started = time.perf_counter()
    model, image, forward, scan, provenance = load_case(ctx, case, torch.device("cuda"))
    array = image.detach().cpu().numpy().squeeze()
    bank = np.load(ctx["parent"] / "private/mri_cache.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(array, bank[case["cache_row"], 0]): raise ValueError("XAI_IMAGE_NOT_TRAINING_INPUT")
    del bank
    mask = array != 0  # Background was held exactly zero by the locked MRI normalization.
    directory = store.root / "private/cases" / name
    directory.mkdir(parents=True, exist_ok=True)
    baselines = {}; maps = {}; details = {}
    for kind in ("zero", "blur"):
        base = array.copy()
        base[mask] = 0 if kind == "zero" else gaussian_filter(array, sigma=policy["blur_sigma"])[mask]
        baselines[kind] = torch.tensor(base, dtype=image.dtype, device=image.device).reshape_as(image)
        def compute(kind=kind):
            def progress(n, i):
                print(json.dumps({"event": "ATTRIBUTION_PROGRESS", "case": name,
                                  "baseline": kind, "quadrature": n, "completed": i}), flush=True)
            value, numeric = integrated_gradients(forward, image, baselines[kind],
                                                  steps=tuple(policy["ig_steps"]), progress=progress)
            if np.count_nonzero(value[~mask]): raise ValueError("ATTRIBUTION_BACKGROUND_CHANGED")
            target = directory / ("ig_" + kind + ".npy"); write_array(target, value)
            return {"numerical": numeric, "outputs": output_hashes(store.root, [target])}
        details[kind] = bound_job(store, name + "_ig_" + kind, compute)
        maps[kind] = np.load(directory / ("ig_" + kind + ".npy"), allow_pickle=False)
    coefficient = fusion_coefficient(ctx, case["fold"])
    fidelity = bound_job(store, name + "_fidelity", lambda: perturbation_controls(
        forward, image, baselines["blur"], maps["blur"], mask,
        seed=policy["seed"]+case["fold"], repeats=policy["random_controls"], fraction=.05))
    def randomization():
        reset_count = reset_model_in_memory(model, seed=policy["seed"])
        value, numeric = integrated_gradients(forward, image, baselines["blur"], steps=tuple(policy["ig_steps"]))
        target = directory / "randomized_ig_blur.npy"; write_array(target, value)
        return {"reset_modules": reset_count, "numerical": numeric,
                "similarity": signed_similarity(maps["blur"], value, mask),
                "label_permutation": "NOT_EXECUTED_REQUIRES_NEW_TRAINING",
                "outputs": output_hashes(store.root, [target])}
    sanity = bound_job(store, name + "_randomized", randomization)
    # All required frozen-model operations precede in-memory randomization.
    del model, forward, image, baselines; gc.collect(); torch.cuda.empty_cache()
    def finish():
        scan_folder = ctx["parent"] / "private/mri_scans" / case["sample_key"]
        reference = nib.load(scan_folder / "registered_t1.nii.gz")
        model_maps = {"mri_expert_ig_blur": maps["blur"],
                      "fusion_loghazard_mri_ig_blur": maps["blur"]*coefficient,
                      "mri_expert_ig_zero": maps["zero"]}
        for key, value in model_maps.items(): save_nifti(directory/(key+".nii"), value, reference.affine)
        overlay(reference, {k: nib.Nifti1Image(v, reference.affine) for k, v in model_maps.items()},
                directory / "model_space_overlay.png", "Actual registered subject MRI; frozen-model attribution")
        native = native_maps(case["raw_path"], scan_folder/"registered_t1.nii.gz", scan_folder,
                             {k: model_maps[k] for k in list(model_maps)[:2]}, mask, directory/"native")
        correlation = sanity["similarity"].get("pearson")
        positive = fidelity.get("positive", {})
        validated = (details["blur"]["numerical"]["numerical_status"] == "PASS"
                     and sanity["numerical"]["numerical_status"] == "PASS"
                     and correlation is not None and abs(correlation) < policy["max_randomized_abs_correlation"]
                     and positive.get("directional_margin_vs_random_mean", -1) > 0
                     and positive.get("descriptive_random_tail_fraction", 1) <= .05)
        outputs = output_hashes(store.root, [p for p in directory.rglob("*") if p.is_file()])
        return {"case_id": name, "fold": case["fold"], "provenance": provenance,
                "numerical": {k: d["numerical"] for k, d in details.items()},
                "baseline_sensitivity": signed_similarity(maps["zero"], maps["blur"], mask),
                "fidelity": fidelity, "sanity": sanity, "native": native, "outputs": outputs,
                "fusion_mri_coefficient": coefficient, "seconds": time.perf_counter()-started,
                "status": "VALIDATED_MODEL_ATTRIBUTION" if validated else "DIAGNOSTIC_ONLY_VALIDATION_NOT_PASSED",
                "disease_localization_allowed": False, "anatomical_roi_labels": "BLOCKED_ATLAS_SPACE_MAPPING",
                "claim_status": "POST_HOC_INTERNAL_EXPLORATORY"}
    return bound_job(store, name, finish)
