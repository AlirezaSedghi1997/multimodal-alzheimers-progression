"""Run BrainIAC R1a sensitivity analysis on 64 sample volumes from cache."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(r"H:\New project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.brainiac_adapter import (
    ADAPTER_SCHEMA_VERSION,
    evaluate_geometry,
    geometry_manifest,
    adapt_volume,
    nonzero_normalize,
    sensitivity_pair,
    adapter_provenance,
    mask_edge_perturbation,
)
from src.utils.io_helpers import atomic_write_json

CACHE_PATH = ROOT / "data" / "cached" / "t1_volume_cache_128.h5"
OUT_DIR = ROOT / "results" / "final_results" / "method_x_brainiac_caft_r1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUT_DIR / "r1a_adapter_geometry_report.json"


def build_r1a_sensitivity_report(
    *,
    geometry_status: str,
    relative_differences: list[float],
    mask_edge_differences: list[float],
    sample_records: list[dict[str, object]],
    geometry: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a sensitivity-only report without upgrading UNKNOWN to PASS."""

    values = [*relative_differences, *mask_edge_differences]
    if not values or not all(np.isfinite(float(value)) for value in values):
        raise RuntimeError("R1A_SENSITIVITY_METRIC_NONFINITE")
    mean_relative = float(np.mean(relative_differences))
    std_relative = float(np.std(relative_differences))
    mean_mask = float(np.mean(mask_edge_differences))
    geometry_pass = str(geometry_status) == "PASS"
    status = (
        "SENSITIVITY_COMPLETE"
        if geometry_pass
        else "SENSITIVITY_COMPLETE_WITH_GEOMETRY_BLOCK"
    )
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "status": status,
        "route_role": "SENSITIVITY_ONLY_NOT_CANONICAL_PREPROCESSING",
        "primary_route_authorized": False,
        "label_read": False,
        "raw_identifier_stored": False,
        "raw_path_stored": False,
        "geometry": dict(geometry or {"status": geometry_status}),
        "samples_evaluated": len(sample_records),
        "mean_relative_difference_staged_vs_direct": mean_relative,
        "std_relative_difference_staged_vs_direct": std_relative,
        "mean_mask_edge_perturbation_sensitivity": mean_mask,
        "skull_stripping_shortcut_risk_level": "UNRESOLVED",
        "conclusion": (
            f"The index-space sensitivity arms differed by a mean of "
            f"{mean_relative * 100.0:.3f}%. This does not validate physical "
            "geometry, skull stripping, embedding stability, or prediction "
            "fidelity; the primary BrainIAC route remains unauthorized."
        ),
        "samples_head": sample_records[:10],
    }


def run_r1a_sensitivity(sample_n: int = 64) -> dict[str, object]:
    geom_report = evaluate_geometry()
    geom_man = geometry_manifest()

    if not CACHE_PATH.exists():
        raise RuntimeError(f"CACHE_NOT_FOUND:{CACHE_PATH}")

    diffs = []
    perturbation_diffs = []
    sample_records = []

    with h5py.File(CACHE_PATH, "r") as h5:
        image_dset = h5["image"]
        is_synth = h5["is_synthetic"][()] if "is_synthetic" in h5 else np.zeros(image_dset.shape[0], dtype=bool)
        sample_hashes = h5["sample_hash"][()]

        # Select real scans only for sensitivity evaluation
        real_indices = np.where(~is_synth)[0][:sample_n]
        print(f"Loaded {len(real_indices)} real scan indices for R1a sensitivity audit.")

        for idx in real_indices:
            raw_vol = image_dset[idx]
            vol = np.squeeze(np.asarray(raw_vol, dtype=np.float32))

            raw_hash_val = sample_hashes[idx]
            if isinstance(raw_hash_val, bytes):
                sample_hash_str = raw_hash_val.decode("utf-8", errors="replace")
            else:
                sample_hash_str = str(raw_hash_val)

            sens = sensitivity_pair(vol)
            diffs.append(sens["relative_difference"])

            # Mask-edge perturbation test.  The shared implementation shifts
            # masks without wrap-around and probes both erosion and dilation.
            adapted_base = adapt_volume(vol, purpose="sensitivity_only")
            edge_differences = []
            for mode in ("erode", "dilate"):
                perturbed_vol = mask_edge_perturbation(vol, mode=mode, voxels=1)
                adapted_perturbed = adapt_volume(
                    perturbed_vol, purpose="sensitivity_only"
                )
                edge_differences.append(
                    float(
                        np.abs(adapted_base - adapted_perturbed).mean()
                        / (np.abs(adapted_base).mean() + 1e-6)
                    )
                )
            pert_diff = max(edge_differences)
            perturbation_diffs.append(pert_diff)

            sample_records.append({
                "sample_index": int(idx),
                "sample_hash": sample_hash_str,
                "relative_difference_staged_vs_direct": float(sens["relative_difference"]),
                "mask_edge_perturbation_sensitivity": float(pert_diff),
                "mean_adapted_intensity": float(adapted_base.mean()),
                "std_adapted_intensity": float(adapted_base.std()),
            })

    report = build_r1a_sensitivity_report(
        geometry_status=geom_report.status,
        relative_differences=diffs,
        mask_edge_differences=perturbation_diffs,
        sample_records=sample_records,
        geometry=geom_man,
    )

    atomic_write_json(REPORT_PATH, report)
    print(f"R1a Sensitivity report written to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    report = run_r1a_sensitivity(sample_n=64)
    print("Report status:", report["status"])
    print("Mean relative diff:", report["mean_relative_difference_staged_vs_direct"])
    print("Mean perturbation diff:", report["mean_mask_edge_perturbation_sensitivity"])
