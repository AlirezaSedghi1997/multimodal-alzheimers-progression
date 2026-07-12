#!/usr/bin/env python3
"""Audit 128^3 MRI geometry/orientation without exporting raw identifiers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="data/preprocessed_mri_128")
    parser.add_argument("--output", default="results/tables/mri_128_geometry_audit.json")
    args = parser.parse_args()

    import nibabel as nib

    input_dir = PROJECT_ROOT / args.input_dir
    files = sorted(input_dir.glob("*.nii.gz"))
    if not files:
        raise RuntimeError(f"No NIfTI files found under {input_dir}")

    shapes: Counter[str] = Counter()
    orientations: Counter[str] = Counter()
    spacings: Counter[str] = Counter()
    determinant_signs: Counter[str] = Counter()
    affine_hashes: Counter[str] = Counter()
    invalid: list[dict[str, object]] = []

    for path in files:
        image = nib.load(str(path))
        affine = np.asarray(image.affine, dtype=np.float64)
        shape = tuple(int(value) for value in image.shape[:3])
        spacing = tuple(round(float(value), 6) for value in image.header.get_zooms()[:3])
        orientation = "".join(str(value) for value in nib.aff2axcodes(affine))
        determinant = float(np.linalg.det(affine[:3, :3]))
        affine_key = hashlib.sha256(np.round(affine, decimals=6).tobytes()).hexdigest()[:16]
        shapes[str(shape)] += 1
        spacings[str(spacing)] += 1
        orientations[orientation] += 1
        determinant_signs["positive" if determinant > 0 else "negative" if determinant < 0 else "zero"] += 1
        affine_hashes[affine_key] += 1
        reasons: list[str] = []
        if shape != (128, 128, 128):
            reasons.append("shape")
        if not np.allclose(spacing, (1.5, 1.5, 1.5), atol=1e-5):
            reasons.append("spacing")
        if not np.isfinite(affine).all() or abs(determinant) < 1e-8:
            reasons.append("affine")
        if reasons:
            invalid.append({"sample_hash": path.name.removesuffix(".nii.gz"), "reasons": reasons})

    axis_consistent = len(orientations) == 1 and "zero" not in determinant_signs
    affine_consistent = len(affine_hashes) == 1
    report = {
        "status": "PASS" if axis_consistent and affine_consistent and not invalid else "REVIEW_REQUIRED",
        "input_scope": "data/preprocessed_mri_128/*.nii.gz",
        "files": len(files),
        "shape_counts": dict(shapes),
        "spacing_counts_mm": dict(spacings),
        "orientation_counts": dict(orientations),
        "determinant_sign_counts": dict(determinant_signs),
        "unique_affine_count": len(affine_hashes),
        "dominant_affine_count": max(affine_hashes.values()),
        "axis_consistent": axis_consistent,
        "affine_consistent": affine_consistent,
        "invalid_count": len(invalid),
        "invalid_first_20": invalid[:20],
        "contains_raw_paths": False,
        "contains_raw_identifiers": False,
    }
    atomic_write_json(PROJECT_ROOT / args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
