"""Label-free sensitivity check for a standard shrink-N4 acceleration.

The full-resolution vendor-style N4 route is retained as the reference.  This
script evaluates whether estimating the N4 bias field on a 4x shrunken image
and applying that field at full resolution materially changes the final
96-cubed image or frozen BrainIAC embedding.  It never reads labels, PET, split
roles, or outer-test predictions and cannot authorize itself for production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import preprocess_brainiac_raw_t1 as primary


REPORT_ROOT = ROOT / "results" / "method_x_registry" / "brainiac_r1"
SENSITIVITY_BASE = ROOT / "data" / "derived" / "method_x_brainiac" / "sensitivity"
REFERENCE_SMOKE = (
    ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "brainiac_preprocessing_smoke.json"
)


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _shrink_n4(
    source: Any, sitk: Any, *, factor: int, use_otsu_mask: bool
) -> Any:
    if factor < 2:
        raise RuntimeError("BRAINIAC_N4_SENSITIVITY_SHRINK_INVALID")
    shrunk = sitk.Shrink(source, [factor] * source.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    if use_otsu_mask:
        mask = sitk.OtsuThreshold(source, 0, 1, 200)
        shrunk_mask = sitk.Shrink(mask, [factor] * source.GetDimension())
        corrector.Execute(shrunk, shrunk_mask)
    else:
        corrector.Execute(shrunk)
    log_bias = corrector.GetLogBiasFieldAsImage(source)
    corrected = source / sitk.Exp(log_bias)
    return sitk.Cast(corrected, sitk.sitkFloat32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64, copy=False).ravel()
    right64 = right.astype(np.float64, copy=False).ravel()
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denominator <= 0:
        raise RuntimeError("BRAINIAC_N4_SENSITIVITY_ZERO_NORM")
    return float(np.dot(left64, right64) / denominator)


def _load_encoder(device: str) -> Any:
    import torch
    from scripts.research.verify_brainiac_official_runtime import (
        CHECKPOINT,
        _load_vendor_class,
    )

    model = _load_vendor_class()(str(CHECKPOINT)).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _embedding(model: Any, array: np.ndarray, device: str) -> np.ndarray:
    import torch

    tensor = torch.as_tensor(array[None], dtype=torch.float32, device=device)
    with torch.inference_mode():
        output = model(tensor)
    result = output.detach().cpu().numpy()[0]
    if result.shape != (768,) or not np.isfinite(result).all():
        raise RuntimeError("BRAINIAC_N4_SENSITIVITY_EMBEDDING_INVALID")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shrink-factor", type=int, choices=(2, 4), default=4)
    parser.add_argument("--mask", choices=("otsu", "none"), default="otsu")
    args = parser.parse_args()
    use_otsu_mask = args.mask == "otsu"
    candidate_slug = f"n4_shrink{args.shrink_factor}_{args.mask}"
    report_path = REPORT_ROOT / f"brainiac_{candidate_slug}_sensitivity.json"
    sensitivity_root = SENSITIVITY_BASE / candidate_slug
    primary._assert_no_cnn_writer()
    if not REFERENCE_SMOKE.is_file():
        raise RuntimeError("BRAINIAC_N4_REFERENCE_SMOKE_MISSING")
    smoke = json.loads(REFERENCE_SMOKE.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or int(smoke.get("rows", 0)) < 3:
        raise RuntimeError("BRAINIAC_N4_REFERENCE_SMOKE_INSUFFICIENT")
    selected_hashes = [str(value) for value in smoke["sample_hashes"][:3]]
    items, fixed, _, _ = primary.preflight()
    by_hash = {item.sample_hash: item for item in items}
    selected = [by_hash[value] for value in selected_hashes]

    import nibabel as nib
    import SimpleITK as sitk
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_encoder(device)
    records: list[dict[str, Any]] = []
    sensitivity_root.mkdir(parents=True, exist_ok=True)
    for item in selected:
        started = time.monotonic()
        work = sensitivity_root / item.sample_hash
        work.mkdir(parents=True, exist_ok=True)
        n4_path = work / "n4_shrink4.nii.gz"
        registered_path = work / f"{item.sample_hash}_0000.nii.gz"
        brain_path = work / "brain.nii.gz"
        candidate_path = work / "candidate_96.nii.gz"

        source = sitk.ReadImage(str(item.source_path), sitk.sitkFloat32)
        n4_started = time.monotonic()
        corrected = _shrink_n4(
            source,
            sitk,
            factor=args.shrink_factor,
            use_otsu_mask=use_otsu_mask,
        )
        primary._write_sitk_atomic(corrected, n4_path, sitk)
        n4_seconds = time.monotonic() - n4_started
        rigid, metric, _ = primary._registration(fixed, corrected, sitk)
        registered = sitk.Resample(
            corrected,
            fixed,
            rigid,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        primary._write_sitk_atomic(registered, registered_path, sitk)
        mask_path = primary._run_hd_bet(registered_path, brain_path, device=device)
        candidate, candidate_affine = primary._official_96_transform(brain_path)
        primary._write_nifti_atomic(candidate, candidate_affine, candidate_path)

        reference_path = primary.OUTPUT_ROOT / f"{item.sample_hash}.nii.gz"
        if not reference_path.is_file():
            raise RuntimeError("BRAINIAC_N4_REFERENCE_OUTPUT_MISSING")
        reference = nib.load(str(reference_path)).get_fdata(dtype=np.float32)[None]
        reference_embedding = _embedding(model, reference, device)
        candidate_embedding = _embedding(model, candidate, device)
        denominator = float(np.mean(np.abs(reference))) or 1.0
        image_relative_l1 = float(np.mean(np.abs(candidate - reference)) / denominator)
        embedding_relative_l2 = float(
            np.linalg.norm(candidate_embedding - reference_embedding)
            / (np.linalg.norm(reference_embedding) or 1.0)
        )
        record = {
            "sample_hash": item.sample_hash,
            "candidate_output_sha256": primary._sha256_file(candidate_path),
            "candidate_mask_sha256": primary._sha256_file(mask_path),
            "candidate_n4_sha256": primary._sha256_file(n4_path),
            "reference_output_sha256": primary._sha256_file(reference_path),
            "n4_seconds": round(n4_seconds, 3),
            "total_seconds": round(time.monotonic() - started, 3),
            "registration_metric": metric,
            "image_cosine": _cosine(candidate, reference),
            "image_relative_l1": image_relative_l1,
            "embedding_cosine": _cosine(candidate_embedding, reference_embedding),
            "embedding_relative_l2": embedding_relative_l2,
            "finite": True,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    embedding_cosines = [float(record["embedding_cosine"]) for record in records]
    image_relative_l1s = [float(record["image_relative_l1"]) for record in records]
    acceptance = {
        "minimum_embedding_cosine": 0.999,
        "maximum_image_relative_l1": 0.08,
        "all_finite": True,
        "minimum_samples": 3,
    }
    passed = (
        len(records) >= acceptance["minimum_samples"]
        and min(embedding_cosines) >= acceptance["minimum_embedding_cosine"]
        and max(image_relative_l1s) <= acceptance["maximum_image_relative_l1"]
        and all(bool(record["finite"]) for record in records)
    )
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if passed else "REJECTED",
        "candidate": (
            f"SHRINK{args.shrink_factor}_N4_"
            f"{'WITH_OTSU_MASK' if use_otsu_mask else 'WITHOUT_EXPLICIT_MASK'}_"
            "FULL_RES_BIAS_APPLICATION"
        ),
        "reference": "VENDOR_STYLE_FULL_RESOLUTION_N4_DEFAULTS",
        "selection_data": "LABEL_FREE_THREE_SAMPLE_PREPROCESSING_SMOKE",
        "labels_read": False,
        "pet_read": False,
        "outer_test_touched": False,
        "records": records,
        "acceptance": acceptance,
        "summary": {
            "minimum_embedding_cosine": min(embedding_cosines),
            "maximum_image_relative_l1": max(image_relative_l1s),
            "median_n4_seconds": float(np.median([record["n4_seconds"] for record in records])),
        },
        "production_authorized": passed,
        "claim_ceiling": "PREPROCESSING_ENGINEERING_SENSITIVITY_ONLY",
    }
    body["integrity_sha256"] = _canonical_sha(body)
    primary._atomic_write_json(report_path, body)
    print(json.dumps(body, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
