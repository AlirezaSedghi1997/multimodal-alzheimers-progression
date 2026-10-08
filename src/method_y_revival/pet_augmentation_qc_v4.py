from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import nibabel as nib
import numpy as np

from .atomic_io import atomic_write_json
from .pet_augmentation_v4 import (
    PetSafeAugmentationConfig,
    apply_pet_safe_augmentation,
)


class PetAugmentationQCV4Error(RuntimeError):
    pass


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_mapping(payload: Mapping[str, Any]) -> str:
    import json

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_locked_foreground_mask(
    *, reference_template: str | Path, output_mask: str | Path
) -> dict[str, Any]:
    source = Path(reference_template)
    output = Path(output_mask)
    if not source.is_file():
        raise PetAugmentationQCV4Error("REFERENCE_TEMPLATE_MISSING")
    reference = nib.load(source)
    data = np.asarray(reference.dataobj, dtype=np.float32)
    mask = np.isfinite(data) & (data > 0)
    if mask.ndim != 3 or int(mask.sum()) < 8:
        raise PetAugmentationQCV4Error("REFERENCE_FOREGROUND_MASK_INVALID")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header), temporary)
    os.replace(temporary, output)
    return {
        "reference_template_sha256": _sha256_file(source),
        "foreground_mask_sha256": _sha256_file(output),
        "shape": [int(value) for value in mask.shape],
        "foreground_voxels": int(mask.sum()),
        "foreground_fraction": float(mask.mean()),
        "definition": "FINITE_AND_REFERENCE_TEMPLATE_GREATER_THAN_ZERO",
    }


def select_largest_passing_envelope(
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    passing = [dict(row) for row in candidates if str(row.get("status")) == "PASS"]
    if not passing:
        raise PetAugmentationQCV4Error("NO_GEOMETRIC_ENVELOPE_PASSED")
    return max(
        passing,
        key=lambda row: (
            float(row.get("rotation", 0.0)),
            float(row.get("translation", 0.0)),
            abs(float(row.get("scale_max", 1.0)) - 1.0),
        ),
    )


def _deterministic_subject_sample(
    subjects: Iterable[str], *, branch: str, seed: int, sample_count: int
) -> list[str]:
    ranked = sorted(
        (str(subject) for subject in subjects),
        key=lambda subject: hashlib.sha256(
            f"{seed}|{branch}|{subject}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[: min(int(sample_count), len(ranked))]


def _metric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise PetAugmentationQCV4Error("QC_METRIC_EMPTY_OR_NONFINITE")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def calibrate_geometric_envelope(
    *,
    branch_caches: Mapping[str, str | Path],
    foreground_mask: str | Path,
    output_json: str | Path,
    sample_count_per_branch: int = 30,
    draws_per_subject: int = 1,
    seed: int = 20260714,
) -> dict[str, Any]:
    mask_path = Path(foreground_mask)
    if not mask_path.is_file():
        raise PetAugmentationQCV4Error("FOREGROUND_MASK_MISSING")
    mask = (np.asarray(nib.load(mask_path).dataobj) > 0).astype(np.float32)[None]
    cache_hashes = {branch: _sha256_file(path) for branch, path in branch_caches.items()}
    design = {
        "schema_version": "4.0",
        "method": "OUTCOME_BLIND_PERTURBATION_STABILITY_QC",
        "seed": int(seed),
        "sample_count_per_branch": int(sample_count_per_branch),
        "draws_per_subject": int(draws_per_subject),
        "foreground_mask_sha256": _sha256_file(mask_path),
        "branch_cache_sha256": cache_hashes,
        "implementation_source_sha256": {
            "pet_augmentation_qc_v4.py": _sha256_file(Path(__file__)),
            "pet_augmentation_v4.py": _sha256_file(
                Path(__file__).with_name("pet_augmentation_v4.py")
            ),
        },
        "candidate_envelopes": [
            {"candidate_id": "identity", "rotation": 0.0, "translation": 0.0},
            {"candidate_id": "primary", "rotation": 1.0, "translation": 0.5},
            {"candidate_id": "upper", "rotation": 2.0, "translation": 1.0},
        ],
        "scale": [1.0, 1.0],
        "thresholds": {
            "pearson_p05_min": 0.95,
            "foreground_dice_p05_min": 0.99,
            "normalized_mae_p95_max": 0.35,
            "absolute_mass_ratio_p05_min": 0.95,
            "absolute_mass_ratio_p95_max": 1.05,
        },
        "explicitly_unused": [
            "outcome",
            "event",
            "time_days",
            "label",
            "outer_test_metric",
        ],
    }
    qc_design_sha = _sha256_mapping(design)
    thresholds = design["thresholds"]
    records: list[dict[str, Any]] = []
    sample_digests: dict[str, str] = {}
    for candidate in design["candidate_envelopes"]:
        metrics = {
            "pearson": [],
            "foreground_dice": [],
            "normalized_mae": [],
            "absolute_mass_ratio": [],
        }
        branch_counts: dict[str, int] = {}
        for branch, cache_path_value in branch_caches.items():
            cache_path = Path(cache_path_value)
            if not cache_path.is_file():
                raise PetAugmentationQCV4Error(f"BRANCH_CACHE_MISSING:{branch}")
            with h5py.File(cache_path, "r") as handle:
                selected = _deterministic_subject_sample(
                    handle.keys(),
                    branch=branch,
                    seed=seed,
                    sample_count=sample_count_per_branch,
                )
                branch_counts[branch] = len(selected)
                sample_digests[branch] = hashlib.sha256(
                    "\n".join(selected).encode("utf-8")
                ).hexdigest()
                for subject in selected:
                    original = np.asarray(handle[subject]["image"], dtype=np.float32)
                    if original.shape != mask.shape or not np.isfinite(original).all():
                        raise PetAugmentationQCV4Error(
                            f"QC_SOURCE_IMAGE_INVALID:{branch}:{subject}"
                        )
                    for draw in range(int(draws_per_subject)):
                        config = PetSafeAugmentationConfig(
                            enabled=True,
                            probability=1.0,
                            max_rotation_degrees=float(candidate["rotation"]),
                            max_translation_voxels=float(candidate["translation"]),
                            scale_min=1.0,
                            scale_max=1.0,
                            qc_contract_sha256=qc_design_sha,
                        )
                        augmented = apply_pet_safe_augmentation(
                            image=original,
                            mask=mask,
                            config=config,
                            split_role="outer_train",
                            branch=branch,
                            subject_key=subject,
                            outer_fold=0,
                            inner_fold=None,
                            epoch=draw + 1,
                            base_seed=seed,
                        )
                        transformed = augmented.image.detach().cpu().numpy()
                        transformed_mask = augmented.mask.detach().cpu().numpy() > 0
                        original_mask = mask > 0
                        intersection = transformed_mask & original_mask
                        if int(intersection.sum()) < 8:
                            raise PetAugmentationQCV4Error("QC_INTERSECTION_EMPTY")
                        x = original[intersection].astype(np.float64)
                        y = transformed[intersection].astype(np.float64)
                        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
                            raise PetAugmentationQCV4Error("QC_SIGNAL_VARIANCE_ZERO")
                        metrics["pearson"].append(float(np.corrcoef(x, y)[0, 1]))
                        metrics["foreground_dice"].append(
                            float(
                                2.0
                                * np.logical_and(original_mask, transformed_mask).sum()
                                / max(1, original_mask.sum() + transformed_mask.sum())
                            )
                        )
                        iqr = max(1e-6, float(np.quantile(x, 0.75) - np.quantile(x, 0.25)))
                        metrics["normalized_mae"].append(float(np.mean(np.abs(x - y)) / iqr))
                        original_mass = float(np.abs(original[original_mask]).sum())
                        transformed_mass = float(np.abs(transformed[transformed_mask]).sum())
                        metrics["absolute_mass_ratio"].append(
                            transformed_mass / max(1e-12, original_mass)
                        )
        summary = {name: _metric_summary(values) for name, values in metrics.items()}
        passed = (
            summary["pearson"]["p05"] >= thresholds["pearson_p05_min"]
            and summary["foreground_dice"]["p05"]
            >= thresholds["foreground_dice_p05_min"]
            and summary["normalized_mae"]["p95"]
            <= thresholds["normalized_mae_p95_max"]
            and summary["absolute_mass_ratio"]["p05"]
            >= thresholds["absolute_mass_ratio_p05_min"]
            and summary["absolute_mass_ratio"]["p95"]
            <= thresholds["absolute_mass_ratio_p95_max"]
        )
        records.append(
            {
                **candidate,
                "scale_min": 1.0,
                "scale_max": 1.0,
                "status": "PASS" if passed else "FAIL",
                "metrics": summary,
                "branch_sample_counts": branch_counts,
            }
        )
    selected = select_largest_passing_envelope(records)
    payload = {
        **design,
        "qc_design_sha256": qc_design_sha,
        "sample_subject_set_sha256": sample_digests,
        "candidate_results": records,
        "selected_envelope": {
            "candidate_id": selected["candidate_id"],
            "max_rotation_degrees": float(selected["rotation"]),
            "max_translation_voxels": float(selected["translation"]),
            "scale_min": 1.0,
            "scale_max": 1.0,
        },
        "psf_smoothing_status": "DISABLED_NO_SCANNER_MODEL_PSF_REGISTRY",
        "count_noise_status": "DISABLED_NO_RAW_COUNT_OR_CALIBRATED_NOISE_MODEL",
        "status": "PASS",
        "claim_scope": "AUGMENTATION_STABILITY_ONLY_NOT_QUANTITATIVE_EQUIVALENCE",
    }
    payload["contract_sha256"] = _sha256_mapping(payload)
    atomic_write_json(Path(output_json), payload)
    return payload
