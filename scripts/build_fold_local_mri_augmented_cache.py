#!/usr/bin/env python3
"""Build a QC-gated, fold-local materialized MRI augmentation cache.

The source cache contains deterministic preprocessed T1 MRI.  This builder
never augments an outer-test, locked-holdout, or inner-validation parent.  It
keeps a complete parent lineage for every materialized derivative so the
training loader can enforce the same boundary again at run time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json, atomic_write_text


AUGMENTATION_PROFILE = "mri_128_signed_zscore_adaptive_background_v3"
REQUIRED_BASE_DATASETS = ("image", "label", "tabular", "patient_hash", "sample_hash", "status")
REQUIRED_OUTPUT_DATASETS = REQUIRED_BASE_DATASETS + (
    "source",
    "source_sample_hash",
    "source_patient_hash",
    "source_base_index",
    "source_fold",
    "source_train_only",
    "is_synthetic",
    "augmentation_seed",
    "augmentation_parameters",
)


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_split_hashes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row.get("sample_hash", "").strip() for row in csv.DictReader(handle)]
    hashes = [value for value in rows if value]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError(f"Duplicate sample_hash values in split: {path}")
    return hashes


def discover_folds(split_dir: Path, requested_folds: int | None) -> list[int]:
    train_pattern = re.compile(r"^fold_(\d+)_train\.csv$")
    test_pattern = re.compile(r"^fold_(\d+)_test\.csv$")
    train_ids = {int(match.group(1)) for path in split_dir.glob("fold_*_train.csv") if (match := train_pattern.match(path.name))}
    test_ids = {int(match.group(1)) for path in split_dir.glob("fold_*_test.csv") if (match := test_pattern.match(path.name))}
    if not train_ids or train_ids != test_ids:
        raise RuntimeError("Split directory must contain matching fold_<n>_train.csv and fold_<n>_test.csv files.")
    summary_path = split_dir / "split_summary.json"
    summary_folds: int | None = None
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        value = payload.get("folds")
        if value is not None:
            summary_folds = int(value)
    fold_ids = sorted(train_ids)
    expected = int(requested_folds) if requested_folds is not None else summary_folds
    if expected is not None and fold_ids != list(range(expected)):
        raise RuntimeError(f"Fold contract mismatch: discovered {fold_ids}, expected range(0, {expected}).")
    if summary_folds is not None and len(fold_ids) != summary_folds:
        raise RuntimeError(f"split_summary.json declares {summary_folds} folds, but files provide {len(fold_ids)}.")
    return fold_ids


def file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def cache_manifest_digest(path: Path) -> str:
    """Digest identity/QC metadata, not multi-gigabyte image payloads."""

    import h5py

    digest = hashlib.sha256()
    with h5py.File(path, "r") as h5:
        for key in sorted(h5.attrs):
            digest.update(str(key).encode("utf-8"))
            digest.update(repr(h5.attrs[key]).encode("utf-8"))
        for name in ("sample_hash", "patient_hash", "status", "label"):
            if name not in h5:
                continue
            digest.update(name.encode("utf-8"))
            for value in h5[name][:]:
                digest.update(decode(value).encode("utf-8"))
                digest.update(b"\0")
    return digest.hexdigest()


def read_qc_pass(qc_dir: Path | None, sample_hash: str) -> bool:
    if qc_dir is None:
        return True
    path = qc_dir / f"{sample_hash}.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        str(payload.get("qc_status", "")).upper() == "PASS"
        and payload.get("training_cache_allowed") is True
        and payload.get("output_contract_valid") is True
        and list(payload.get("output_shape", [])) == [128, 128, 128]
        and payload.get("contains_pet_input") is False
        and payload.get("contains_raw_identifiers") is False
    )


def inspect_base_cache(real_cache: Path, qc_dir: Path | None) -> dict[str, Any]:
    import h5py

    with h5py.File(real_cache, "r") as h5:
        missing = [name for name in REQUIRED_BASE_DATASETS if name not in h5]
        if missing:
            raise RuntimeError(f"Base cache is missing required datasets: {missing}")
        image_shape = tuple(int(value) for value in h5["image"].shape[1:])
        if image_shape != (1, 128, 128, 128):
            raise RuntimeError(f"Expected (1, 128, 128, 128), found {image_shape}.")
        hashes = [decode(value) for value in h5["sample_hash"][:]]
        patients = [decode(value) for value in h5["patient_hash"][:]]
        statuses = [decode(value) for value in h5["status"][:]]
        labels = np.asarray(h5["label"][:], dtype=np.int8)
        attrs = {str(key): h5.attrs[key] for key in h5.attrs}
        tabular_width = int(h5["tabular"].shape[1])
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Base cache has duplicate sample hashes.")
    if attrs.get("resolution") != 128 or bool(attrs.get("contains_pet_input", False)):
        raise RuntimeError("Base cache violates the 128^3 PET-free Method X contract.")
    eligible_indices = [
        idx
        for idx, (sample_hash, status) in enumerate(zip(hashes, statuses, strict=True))
        if status == "OK" and read_qc_pass(qc_dir, sample_hash)
    ]
    if not eligible_indices:
        raise RuntimeError("No QC-eligible MRI rows were found.")
    return {
        "n_rows": len(hashes),
        "image_shape": image_shape,
        "hashes": hashes,
        "patients": patients,
        "statuses": statuses,
        "labels": labels,
        "attrs": attrs,
        "tabular_width": tabular_width,
        "eligible_indices": eligible_indices,
        "eligible_hashes": {hashes[idx] for idx in eligible_indices},
        "hash_to_index": {sample_hash: idx for idx, sample_hash in enumerate(hashes)},
        "manifest_digest": cache_manifest_digest(real_cache),
    }


def inner_fit_hashes(
    outer_train_hashes: list[str],
    eligible_hashes: set[str],
    *,
    seed: int,
    fold: int,
    val_fraction: float,
) -> tuple[list[str], list[str]]:
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("inner validation fraction must be in (0, 1).")
    real_train = [sample_hash for sample_hash in outer_train_hashes if sample_hash in eligible_hashes]
    if len(real_train) < 2:
        raise RuntimeError(f"Fold {fold} has fewer than two QC-eligible real training rows.")
    random.Random(int(seed) + int(fold)).shuffle(real_train)
    val_n = max(int(len(real_train) * float(val_fraction)), 1)
    val_n = min(val_n, len(real_train) - 1)
    return real_train[val_n:], real_train[:val_n]


def derive_row_seed(seed: int, fold: int, sample_hash: str, ordinal: int) -> int:
    payload = f"{AUGMENTATION_PROFILE}|{seed}|{fold}|{sample_hash}|{ordinal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False)


def augment_image(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a conservative signed-zscore MRI perturbation with zero background.

    There are deliberately no flips, gamma/power transforms, global offsets, or
    global noise.  The HDF5 cache has no affine-axis contract and its background
    is exactly zero, so every intensity operation is constrained to the brain
    mask derived from the source image.
    """

    from scipy.ndimage import affine_transform
    from scipy.spatial.transform import Rotation

    array = np.asarray(image, dtype=np.float32)
    if array.shape != (1, 128, 128, 128):
        raise ValueError(f"Expected (1,128,128,128), received {array.shape}.")
    source = np.nan_to_num(array[0], nan=0.0, posinf=0.0, neginf=0.0)
    source_mask = source != 0.0
    if int(source_mask.sum()) < 1000:
        raise ValueError("Source image has an implausibly small nonzero brain mask.")

    zero_fraction = float(np.mean(~source_mask))
    has_explicit_zero_background = zero_fraction >= 0.05
    if not has_explicit_zero_background:
        # Most current QC-pass volumes are full-field z-scores and carry no
        # recoverable brain mask in HDF5.  For them, only smooth multiplicative
        # intensity perturbations are defensible; geometric/additive operations
        # would manufacture a background boundary that is absent in the source.
        source_mask = np.ones_like(source_mask, dtype=bool)

    apply_affine = bool(has_explicit_zero_background and rng.random() < 0.65)
    angles = np.zeros(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    scale = 1.0
    transformed = source.copy()
    mask = source_mask.copy()
    if apply_affine:
        angles = rng.uniform(-3.0, 3.0, size=3)
        translation = rng.uniform(-1.5, 1.5, size=3)
        scale = float(rng.uniform(0.98, 1.02))
        rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        matrix = rotation.T / scale
        centre = (np.asarray(source.shape, dtype=np.float64) - 1.0) / 2.0
        offset = centre - matrix @ (centre + translation)
        transformed = affine_transform(
            source,
            matrix=matrix,
            offset=offset,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32, copy=False)
        mask = affine_transform(
            source_mask.astype(np.uint8),
            matrix=matrix,
            offset=offset,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(bool, copy=False)

    mask_ratio = float(mask.sum()) / float(source_mask.sum())
    if not 0.90 <= mask_ratio <= 1.10:
        raise ValueError(f"Affine changed brain-mask volume by an unsafe ratio: {mask_ratio:.3f}")

    values = transformed[mask]
    value_std = float(np.std(values, dtype=np.float64))
    contrast = float(rng.uniform(0.97, 1.03))
    offset_intensity = float(rng.uniform(-0.025, 0.025)) if has_explicit_zero_background else 0.0
    transformed[mask] = transformed[mask] * contrast + offset_intensity

    coeffs = rng.uniform(-0.025, 0.025, size=3).astype(np.float32)
    axis = np.linspace(-1.0, 1.0, num=128, dtype=np.float32)
    bias = 1.0 + coeffs[0] * axis[:, None, None] + coeffs[1] * axis[None, :, None] + coeffs[2] * axis[None, None, :]
    transformed[mask] *= bias[mask]

    noise_std = max(value_std, 0.1) * float(rng.uniform(0.003, 0.010)) if has_explicit_zero_background else 0.0
    if noise_std > 0.0:
        transformed[mask] += rng.normal(0.0, noise_std, size=int(mask.sum())).astype(np.float32)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    transformed[~mask] = 0.0
    output = transformed[None].astype(np.float16)
    if not np.isfinite(output).all() or np.any(output[:, ~mask] != 0.0):
        raise ValueError("Augmentation violated finite/zero-background constraints.")
    return output, {
        "profile": AUGMENTATION_PROFILE,
        "mask_mode": "exact_zero_background" if has_explicit_zero_background else "full_field_no_explicit_background",
        "source_zero_fraction": round(zero_fraction, 8),
        "affine_applied": apply_affine,
        "rotation_degrees": [round(float(value), 5) for value in angles],
        "translation_vox": [round(float(value), 5) for value in translation],
        "scale": round(scale, 6),
        "contrast": round(contrast, 6),
        "offset_zscore": round(offset_intensity, 6),
        "bias_linear_coefficients": [round(float(value), 6) for value in coeffs],
        "noise_std": round(noise_std, 8),
        "source_mask_voxels": int(source_mask.sum()),
        "output_mask_voxels": int(mask.sum()),
    }


def build_plan(
    info: dict[str, Any],
    split_dir: Path,
    *,
    folds: list[int],
    training_seed: int,
    inner_val_fraction: float,
    augments_per_fit_real: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, set[str]]:
    if augments_per_fit_real < 1:
        raise ValueError("augments_per_fit_real must be >= 1.")
    split_paths: list[Path] = []
    external_path = split_dir / "external_holdout.csv"
    if external_path.exists():
        split_paths.append(external_path)
    for fold in folds:
        split_paths.extend([split_dir / f"fold_{fold}_train.csv", split_dir / f"fold_{fold}_test.csv"])
    split_digest = file_digest(split_paths)
    external_hashes = set(read_split_hashes(external_path)) if external_path.exists() else set()
    plan: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    eligible_hashes = set(info["eligible_hashes"])
    for fold in folds:
        train_hashes = read_split_hashes(split_dir / f"fold_{fold}_train.csv")
        test_hashes = set(read_split_hashes(split_dir / f"fold_{fold}_test.csv"))
        if set(train_hashes).intersection(test_hashes):
            raise RuntimeError(f"Fold {fold} has outer train/test overlap.")
        fit_hashes, inner_val_hashes = inner_fit_hashes(
            train_hashes,
            eligible_hashes,
            seed=training_seed,
            fold=fold,
            val_fraction=inner_val_fraction,
        )
        if set(fit_hashes).intersection(set(inner_val_hashes)):
            raise RuntimeError(f"Fold {fold} inner fit/validation overlap.")
        if set(fit_hashes).intersection(test_hashes) or set(fit_hashes).intersection(external_hashes):
            raise RuntimeError(f"Fold {fold} attempted to augment an outer-test or locked-holdout parent.")
        label_counts = {"0": 0, "1": 0}
        for source_hash in fit_hashes:
            source_index = int(info["hash_to_index"][source_hash])
            for ordinal in range(augments_per_fit_real):
                plan.append(
                    {
                        "fold": fold,
                        "source_hash": source_hash,
                        "source_index": source_index,
                        "ordinal": ordinal,
                        "seed": derive_row_seed(training_seed, fold, source_hash, ordinal),
                    }
                )
                label_counts[str(int(info["labels"][source_index]))] += 1
        fold_reports.append(
            {
                "fold": fold,
                "outer_train_eligible_real_rows": len([value for value in train_hashes if value in eligible_hashes]),
                "outer_test_eligible_real_rows": len([value for value in test_hashes if value in eligible_hashes]),
                "inner_fit_parent_rows": len(fit_hashes),
                "inner_validation_real_rows": len(inner_val_hashes),
                "materialized_augmented_rows": len(fit_hashes) * augments_per_fit_real,
                "augmented_label_counts": label_counts,
                "inner_fit_parent_hashes": fit_hashes,
            }
        )
    return plan, fold_reports, split_digest, external_hashes


def require_free_space(directory: Path, row_count: int, image_shape: tuple[int, ...]) -> dict[str, int]:
    raw_bytes = int(row_count * np.prod(image_shape, dtype=np.int64) * np.dtype(np.float16).itemsize)
    required = int(raw_bytes * 1.08)
    free = int(shutil.disk_usage(directory).free)
    if free < required:
        raise RuntimeError(f"Insufficient free space: need at least {required} bytes, have {free} bytes.")
    return {"raw_image_bytes_upper_bound": raw_bytes, "required_free_bytes": required, "free_bytes_before_build": free}


def create_output_datasets(out, *, total_rows: int, image_shape: tuple[int, ...], tabular_width: int) -> None:
    import h5py

    string_dtype = h5py.string_dtype(encoding="utf-8")
    out.create_dataset(
        "image",
        shape=(total_rows, *image_shape),
        dtype="float16",
        compression="gzip",
        compression_opts=4,
        chunks=(1, *image_shape),
    )
    out.create_dataset("label", shape=(total_rows,), dtype="int8")
    out.create_dataset("tabular", shape=(total_rows, tabular_width), dtype="float32", compression="gzip", compression_opts=4)
    for name in ("patient_hash", "sample_hash", "status", "source", "source_sample_hash", "source_patient_hash", "augmentation_parameters"):
        out.create_dataset(name, shape=(total_rows,), dtype=string_dtype)
    out.create_dataset("source_base_index", shape=(total_rows,), dtype="int32")
    out.create_dataset("source_fold", shape=(total_rows,), dtype="int16")
    out.create_dataset("source_train_only", shape=(total_rows,), dtype="bool")
    out.create_dataset("is_synthetic", shape=(total_rows,), dtype="bool")
    out.create_dataset("augmentation_seed", shape=(total_rows,), dtype="uint64")


def validate_materialized_cache(
    path: Path,
    *,
    info: dict[str, Any],
    plan: list[dict[str, Any]],
    fold_reports: list[dict[str, Any]],
    external_hashes: set[str],
) -> dict[str, Any]:
    import h5py

    expected_by_fold = {int(row["fold"]): set(row["inner_fit_parent_hashes"]) for row in fold_reports}
    expected_real = [info["hashes"][idx] for idx in info["eligible_indices"]]
    checked_images = 0
    with h5py.File(path, "r") as h5:
        missing = [name for name in REQUIRED_OUTPUT_DATASETS if name not in h5]
        if missing:
            raise RuntimeError(f"Output cache missing datasets: {missing}")
        total = int(h5["image"].shape[0])
        if total != len(expected_real) + len(plan):
            raise RuntimeError("Output row count does not match real + planned augmented rows.")
        if tuple(h5["image"].shape[1:]) != tuple(info["image_shape"]):
            raise RuntimeError("Output image shape drifted from the base cache.")
        hashes = [decode(value) for value in h5["sample_hash"][:]]
        patients = [decode(value) for value in h5["patient_hash"][:]]
        statuses = [decode(value) for value in h5["status"][:]]
        sources = [decode(value) for value in h5["source"][:]]
        parent_hashes = [decode(value) for value in h5["source_sample_hash"][:]]
        parent_patients = [decode(value) for value in h5["source_patient_hash"][:]]
        folds = [int(value) for value in h5["source_fold"][:]]
        train_only = [bool(value) for value in h5["source_train_only"][:]]
        synthetic = [bool(value) for value in h5["is_synthetic"][:]]
        labels = [int(value) for value in h5["label"][:]]
        parameters = [decode(value) for value in h5["augmentation_parameters"][:]]
        if hashes[: len(expected_real)] != expected_real:
            raise RuntimeError("Real rows were not preserved in base-cache order.")
        if any(status != "OK" for status in statuses):
            raise RuntimeError("Final training cache contains non-OK rows.")
        real_lookup = {sample_hash: index for index, sample_hash in enumerate(expected_real)}
        for idx in range(total):
            image = np.asarray(h5["image"][idx], dtype=np.float32)
            checked_images += 1
            if not np.isfinite(image).all() or float(np.std(image, dtype=np.float64)) < 1e-6:
                raise RuntimeError(f"Image validation failed at output row {idx}.")
            if checked_images % 100 == 0 or checked_images == total:
                print(f"Validated output images: {checked_images}/{total}", flush=True)
            if not synthetic[idx]:
                if sources[idx] != "real" or folds[idx] != -1 or train_only[idx] or parent_hashes[idx]:
                    raise RuntimeError(f"Invalid real-row lineage at {idx}.")
                continue
            parent = parent_hashes[idx]
            if not train_only[idx] or not parent or parent not in real_lookup:
                raise RuntimeError(f"Missing or invalid synthetic lineage at {idx}.")
            if parent in external_hashes or parent not in expected_by_fold.get(folds[idx], set()):
                raise RuntimeError(f"Synthetic parent escapes its allowed fold at {idx}.")
            transform_parameters = json.loads(parameters[idx])
            if transform_parameters.get("mask_mode") == "exact_zero_background" and float(np.mean(image == 0.0)) < 0.05:
                raise RuntimeError(f"Masked synthetic row lost its zero background at {idx}.")
            parent_idx = real_lookup[parent]
            if labels[idx] != labels[parent_idx] or patients[idx] != patients[parent_idx] or parent_patients[idx] != patients[parent_idx]:
                raise RuntimeError(f"Synthetic label/patient lineage mismatch at {idx}.")
    return {
        "status": "PASS",
        "rows": len(expected_real) + len(plan),
        "real_rows": len(expected_real),
        "supplementary_rows": len(plan),
        "checked_images": checked_images,
    }


def build_materialized_cache(
    *,
    real_cache: Path,
    split_dir: Path,
    out_path: Path,
    qc_dir: Path | None,
    requested_folds: int | None,
    training_seed: int,
    inner_val_fraction: float,
    augments_per_fit_real: int,
) -> dict[str, Any]:
    import h5py

    if real_cache.resolve() == out_path.resolve():
        raise RuntimeError("Build into a distinct staging path; publish atomically only after validation.")
    if out_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing staging cache: {out_path}")
    info = inspect_base_cache(real_cache, qc_dir)
    folds = discover_folds(split_dir, requested_folds)
    plan, fold_reports, split_digest, external_hashes = build_plan(
        info,
        split_dir,
        folds=folds,
        training_seed=training_seed,
        inner_val_fraction=inner_val_fraction,
        augments_per_fit_real=augments_per_fit_real,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = len(info["eligible_indices"]) + len(plan)
    space = require_free_space(out_path.parent, total_rows, tuple(info["image_shape"]))
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with h5py.File(real_cache, "r") as source, h5py.File(out_path, "x") as out:
        create_output_datasets(out, total_rows=total_rows, image_shape=tuple(info["image_shape"]), tabular_width=int(info["tabular_width"]))
        for name, value in info["attrs"].items():
            if name not in {"source_mode", "resolution"}:
                out.attrs[name] = value
        out.attrs["resolution"] = 128
        out.attrs["source_mode"] = "preprocessed_mri_plus_foldlocal_materialized_augmentation"
        out.attrs["cache_role"] = "final_training_candidate"
        out.attrs["contains_supplementary"] = True
        out.attrs["contains_synthetic"] = True
        out.attrs["source_scope"] = "outer_train_and_inner_fit_only"
        out.attrs["augmentation_profile"] = AUGMENTATION_PROFILE
        out.attrs["source_cache_manifest_sha256"] = info["manifest_digest"]
        out.attrs["split_contract_sha256"] = split_digest
        out.attrs["split_folds"] = len(folds)
        out.attrs["inner_validation_fraction"] = float(inner_val_fraction)
        out.attrs["training_seed"] = int(training_seed)
        out.attrs["augments_per_fit_real"] = int(augments_per_fit_real)
        out.attrs["n_real_eligible"] = len(info["eligible_indices"])
        out.attrs["n_real_excluded_qc"] = int(info["n_rows"] - len(info["eligible_indices"]))
        out.attrs["created_at_utc"] = started

        for dst_idx, src_idx in enumerate(info["eligible_indices"]):
            image = np.asarray(source["image"][src_idx], dtype=np.float16)
            if not np.isfinite(image).all() or float(np.std(image, dtype=np.float64)) < 1e-6:
                raise RuntimeError(f"Base image {src_idx} violates finite/nonconstant validation.")
            out["image"][dst_idx] = image
            out["label"][dst_idx] = source["label"][src_idx]
            out["tabular"][dst_idx] = source["tabular"][src_idx]
            out["patient_hash"][dst_idx] = info["patients"][src_idx]
            out["sample_hash"][dst_idx] = info["hashes"][src_idx]
            out["status"][dst_idx] = "OK"
            out["source"][dst_idx] = "real"
            out["source_sample_hash"][dst_idx] = ""
            out["source_patient_hash"][dst_idx] = ""
            out["source_base_index"][dst_idx] = src_idx
            out["source_fold"][dst_idx] = -1
            out["source_train_only"][dst_idx] = False
            out["is_synthetic"][dst_idx] = False
            out["augmentation_seed"][dst_idx] = 0
            out["augmentation_parameters"][dst_idx] = ""
            if (dst_idx + 1) % 100 == 0 or dst_idx + 1 == len(info["eligible_indices"]):
                print(f"Copied eligible real rows: {dst_idx + 1}/{len(info['eligible_indices'])}", flush=True)

        offset = len(info["eligible_indices"])
        for ordinal, row in enumerate(plan, start=1):
            src_idx = int(row["source_index"])
            row_seed = int(row["seed"])
            augmented, parameters = augment_image(source["image"][src_idx], np.random.default_rng(row_seed))
            dst_idx = offset + ordinal - 1
            out["image"][dst_idx] = augmented
            out["label"][dst_idx] = source["label"][src_idx]
            out["tabular"][dst_idx] = source["tabular"][src_idx]
            out["patient_hash"][dst_idx] = info["patients"][src_idx]
            out["sample_hash"][dst_idx] = f"aug_f{row['fold']}_{row['source_hash']}_{row['ordinal']:02d}"
            out["status"][dst_idx] = "OK"
            out["source"][dst_idx] = "materialized_masked_augmentation"
            out["source_sample_hash"][dst_idx] = row["source_hash"]
            out["source_patient_hash"][dst_idx] = info["patients"][src_idx]
            out["source_base_index"][dst_idx] = src_idx
            out["source_fold"][dst_idx] = int(row["fold"])
            out["source_train_only"][dst_idx] = True
            out["is_synthetic"][dst_idx] = True
            out["augmentation_seed"][dst_idx] = row_seed
            out["augmentation_parameters"][dst_idx] = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
            if ordinal % 50 == 0 or ordinal == len(plan):
                print(f"Generated materialized rows: {ordinal}/{len(plan)}", flush=True)

    validation = validate_materialized_cache(
        out_path,
        info=info,
        plan=plan,
        fold_reports=fold_reports,
        external_hashes=external_hashes,
    )
    return {
        "status": "VALIDATED_STAGED",
        "stage_path": project_relative(out_path),
        "real_cache": project_relative(real_cache),
        "source_cache_manifest_sha256": info["manifest_digest"],
        "split_contract_sha256": split_digest,
        "split_folds": len(folds),
        "eligible_real_rows": len(info["eligible_indices"]),
        "excluded_qc_rows": int(info["n_rows"] - len(info["eligible_indices"])),
        "materialized_augmented_rows": len(plan),
        "total_rows": total_rows,
        "augments_per_fit_real": int(augments_per_fit_real),
        "inner_validation_fraction": float(inner_val_fraction),
        "training_seed": int(training_seed),
        "folds": [{key: value for key, value in row.items() if key != "inner_fit_parent_hashes"} for row in fold_reports],
        "validation": validation,
        "space": space,
    }


def validate_existing_stage(
    *,
    real_cache: Path,
    split_dir: Path,
    stage_path: Path,
    qc_dir: Path | None,
    requested_folds: int | None,
    training_seed: int,
    inner_val_fraction: float,
    augments_per_fit_real: int,
) -> dict[str, Any]:
    """Reconstruct the deterministic plan and validate an interrupted stage."""

    import h5py

    info = inspect_base_cache(real_cache, qc_dir)
    folds = discover_folds(split_dir, requested_folds)
    plan, fold_reports, split_digest, external_hashes = build_plan(
        info,
        split_dir,
        folds=folds,
        training_seed=training_seed,
        inner_val_fraction=inner_val_fraction,
        augments_per_fit_real=augments_per_fit_real,
    )
    with h5py.File(stage_path, "r") as h5:
        if decode(h5.attrs.get("source_cache_manifest_sha256", "")) != info["manifest_digest"]:
            raise RuntimeError("Existing stage was not built from the current canonical source cache.")
        if decode(h5.attrs.get("split_contract_sha256", "")) != split_digest:
            raise RuntimeError("Existing stage was not built from the current split contract.")
        if decode(h5.attrs.get("augmentation_profile", "")) != AUGMENTATION_PROFILE:
            raise RuntimeError("Existing stage uses a different augmentation profile.")
    validation = validate_materialized_cache(
        stage_path,
        info=info,
        plan=plan,
        fold_reports=fold_reports,
        external_hashes=external_hashes,
    )
    return {
        "status": "VALIDATED_STAGED",
        "stage_path": project_relative(stage_path),
        "real_cache": project_relative(real_cache),
        "source_cache_manifest_sha256": info["manifest_digest"],
        "split_contract_sha256": split_digest,
        "split_folds": len(folds),
        "eligible_real_rows": len(info["eligible_indices"]),
        "excluded_qc_rows": int(info["n_rows"] - len(info["eligible_indices"])),
        "materialized_augmented_rows": len(plan),
        "total_rows": len(info["eligible_indices"]) + len(plan),
        "augments_per_fit_real": int(augments_per_fit_real),
        "inner_validation_fraction": float(inner_val_fraction),
        "training_seed": int(training_seed),
        "folds": [{key: value for key, value in row.items() if key != "inner_fit_parent_hashes"} for row in fold_reports],
        "validation": validation,
        "recovered_complete_stage": True,
    }


@contextmanager
def build_lock(lock_path: Path) -> Iterator[None]:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another augmentation build may be active: {lock_path}") from exc
    try:
        os.write(fd, f"pid={os.getpid()}\nstarted={datetime.now(timezone.utc).isoformat()}\n".encode("utf-8"))
        yield
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)


def publish_canonical(*, real_cache: Path, stage_path: Path, archive_dir: Path, expected_digest: str) -> Path:
    if cache_manifest_digest(real_cache) != expected_digest:
        raise RuntimeError("The source cache changed during augmentation; refusing to publish.")
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = archive_dir / f"{real_cache.stem}_real_{timestamp}_{expected_digest[:12]}{real_cache.suffix}"
    if backup_path.exists():
        raise RuntimeError(f"Archive target already exists: {backup_path}")
    real_cache.replace(backup_path)
    try:
        stage_path.replace(real_cache)
    except Exception:
        backup_path.replace(real_cache)
        raise
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_cache", default="data/cached/t1_volume_cache_128.h5")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--qc_dir", default="data/preprocessed_mri_128/qc")
    parser.add_argument("--out_combined", required=True, help="Distinct staging .h5 path; it is validated before any publish.")
    parser.add_argument("--folds", type=int, default=None, help="Must match split_summary.json when supplied.")
    parser.add_argument("--training_seed", type=int, default=13)
    parser.add_argument("--inner_val_fraction", type=float, default=0.15)
    parser.add_argument("--augments_per_fit_real", type=int, default=1)
    parser.add_argument("--publish_canonical", action="store_true", help="Atomically replace real_cache after validation and archive the real-only source.")
    parser.add_argument("--use_existing_valid_stage", action="store_true", help="Validate and publish a fully written stage left by an interrupted run.")
    parser.add_argument("--archive_dir", default="data/cached/archive")
    args = parser.parse_args()

    real_cache = PROJECT_ROOT / args.real_cache
    split_dir = PROJECT_ROOT / args.splits
    qc_dir = PROJECT_ROOT / args.qc_dir if args.qc_dir else None
    stage_path = PROJECT_ROOT / args.out_combined
    archive_dir = PROJECT_ROOT / args.archive_dir
    if not real_cache.exists() or not split_dir.exists() or (qc_dir is not None and not qc_dir.exists()):
        raise SystemExit("real_cache, splits, and qc_dir must exist before building.")
    lock_path = real_cache.parent / ".t1_volume_cache_128.augmentation.lock"
    with build_lock(lock_path):
        if args.use_existing_valid_stage:
            if not stage_path.exists():
                raise RuntimeError(f"Existing staging cache was not found: {stage_path}")
            summary = validate_existing_stage(
                real_cache=real_cache,
                split_dir=split_dir,
                stage_path=stage_path,
                qc_dir=qc_dir,
                requested_folds=args.folds,
                training_seed=args.training_seed,
                inner_val_fraction=args.inner_val_fraction,
                augments_per_fit_real=args.augments_per_fit_real,
            )
        else:
            summary = build_materialized_cache(
                real_cache=real_cache,
                split_dir=split_dir,
                out_path=stage_path,
                qc_dir=qc_dir,
                requested_folds=args.folds,
                training_seed=args.training_seed,
                inner_val_fraction=args.inner_val_fraction,
                augments_per_fit_real=args.augments_per_fit_real,
            )
        if args.publish_canonical:
            backup = publish_canonical(
                real_cache=real_cache,
                stage_path=stage_path,
                archive_dir=archive_dir,
                expected_digest=str(summary["source_cache_manifest_sha256"]),
            )
            summary["status"] = "PUBLISHED_CANONICAL"
            summary["canonical_cache"] = project_relative(real_cache)
            summary["real_only_backup"] = project_relative(backup)
        legacy_paths = [
            "data/cached/t1_foldlocal_aug_128.h5",
            "data/cached/t1_combined_foldlocal_4000_128.h5",
            "data/cached/t1_combined_128.h5",
        ]
        summary["superseded_artifacts"] = [path for path in legacy_paths if (PROJECT_ROOT / path).exists()]
        summary["corrections_applied"] = [
            "current_five_fold_split_contract",
            "outer_train_and_inner_fit_only_parent_selection",
            "locked_holdout_and_outer_test_parent_exclusion",
            "status_OK_and_qc_PASS_training_admission",
            "signed_zscore_brain_mask_preserving_transforms",
            "no_axis_flips_or_global_background_noise",
            "complete_parent_and_transform_lineage",
        ]
        atomic_write_json(PROJECT_ROOT / "results/tables/fold_local_mri_augmentation_summary.json", summary)
        lines = [
            "# Materialized MRI augmentation summary\n\n",
            f"- status: `{summary['status']}`\n",
            f"- canonical cache: `{summary.get('canonical_cache', summary['stage_path'])}`\n",
            f"- QC-eligible real rows: `{summary['eligible_real_rows']}`\n",
            f"- QC-excluded rows: `{summary['excluded_qc_rows']}`\n",
            f"- materialized train-only rows: `{summary['materialized_augmented_rows']}`\n",
            f"- total rows: `{summary['total_rows']}`\n",
            f"- fold contract: `{summary['split_folds']}`\n",
            "- validation/test/locked holdout: `real-only`\n",
            "- Method X PET input: `False`\n",
        ]
        atomic_write_text(PROJECT_ROOT / "results/tables/fold_local_mri_augmentation_summary.md", "".join(lines))
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
