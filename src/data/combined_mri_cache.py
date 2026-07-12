"""Helpers for real + synthetic + simulated MRI caches and CV splits."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

SUPPLEMENTARY_PREFIXES = ("syn_ddpm_", "syn_sim_", "syn_")
METHOD_X_PHENOTYPE_FEATURES: tuple[str, ...] = (
    "age",
    "sex",
    "education_years",
    "apoe4_carrier",
    "mmse_base",
)
FORBIDDEN_PHENOTYPE_SIMILARITY_FEATURES: tuple[str, ...] = (
    "cohort_id",
    "cohort",
    "site",
    "scanner_id",
    "study",
    "study_name",
    "pet",
    "suvr",
    "centiloid",
)


def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalized_feature_name(value: str | bytes) -> str:
    return decode_h5_string(value).strip().lower().replace("-", "_").replace(" ", "_")


def validate_phenotype_feature_names(feature_names: Iterable[str | bytes]) -> tuple[str, ...]:
    """Validate AMUSE-X A2 phenotype features for PET-free, cohort-free similarity.

    The contrastive similarity is allowed to use only stable Method X phenotype
    covariates. Cohort/site/scanner/PET-derived fields must not enter this path.
    """

    normalized = tuple(normalized_feature_name(name) for name in feature_names)
    forbidden = [
        name
        for name in normalized
        if any(token in name for token in FORBIDDEN_PHENOTYPE_SIMILARITY_FEATURES)
    ]
    if forbidden:
        raise ValueError("Forbidden phenotype-similarity feature(s): " + ", ".join(forbidden))
    if normalized != METHOD_X_PHENOTYPE_FEATURES:
        raise ValueError(
            "Method X phenotype features must be exactly "
            f"{METHOD_X_PHENOTYPE_FEATURES}; got {normalized}"
        )
    return normalized


def phenotype_features_from_tabular(tabular):
    """Return the five allowed phenotype features used by AMUSE-X A2.

    The current Method X cache stores tabular columns in the order documented by
    the audit: age, sex, education_years, apoe4_carrier, mmse_base.
    """

    if tabular.shape[-1] < len(METHOD_X_PHENOTYPE_FEATURES):
        raise ValueError(
            f"Expected at least {len(METHOD_X_PHENOTYPE_FEATURES)} tabular columns; got {tabular.shape[-1]}"
        )
    return tabular[..., : len(METHOD_X_PHENOTYPE_FEATURES)]


def sample_source(sample_hash: str, source_value: str | bytes | None = None) -> str:
    if source_value is not None:
        if isinstance(source_value, bytes):
            source_value = source_value.decode("utf-8", errors="replace")
        src = str(source_value).strip().lower()
        if src:
            return src
    h = str(sample_hash)
    if h.startswith("syn_ddpm_"):
        return "synthetic_ddpm"
    if h.startswith("syn_sim_"):
        return "simulated_augment"
    if h.startswith("syn_"):
        return "simulated_augment"
    return "real"


def is_supplementary_sample(sample_hash: str, source_value: str | bytes | None = None) -> bool:
    return sample_source(sample_hash, source_value) != "real"


def read_cache_sources(h5) -> list[str] | None:
    if "source" not in h5:
        return None
    raw = h5["source"][:]
    out: list[str] = []
    for item in raw:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="replace"))
        else:
            out.append(str(item))
    return out


def read_source_sample_hashes(h5) -> list[str] | None:
    """Read parent sample lineage for materialized supplementary rows.

    A cache without this field cannot prove that a generated row came from the
    current training partition and is therefore unsafe for final training.
    """

    if "source_sample_hash" not in h5:
        return None
    return [decode_h5_string(item) for item in h5["source_sample_hash"][:]]


def read_source_folds(h5) -> list[int] | None:
    if "source_fold" not in h5:
        return None
    return [int(v) for v in h5["source_fold"][:]]


def read_source_train_only(h5) -> list[bool] | None:
    if "source_train_only" not in h5:
        return None
    return [bool(v) for v in h5["source_train_only"][:]]


def supplementary_train_indices(
    sample_hashes: Iterable[str],
    statuses: Iterable[str],
    sources: list[str] | None = None,
    *,
    source_folds: list[int] | None = None,
    source_train_only: list[bool] | None = None,
    fold: int | None = None,
) -> list[int]:
    indices: list[int] = []
    for idx, (h, status) in enumerate(zip(sample_hashes, statuses, strict=True)):
        if str(status) != "OK":
            continue
        src = sources[idx] if sources is not None else None
        if is_supplementary_sample(str(h), src):
            if fold is not None:
                if source_folds is None or source_train_only is None:
                    continue
                if not source_train_only[idx] or int(source_folds[idx]) != int(fold):
                    continue
            indices.append(idx)
    return indices


def real_only_indices(
    sample_hashes: Iterable[str],
    statuses: Iterable[str],
    sources: list[str] | None = None,
) -> list[int]:
    indices: list[int] = []
    for idx, (h, status) in enumerate(zip(sample_hashes, statuses, strict=True)):
        if str(status) != "OK":
            continue
        src = sources[idx] if sources is not None else None
        if not is_supplementary_sample(str(h), src):
            indices.append(idx)
    return indices


def merge_train_with_supplementary(
    real_train_indices: list[int],
    supplementary_indices: list[int],
) -> list[int]:
    seen: set[int] = set()
    merged: list[int] = []
    for idx in list(real_train_indices) + list(supplementary_indices):
        if idx in seen:
            continue
        seen.add(idx)
        merged.append(idx)
    return merged


def cache_has_supplementary(cache_path: Path) -> bool:
    try:
        import h5py
    except ImportError:
        return False
    if not cache_path.exists():
        return False
    with h5py.File(cache_path, "r") as h5:
        hashes = [decode_h5_string(x) for x in h5["sample_hash"][:]]
        sources = read_cache_sources(h5)
        statuses = [decode_h5_string(x) for x in h5["status"][:]]
        return bool(supplementary_train_indices(hashes, statuses, sources))


def count_by_source(cache_path: Path) -> dict[str, int]:
    try:
        import h5py
    except ImportError:
        return {}
    if not cache_path.exists():
        return {}
    counts: dict[str, int] = {"real": 0, "synthetic_ddpm": 0, "simulated_augment": 0}
    with h5py.File(cache_path, "r") as h5:
        hashes = [decode_h5_string(x) for x in h5["sample_hash"][:]]
        sources = read_cache_sources(h5)
        statuses = [decode_h5_string(x) for x in h5["status"][:]]
        for idx, (h, status) in enumerate(zip(hashes, statuses, strict=True)):
            if str(status) != "OK":
                continue
            src = sources[idx] if sources is not None else None
            key = sample_source(h, src)
            counts[key] = counts.get(key, 0) + 1
    return counts
