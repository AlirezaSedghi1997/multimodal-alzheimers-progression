from __future__ import annotations

import csv
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.data.a4_schema import REQUIRED_RELATIVE_FILES, resolve_data_root
from src.utils.pet_spatial_qc import atomic_write_json, centiloid_from_suvr, load_yaml


MRI_FAIL_RULES = {
    "min_brain_volume_ml": 750.0,
    "max_brain_volume_ml": 1900.0,
    "min_gm_wm_cnr": 0.20,
    "min_normalized_nonzero_fraction": 0.05,
    "min_normalized_brain_std": 0.50,
}

MRI_WARNING_THRESHOLDS = {
    "min_brain_volume_ml": 800.0,
    "max_brain_volume_ml": 1800.0,
    "min_gm_fraction": 0.25,
    "max_gm_fraction": 0.70,
    "min_gm_wm_cnr": 0.25,
    "max_abs_normalized_brain_mean": 0.15,
    "min_normalized_brain_std": 0.75,
    "max_normalized_brain_std": 1.25,
    "max_icv_brain_volume_rel_delta": 0.30,
}

PET_QC_FAIL = {
    "ncc_fail_below": 0.60,
    "ncc_warn_below": 0.75,
    "centiloid_abs_delta_max": 15.0,
}

SUVR_AMYLOID_REL = REQUIRED_RELATIVE_FILES["suvr_amyloid"]
SUVR_POLICY_PATH = Path("configs/model/method_y_suvr_reference_policy.yaml")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_float(value: object) -> float | None:
    if value in (None, "", "NA", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def classify_mri_numeric_qc(qc: Mapping[str, Any]) -> dict[str, Any]:
    """Return qc_status PASS | HOLD | FAIL with failures/warnings lists."""
    failures: list[str] = []
    warnings: list[str] = []

    brain_volume = float(qc.get("brain_volume_ml") or 0.0)
    gm_wm_cnr = float(qc.get("gm_wm_cnr") or 0.0)
    nonzero_frac = float(qc.get("normalized_nonzero_fraction") or 0.0)
    norm_std = float(qc.get("normalized_brain_std") or 0.0)
    icv_ml = _parse_float(qc.get("icv_ml_official"))

    if brain_volume < MRI_FAIL_RULES["min_brain_volume_ml"] or brain_volume > MRI_FAIL_RULES["max_brain_volume_ml"]:
        failures.append("brain_volume_fail_range")
    elif (
        brain_volume < MRI_WARNING_THRESHOLDS["min_brain_volume_ml"]
        or brain_volume > MRI_WARNING_THRESHOLDS["max_brain_volume_ml"]
    ):
        warnings.append("brain_volume_outside_expected_range")

    if gm_wm_cnr < MRI_FAIL_RULES["min_gm_wm_cnr"]:
        failures.append("gm_wm_cnr_fail")
    elif gm_wm_cnr < MRI_WARNING_THRESHOLDS["min_gm_wm_cnr"]:
        warnings.append("low_gm_wm_contrast_to_noise")

    if nonzero_frac < MRI_FAIL_RULES["min_normalized_nonzero_fraction"]:
        failures.append("normalized_nonzero_fraction_fail")
    if norm_std < MRI_FAIL_RULES["min_normalized_brain_std"]:
        failures.append("normalized_brain_std_fail")
    elif not (
        MRI_WARNING_THRESHOLDS["min_normalized_brain_std"]
        <= norm_std
        <= MRI_WARNING_THRESHOLDS["max_normalized_brain_std"]
    ):
        warnings.append("normalized_brain_std_outside_expected_range")

    gm_fraction = float(qc.get("tissue_fraction_gm") or 0.0)
    if not (
        MRI_WARNING_THRESHOLDS["min_gm_fraction"] <= gm_fraction <= MRI_WARNING_THRESHOLDS["max_gm_fraction"]
    ):
        warnings.append("gray_matter_fraction_outside_expected_range")

    norm_mean = float(qc.get("normalized_brain_mean") or 0.0)
    if abs(norm_mean) > MRI_WARNING_THRESHOLDS["max_abs_normalized_brain_mean"]:
        warnings.append("normalized_brain_mean_not_centered")

    if icv_ml is not None and icv_ml > 0 and brain_volume > 0:
        rel_delta = abs(brain_volume - icv_ml) / icv_ml
        qc = {**dict(qc), "icv_brain_volume_rel_delta": round(rel_delta, 6)}
        if rel_delta > MRI_WARNING_THRESHOLDS["max_icv_brain_volume_rel_delta"]:
            warnings.append("icv_brain_volume_mismatch_review")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "HOLD"
    else:
        status = "PASS"

    return {
        **dict(qc),
        "qc_status": status,
        "failures": failures,
        "warnings": warnings,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
    }


def load_official_centiloid_by_bid(data_root: Path | None = None) -> dict[tuple[str, str], float]:
    """Map (SUBSTUDY, BID) -> official centiloid from imaging_PET_VA.csv."""
    root = resolve_data_root(str(data_root) if data_root else None)
    va_path = root / REQUIRED_RELATIVE_FILES["ab_label"]
    if not va_path.exists():
        va_path = _resolve_by_name(root, "imaging_PET_VA.csv")
    if not va_path.exists():
        return {}
    out: dict[tuple[str, str], float] = {}
    for row in read_csv_dicts(va_path):
        substudy = (row.get("SUBSTUDY") or row.get("substudy") or "").strip().upper()
        bid = (row.get("BID") or row.get("bid") or "").strip().upper()
        score = row.get("overall_score") or row.get("CENTILOID") or row.get("centiloid")
        val = _parse_float(score)
        if substudy and bid and val is not None:
            out[(substudy, bid)] = val
    return out


def load_official_suvr_by_bid(data_root: Path | None = None) -> dict[tuple[str, str], float]:
    """Map (SUBSTUDY, BID) -> official amyloid SUVR from imaging_SUVR_amyloid.csv."""
    root = resolve_data_root(str(data_root) if data_root else None)
    suvr_path = root / SUVR_AMYLOID_REL
    if not suvr_path.exists():
        suvr_path = _resolve_by_name(root, "imaging_SUVR_amyloid.csv")
    if not suvr_path.exists():
        return {}
    candidates = (
        "SUVR_COMPOSITE",
        "suvr_composite",
        "COMPOSITE_SUVR",
        "composite_suvr",
        "SUVR",
        "suvr",
        "overall_suvr",
    )
    out: dict[tuple[str, str], float] = {}
    for row in read_csv_dicts(suvr_path):
        substudy = (row.get("SUBSTUDY") or row.get("substudy") or "").strip().upper()
        bid = (row.get("BID") or row.get("bid") or "").strip().upper()
        suvr_val = None
        for key in candidates:
            if key in row:
                suvr_val = _parse_float(row[key])
                if suvr_val is not None:
                    break
        if not substudy or not bid or suvr_val is None:
            continue
        out[(substudy, bid)] = suvr_val
    return out


def load_icv_ml_by_sample_hash(
    data_root: Path | None,
    catalog_path: Path,
) -> dict[str, float]:
    """Map sample_hash -> official ICV (ml) via volumetric MRI + WMH QC tables."""
    root = resolve_data_root(str(data_root) if data_root else None)
    if not catalog_path.exists():
        return {}
    catalog_rows = read_csv_dicts(catalog_path)
    patient_to_sample = {r.get("patient_hash", ""): r.get("sample_hash", "") for r in catalog_rows if r.get("patient_hash")}

    wmh_path = root / "Clinical/External Data/imaging_FLAIR_WMH_QC.csv"
    if not wmh_path.exists():
        wmh_path = _resolve_by_name(root, "imaging_FLAIR_WMH_QC.csv")
    icv_by_patient: dict[str, float] = {}
    if wmh_path.exists():
        for row in read_csv_dicts(wmh_path):
            substudy = (row.get("SUBSTUDY") or "").strip().upper()
            bid = (row.get("BID") or "").strip()
            icv = _parse_float(row.get("ICV"))
            if substudy in {"A4", "LEARN"} and bid and icv is not None:
                from src.utils.privacy import stable_hash

                patient_hash = stable_hash(substudy + "|" + bid, prefix="pt")
                icv_by_patient[patient_hash] = icv

    out: dict[str, float] = {}
    for patient_hash, icv in icv_by_patient.items():
        sample_hash = patient_to_sample.get(patient_hash)
        if sample_hash:
            out[sample_hash] = icv
    return out


def _resolve_by_name(root: Path, name: str) -> Path:
    if not root.exists():
        return root / name
    for path in root.rglob(name):
        if path.is_file():
            return path
    return root / name


def load_suvr_reference_policy(path: Path | None = None) -> dict[str, Any]:
    policy_path = path or SUVR_POLICY_PATH
    if not policy_path.exists():
        return {}
    return load_yaml(policy_path)


def derived_centiloid_from_official_suvr(
    official_suvr: float | None,
    *,
    tracer: str,
    suvr_policy: Mapping[str, Any] | None = None,
) -> float | None:
    if official_suvr is None:
        return None
    policy = suvr_policy or load_suvr_reference_policy()
    return centiloid_from_suvr(float(official_suvr), tracer, policy)


def attach_pet_official_qc_fields(
    qc_after: dict[str, Any],
    *,
    substudy: str,
    bid: str,
    tracer: str,
    centiloid_lookup: dict[tuple[str, str], float] | None,
    suvr_lookup: dict[tuple[str, str], float] | None,
    suvr_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    key = (substudy.strip().upper(), bid.strip().upper())
    if centiloid_lookup and key in centiloid_lookup:
        qc_after["official_centiloid"] = centiloid_lookup[key]
    if suvr_lookup and key in suvr_lookup:
        qc_after["official_suvr"] = suvr_lookup[key]
        derived = derived_centiloid_from_official_suvr(
            suvr_lookup[key], tracer=tracer, suvr_policy=suvr_policy
        )
        if derived is not None:
            qc_after["derived_centiloid"] = derived
            qc_after["derived_centiloid_source"] = "official_suvr_equation"
    return qc_after


def flag_icv_mad_outliers(rows: list[dict]) -> list[dict]:
    """Flag brain_volume/icv ratios outside cohort median ± 3 MAD (audit layer)."""
    pairs: list[tuple[dict, float]] = []
    for row in rows:
        bv = _parse_float(row.get("brain_volume_ml"))
        icv = _parse_float(row.get("icv_ml_official"))
        if bv is None or icv is None or icv <= 0:
            continue
        pairs.append((row, bv / icv))
    if len(pairs) < 5:
        return []
    ratios = [ratio for _, ratio in pairs]
    med = statistics.median(ratios)
    abs_dev = [abs(r - med) for r in ratios]
    mad = statistics.median(abs_dev) or 1e-6
    scaled_mad = 1.4826 * mad
    outliers: list[dict] = []
    for row, ratio in pairs:
        if abs(ratio - med) > 3 * scaled_mad:
            outliers.append(
                {
                    "sample_hash": row.get("sample_hash"),
                    "brain_icv_ratio": round(ratio, 6),
                    "cohort_median_ratio": round(med, 6),
                    "mad_scaled": round(scaled_mad, 6),
                }
            )
    return outliers


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    try:
        return float(statistics.correlation(xs, ys))
    except statistics.StatisticsError:
        return None


def assess_pet_centiloid_agreement(
    qc: Mapping[str, Any],
    *,
    official_centiloid: float | None,
    derived_centiloid: float | None = None,
) -> dict[str, Any]:
    failures = list(qc.get("failures") or [])
    warnings = list(qc.get("warnings") or [])
    if official_centiloid is not None:
        qc = {**dict(qc), "official_centiloid": official_centiloid}
    if derived_centiloid is not None and official_centiloid is not None:
        delta = abs(float(derived_centiloid) - float(official_centiloid))
        qc = {**dict(qc), "derived_centiloid": derived_centiloid, "centiloid_abs_delta": delta}
        if delta > PET_QC_FAIL["centiloid_abs_delta_max"]:
            failures.append("CENTILOID_DELTA_FAIL")
        elif delta > PET_QC_FAIL["centiloid_abs_delta_max"] / 2:
            warnings.append("CENTILOID_DELTA_REVIEW")
    status = "FAIL" if failures else ("PASS" if not warnings else qc.get("status", "PASS"))
    if failures:
        status = "FAIL"
    elif warnings and status == "PASS":
        status = "WARN"
    return {**dict(qc), "failures": failures, "warnings": warnings, "status": status}
