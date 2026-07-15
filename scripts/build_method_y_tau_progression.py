from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import contains_raw_identifier, stable_hash


DEFAULT_CONFIG = Path("configs/method_y_tau_progression_v1.yaml")
REFERENCE_PATTERN = re.compile(
    r"(manufacturer|model|scanner|station|site|institution|device|series|protocol|tracer|radiopharmaceutical|modality|software)",
    re.IGNORECASE,
)
FTP_TERMS = ("flortaucipir", "av-1451", "av1451", "tau", "1451")
NON_FTP_TERMS = ("florbetapir", "fdg", "fluorodeoxyglucose")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NAN", "NULL", "NONE", "."}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def norm(value: object) -> str:
    return "" if value is None else str(value).strip()


def norm_upper(value: object) -> str:
    return norm(value).upper()


def subject_key(row: dict[str, str]) -> tuple[str, str]:
    return norm_upper(row.get("SUBSTUDY")), norm(row.get("BID"))


def patient_hash_for_key(key: tuple[str, str]) -> str:
    return stable_hash(f"{key[0]}|{key[1]}", prefix="pt")


def norm_visit(value: object) -> int | None:
    v = safe_float(value)
    if v is None:
        return None
    return int(v)


def sanitize_region(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return text or "unknown_region"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "column_count": 0}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return {
        "exists": True,
        "rows": rows,
        "column_count": len(header),
        "raw_identifier_field_names_suppressed": True,
    }


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    content = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "status",
        "blocker",
        "internal_blocker",
        "internal_s1_status",
        "cross_tracer_claim_allowed",
        "option3_succeeded",
        "candidate_non_fbp_amyloid_pet_images",
        "internal_metrics",
        "s2_s5_allowed",
        "heldout_confirmation_status",
    )
    return {"exists": True, "summary": {key: content.get(key) for key in keys if key in content}}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def cohort_for_subjinfo(row: dict[str, str] | None) -> str:
    if not row:
        return "unknown"
    if norm(row.get("LEARNFL")) == "1":
        return "LEARN"
    tx = norm_upper(row.get("TX"))
    if "PLACEBO" in tx:
        return "A4_placebo"
    if "SOLANEZUMAB" in tx:
        return "A4_treatment"
    if tx and tx != "NA":
        return "A4_other"
    return "unknown"


def load_first_tau_scans(tau_path: Path, config: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    primary_ref = config["inputs"]["tau_primary_suvr_column"]
    sensitivity_ref = config["inputs"]["tau_sensitivity_suvr_column"]
    scan_required = config["inputs"]["tau_scan_analyzed_required"].lower()
    ligand_required = config["inputs"]["tau_ligand_required"].lower()
    groups: dict[tuple[tuple[str, str], str, str, float], dict[str, Any]] = {}
    row_count = 0
    ligand_counts: Counter[str] = Counter()
    scan_analyzed_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    suvr_nonmissing: Counter[str] = Counter()

    with tau_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row_count += 1
            ligand = norm(row.get("ligand"))
            scan_analyzed = norm(row.get("scan_analyzed"))
            ligand_counts[ligand] += 1
            scan_analyzed_counts[scan_analyzed] += 1
            region = norm(row.get("brain_region"))
            if region:
                region_counts[region] += 1
            for col in ("suvr_cer", "suvr_persi", "suvr_crus"):
                if safe_float(row.get(col)) is not None:
                    suvr_nonmissing[col] += 1
            if scan_analyzed.lower() != scan_required or ligand.lower() != ligand_required:
                continue
            scan_date = safe_float(row.get("scan_date_DAYS_T0"))
            if scan_date is None:
                continue
            key = subject_key(row)
            if not key[0] or not key[1]:
                continue
            group_key = (key, norm(row.get("VISCODE")), norm(row.get("scan_number")), scan_date)
            group = groups.setdefault(
                group_key,
                {
                    "subject_key": key,
                    "viscode": norm(row.get("VISCODE")),
                    "scan_number": norm(row.get("scan_number")),
                    "scan_date": scan_date,
                    "regions": {},
                },
            )
            if region:
                group["regions"][region] = {
                    primary_ref: safe_float(row.get(primary_ref)),
                    sensitivity_ref: safe_float(row.get(sensitivity_ref)),
                }

    first: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups.values():
        key = group["subject_key"]
        if key not in first or (group["scan_date"], group["viscode"], group["scan_number"]) < (
            first[key]["scan_date"],
            first[key]["viscode"],
            first[key]["scan_number"],
        ):
            first[key] = group

    audit = {
        "tau_suvr_rows": row_count,
        "eligible_scan_groups": len(groups),
        "first_analyzed_tau_subjects": len(first),
        "ligand_counts": dict(ligand_counts),
        "scan_analyzed_counts": dict(scan_analyzed_counts),
        "unique_brain_regions": len(region_counts),
        "top_brain_regions": dict(region_counts.most_common(25)),
        "suvr_nonmissing_rows": dict(suvr_nonmissing),
        "has_explicit_braak_region_names": any("braak" in name.lower() for name in region_counts),
        "has_explicit_meta_roi_names": any("meta" in name.lower() for name in region_counts),
        "primary_suvr_column": primary_ref,
        "sensitivity_suvr_column": sensitivity_ref,
        "unavailable_suvr_column": config["inputs"]["tau_unavailable_suvr_column"],
    }
    return first, audit


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = subject_key(row)
        if key[0] and key[1]:
            out[key].append(row)
    return out


def sorted_cdr_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        date = safe_float(row.get("CDADTC_DAYS_T0"))
        cdglobal = safe_float(row.get("CDGLOBAL"))
        if date is None or cdglobal is None:
            continue
        item: dict[str, Any] = dict(row)
        item["_date"] = date
        item["_cdglobal"] = cdglobal
        item["_cdrsb"] = safe_float(row.get("CDRSB"))
        out.append(item)
    return sorted(out, key=lambda r: (r["_date"], norm(r.get("VISCODE"))))


def select_baseline_cdr(cdr_rows: list[dict[str, Any]], tau_date: float, window_days: float) -> tuple[dict[str, Any] | None, float | None]:
    if not cdr_rows:
        return None, None
    row = min(cdr_rows, key=lambda r: abs(float(r["_date"]) - tau_date))
    gap = abs(float(row["_date"]) - tau_date)
    if gap > window_days:
        return None, gap
    return row, gap


def selected_pacc_row(pacc_rows: list[dict[str, str]], tau_viscode: str) -> dict[str, str] | None:
    if not pacc_rows:
        return None
    target_visit = norm_visit(tau_viscode)
    candidates: list[tuple[int, dict[str, str]]] = []
    for row in pacc_rows:
        visit = norm_visit(row.get("VISCODE"))
        if visit is None:
            continue
        if target_visit is None or visit <= target_visit:
            candidates.append((visit, row))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return min(pacc_rows, key=lambda row: norm_visit(row.get("VISCODE")) or 999999)


def detect_sustained_event(
    followups: list[dict[str, Any]],
    *,
    event_time_rule: str = "confirmation_visit",
) -> tuple[bool, float | None]:
    prev_positive = False
    prev_positive_date: float | None = None
    for row in followups:
        current_positive = float(row["_cdglobal"]) >= 0.5
        if prev_positive and current_positive:
            if event_time_rule == "first_increase_if_confirmed_by_next_visit":
                return True, prev_positive_date
            return True, float(row["_date"])
        prev_positive = current_positive
        prev_positive_date = float(row["_date"]) if current_positive else None
    return False, None


def add_numeric(row: dict[str, Any], key: str, value: object) -> None:
    parsed = safe_float(value)
    row[key] = "" if parsed is None else parsed


def build_label_rows(
    first_tau: dict[tuple[str, str], dict[str, Any]],
    subj_rows: list[dict[str, str]],
    cdr_rows: list[dict[str, str]],
    pacc_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    subj_by_key = {subject_key(row): row for row in subj_rows}
    cdr_by_key = {key: sorted_cdr_rows(rows) for key, rows in index_rows(cdr_rows).items()}
    pacc_by_key = index_rows(pacc_rows)
    endpoint = config["endpoint"]
    primary_ref = config["inputs"]["tau_primary_suvr_column"]
    sensitivity_ref = config["inputs"]["tau_sensitivity_suvr_column"]
    baseline_window = float(endpoint["baseline_tau_to_cdr_window_days"])
    followup_window = float(endpoint["followup_window_days"])

    exclusions: Counter[str] = Counter()
    baseline_gaps: list[float] = []
    event_counts_by_cohort: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    all_region_features: set[str] = set()

    for key, tau in sorted(first_tau.items(), key=lambda item: patient_hash_for_key(item[0])):
        subj = subj_by_key.get(key)
        if subj is None:
            exclusions["missing_subjinfo"] += 1
            continue
        cohort = cohort_for_subjinfo(subj)
        cdr_series = cdr_by_key.get(key, [])
        baseline_cdr, gap = select_baseline_cdr(cdr_series, float(tau["scan_date"]), baseline_window)
        if baseline_cdr is None:
            exclusions["missing_baseline_cdr_within_window"] += 1
            continue
        baseline_gaps.append(float(gap or 0.0))
        if float(baseline_cdr["_cdglobal"]) != float(endpoint["baseline_cdr_required"]):
            exclusions["baseline_cdr_not_zero"] += 1
            continue
        index_date = max(float(tau["scan_date"]), float(baseline_cdr["_date"]))
        followups = [
            row
            for row in cdr_series
            if float(row["_date"]) > index_date and float(row["_date"]) <= index_date + followup_window
        ]
        if not followups:
            exclusions["no_post_index_cdr_followup"] += 1
            continue
        has_event, event_date = detect_sustained_event(
            followups,
            event_time_rule=str(endpoint.get("event_time_rule", "confirmation_visit")),
        )
        if has_event and event_date is not None:
            event_time = event_date - index_date
        else:
            event_time = min(followup_window, float(followups[-1]["_date"]) - index_date)
        if event_time <= 0:
            exclusions["nonpositive_observed_time"] += 1
            continue

        pacc = selected_pacc_row(pacc_by_key.get(key, []), str(tau["viscode"]))
        regions = tau["regions"]
        primary_values = [vals.get(primary_ref) for vals in regions.values() if vals.get(primary_ref) is not None]
        sensitivity_values = [vals.get(sensitivity_ref) for vals in regions.values() if vals.get(sensitivity_ref) is not None]
        row: dict[str, Any] = {
            "patient_hash": patient_hash_for_key(key),
            "cohort": cohort,
            "treatment_arm": norm(subj.get("TX")) or "NA",
            "baseline_gap_days_abs": round(float(gap or 0.0), 6),
            "index_source": "max_ftp_scan_and_baseline_cdr_dates",
            "event": int(has_event),
            "time_days": round(float(event_time), 6),
            "censoring_rule": "event_time" if has_event else "right_censored",
            "followup_cdr_visits": len(followups),
            "single_visit_cdglobal_ge_0_5": int(any(float(r["_cdglobal"]) >= 0.5 for r in followups)),
            "baseline_cdglobal": float(baseline_cdr["_cdglobal"]),
            "baseline_cdrsb": "" if baseline_cdr.get("_cdrsb") is None else float(baseline_cdr["_cdrsb"]),
            "global_tau_persi_mean": "" if not primary_values else float(np.mean(primary_values)),
            "global_tau_crus_mean": "" if not sensitivity_values else float(np.mean(sensitivity_values)),
            "tau_region_count": len(regions),
            "uses_raw_identifier": False,
            "uses_raw_path": False,
        }
        add_numeric(row, "age", subj.get("AGEYR"))
        sex = norm(subj.get("SEX"))
        row["sex_binary"] = 1.0 if sex in {"2", "M", "Male", "male"} else 0.0 if sex else ""
        add_numeric(row, "education_years", subj.get("EDCCNTU"))
        carrier = safe_float(subj.get("AAPOEGNPRSNFLG"))
        if carrier is None:
            carrier = safe_float(subj.get("APOEGNPRSNFLG"))
        row["apoe4_carrier"] = "" if carrier is None else carrier
        add_numeric(row, "baseline_amyloid_centiloid", subj.get("AMYLCENT"))
        amyloid_centiloid = safe_float(subj.get("AMYLCENT"))
        row["baseline_amyloid_status_cl24"] = "" if amyloid_centiloid is None else int(amyloid_centiloid >= 24.0)
        for out_name, src_name in {
            "baseline_pacc_raw": "PACC.raw",
            "baseline_pacc_change": "PACC",
            "baseline_fctotal96": "FCTOTAL96",
            "baseline_ldeltotal": "LDELTOTAL",
            "baseline_digittotal": "DIGITTOTAL",
            "baseline_mmscore": "MMSCORE",
        }.items():
            add_numeric(row, out_name, pacc.get(src_name) if pacc else None)
        for region, vals in regions.items():
            feature = "tau_persi__" + sanitize_region(region)
            row[feature] = "" if vals.get(primary_ref) is None else float(vals[primary_ref])
            all_region_features.add(feature)
        rows.append(row)
        if has_event:
            event_counts_by_cohort[cohort] += 1

    rows.sort(key=lambda row: str(row["patient_hash"]))
    endpoint_audit = {
        "status": "COMPLETED",
        "endpoint_name": endpoint["primary_name"],
        "event_definition": endpoint["event_definition"],
        "time_zero_rule": endpoint["index_date_rule"],
        "event_flag_columns_used_as_predictors": False,
        "forbidden_label_sources_not_used": endpoint["forbidden_label_sources"],
        "eligible_rows": len(rows),
        "events": sum(int(row["event"]) for row in rows),
        "censored": sum(1 for row in rows if int(row["event"]) == 0),
        "exclusions": dict(exclusions),
        "cohort_counts": dict(Counter(str(row["cohort"]) for row in rows)),
        "event_counts_by_cohort": dict(event_counts_by_cohort),
        "baseline_gap_days_abs": {
            "n": len(baseline_gaps),
            "max": max(baseline_gaps) if baseline_gaps else None,
            "median": float(np.median(baseline_gaps)) if baseline_gaps else None,
        },
        "stop_conditions": {
            "min_total_events": config["validation"]["min_total_events"],
            "min_events_per_heldout_arm": config["validation"]["min_events_per_heldout_arm"],
            "total_event_gate_passed": sum(int(row["event"]) for row in rows) >= int(config["validation"]["min_total_events"]),
            "heldout_arm_event_gate_passed": all(
                count >= int(config["validation"]["min_events_per_heldout_arm"])
                for cohort, count in event_counts_by_cohort.items()
                if cohort in {"A4_placebo", "A4_treatment"}
            ),
        },
    }
    return rows, endpoint_audit, sorted(all_region_features)


def iter_json_items(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                yield from iter_json_items(value, name)
            else:
                yield name, value
    elif isinstance(obj, list):
        for value in obj[:5]:
            name = f"{prefix}[]" if prefix else "[]"
            if isinstance(value, (dict, list)):
                yield from iter_json_items(value, name)
            else:
                yield name, value


def audit_ftp_sidecars(ftp_dir: Path) -> dict[str, Any]:
    if not ftp_dir.exists():
        return {"exists": False, "status": "FTP_IMAGE_DIR_MISSING"}
    json_files = sorted(ftp_dir.glob("*.json"))
    nii_files = sorted(ftp_dir.glob("*.nii.gz"))
    root_types: Counter[str] = Counter()
    keys: Counter[str] = Counter()
    interesting: dict[str, Counter[str]] = defaultdict(Counter)
    parse_errors = 0
    for path in json_files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            parse_errors += 1
            continue
        root_types[type(obj).__name__] += 1
        for key, value in iter_json_items(obj):
            keys[key] += 1
            if REFERENCE_PATTERN.search(key):
                text = norm(value)
                if text:
                    interesting[key][text] += 1
    radiopharm_counts: Counter[str] = Counter()
    manufacturer_counts: Counter[str] = Counter()
    for key, values in interesting.items():
        if "radiopharmaceutical" in key.lower():
            radiopharm_counts.update(values)
        if "manufacturer" in key.lower():
            manufacturer_counts.update(values)
    radiopharm_texts = [text.lower() for text in radiopharm_counts]
    non_ftp_hits = {term: sum(count for text, count in radiopharm_counts.items() if term in text.lower()) for term in NON_FTP_TERMS}
    ftp_hits = {term: sum(count for text, count in radiopharm_counts.items() if term in text.lower()) for term in FTP_TERMS}
    image_cache_status = "IMAGE_CACHE_BLOCKED_PENDING_SIDECAR_QC" if any(non_ftp_hits.values()) else "IMAGE_CACHE_QC_AUDIT_PASSED_AGGREGATE"
    return {
        "exists": True,
        "json_sidecar_files": len(json_files),
        "nii_gz_files": len(nii_files),
        "parse_errors": parse_errors,
        "root_json_types": dict(root_types),
        "top_keys": dict(keys.most_common(50)),
        "manufacturer_counts": dict(manufacturer_counts.most_common(20)),
        "radiopharmaceutical_counts": dict(radiopharm_counts.most_common(30)),
        "ftp_term_hits": ftp_hits,
        "non_ftp_term_hits": non_ftp_hits,
        "image_cache_status": image_cache_status,
        "note": "Mixed radiopharmaceutical sidecar text blocks image-CNN/cache claims until per-scan QC links sidecars to selected FTP scans.",
    }


def build_splits(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cohort[str(row["cohort"])].append(row)
    primary_folds = []
    for heldout in ("A4_placebo", "A4_treatment"):
        test = by_cohort.get(heldout, [])
        train = [row for row in rows if str(row["cohort"]) != heldout]
        primary_folds.append(
            {
                "fold": f"holdout_{heldout}",
                "heldout": heldout,
                "train_patient_hashes": [str(row["patient_hash"]) for row in train],
                "test_patient_hashes": [str(row["patient_hash"]) for row in test],
                "train_n": len(train),
                "test_n": len(test),
                "train_events": sum(int(row["event"]) for row in train),
                "test_events": sum(int(row["event"]) for row in test),
            }
        )

    try:
        from sklearn.model_selection import StratifiedKFold
    except Exception:
        secondary_folds = []
    else:
        labels = np.array([f"{row['cohort']}|{row['event']}" for row in rows])
        idx = np.arange(len(rows))
        # StratifiedKFold requires every class to have at least n_splits members.
        n_splits = min(5, min(Counter(labels).values())) if len(rows) else 0
        secondary_folds = []
        if n_splits >= 2:
            for fold_idx, (train_idx, test_idx) in enumerate(
                StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17).split(idx, labels),
                start=1,
            ):
                train = [rows[i] for i in train_idx]
                test = [rows[i] for i in test_idx]
                secondary_folds.append(
                    {
                        "fold": fold_idx,
                        "train_patient_hashes": [str(row["patient_hash"]) for row in train],
                        "test_patient_hashes": [str(row["patient_hash"]) for row in test],
                        "train_n": len(train),
                        "test_n": len(test),
                        "train_events": sum(int(row["event"]) for row in train),
                        "test_events": sum(int(row["event"]) for row in test),
                    }
                )
    return {
        "schema_version": 1,
        "primary_split": config["validation"]["primary_split"],
        "subject_level_only": True,
        "blocked_split_until_new_evidence": config["validation"]["blocked_split_until_new_evidence"],
        "primary_folds": primary_folds,
        "secondary_split": config["validation"]["secondary_split"],
        "secondary_folds": secondary_folds,
    }


def ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float | None:
    if len(y_true) == 0:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask])))
    return float(ece)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(y_true)) < 2 or len(y_true) < 10:
        return {"slope": None, "intercept": None}
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return {"slope": None, "intercept": None}
    eps = 1e-6
    logits = np.log(np.clip(y_prob, eps, 1 - eps) / np.clip(1 - y_prob, eps, 1 - eps)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, y_true.astype(int))
    return {"slope": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])}


def bootstrap_ci(values_fn, n: int, n_boot: int = 1000, seed: int = 23) -> list[float | None]:
    if n <= 1:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        try:
            val = values_fn(indices)
        except Exception:
            continue
        if val is not None and np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bootstrap_ci_from_values(values: list[float | None], n_boot: int = 1000, seed: int = 29) -> list[float | None]:
    finite = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=float)
    if len(finite) <= 1:
        return [None, None]
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(finite[rng.integers(0, len(finite), size=len(finite))])) for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def calibration_curve_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    if len(y_true) == 0:
        return []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, Any]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        out.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "n": int(np.sum(mask)),
                "mean_predicted_risk": float(np.mean(y_prob[mask])),
                "observed_event_fraction": float(np.mean(y_true[mask])),
            }
        )
    return out


def run_survival_models(rows: list[dict[str, Any]], region_features: list[str], split_artifact: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.metrics import brier_score, concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
        from sksurv.util import Surv
    except Exception as exc:
        return {
            "status": "SURVIVAL_DEPENDENCY_BLOCKED",
            "error": str(exc),
            "primary_metrics_valid": False,
        }

    by_hash = {str(row["patient_hash"]): row for row in rows}
    clinical = [
        "age",
        "sex_binary",
        "education_years",
        "apoe4_carrier",
        "baseline_amyloid_centiloid",
        "baseline_amyloid_status_cl24",
    ]
    cognition = [
        "baseline_pacc_raw",
        "baseline_fctotal96",
        "baseline_ldeltotal",
        "baseline_digittotal",
        "baseline_mmscore",
        "baseline_cdrsb",
    ]
    variants = {
        "clinical_only": {"features": clinical, "default_alpha": 0.1},
        "clinical_cognition": {"features": clinical + cognition, "default_alpha": 0.2},
        "tau_global": {"features": ["global_tau_persi_mean", "global_tau_crus_mean"], "default_alpha": 0.1},
        "regional_tau": {"features": region_features, "default_alpha": 1.0},
        "regional_tau_clinical": {"features": region_features + clinical + cognition, "default_alpha": 2.0},
    }
    requested_times = np.array([float(v) for v in config["validation"]["evaluation_times_days"]], dtype=float)
    n_boot = int(config["validation"]["bootstrap_iterations"])
    alpha_grid = [float(v) for v in config["validation"].get("alpha_grid", [])]
    inner_cv_folds = int(config["validation"].get("inner_cv_folds", 3))

    def make_x(rows_: list[dict[str, Any]], features_: list[str]) -> pd.DataFrame:
        return pd.DataFrame([{feature: safe_float(row.get(feature)) for feature in features_} for row in rows_])

    def make_y(rows_: list[dict[str, Any]]):
        return Surv.from_arrays(
            event=np.array([bool(int(row["event"])) for row in rows_]),
            time=np.array([float(row["time_days"]) for row in rows_]),
        )

    def fit_pipeline(alpha: float):
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("cox", CoxPHSurvivalAnalysis(alpha=float(alpha), n_iter=200)),
            ]
        )

    def select_alpha(train_rows: list[dict[str, Any]], features: list[str], default_alpha: float) -> dict[str, Any]:
        candidates = alpha_grid or [default_alpha]
        labels = np.array([int(row["event"]) for row in train_rows], dtype=int)
        counts = Counter(labels)
        n_splits = min(inner_cv_folds, min(counts.values())) if len(counts) >= 2 else 0
        if n_splits < 2:
            return {
                "method": "default_alpha_inner_cv_unavailable",
                "selected_alpha": default_alpha,
                "candidate_mean_c_index": {},
            }
        x_all = make_x(train_rows, features)
        y_all = make_y(train_rows)
        idx = np.arange(len(train_rows))
        scores: dict[str, float | None] = {}
        for alpha in candidates:
            fold_scores: list[float] = []
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=31)
            for inner_train_idx, inner_valid_idx in splitter.split(idx, labels):
                try:
                    model = fit_pipeline(alpha)
                    model.fit(x_all.iloc[inner_train_idx], y_all[inner_train_idx])
                    risk = np.asarray(model.predict(x_all.iloc[inner_valid_idx]), dtype=float)
                    cindex = concordance_index_censored(
                        y_all["event"][inner_valid_idx],
                        y_all["time"][inner_valid_idx],
                        risk,
                    )[0]
                    if np.isfinite(cindex):
                        fold_scores.append(float(cindex))
                except Exception:
                    continue
            scores[str(alpha)] = float(np.mean(fold_scores)) if fold_scores else None
        finite_scores = {float(k): v for k, v in scores.items() if v is not None and np.isfinite(float(v))}
        selected = max(finite_scores, key=lambda a: (float(finite_scores[a]), -abs(a - default_alpha))) if finite_scores else default_alpha
        return {
            "method": "nested_inner_cv_c_index",
            "inner_cv_folds": n_splits,
            "selected_alpha": float(selected),
            "candidate_mean_c_index": scores,
        }

    def fixed_horizon_metrics(
        horizon: float,
        y_test,
        test_rows: list[dict[str, Any]],
        fixed_risk: np.ndarray | None,
    ) -> dict[str, Any]:
        if fixed_risk is None:
            return {"status": "NOT_EVALUABLE", "resolved_n": 0}
        resolved_y = []
        resolved_p = []
        for idx_, row in enumerate(test_rows):
            time = float(row["time_days"])
            event = int(row["event"])
            if event and time <= horizon:
                resolved_y.append(1)
                resolved_p.append(float(fixed_risk[idx_]))
            elif time >= horizon:
                resolved_y.append(0)
                resolved_p.append(float(fixed_risk[idx_]))
        y_arr = np.asarray(resolved_y, dtype=int)
        p_arr = np.asarray(resolved_p, dtype=float)
        brier = float(np.mean((y_arr - p_arr) ** 2)) if len(y_arr) else None
        ece = ece_binary(y_arr, p_arr) if len(y_arr) else None
        cal = calibration_slope_intercept(y_arr, p_arr) if len(y_arr) else {"slope": None, "intercept": None}
        return {
            "status": "OK" if len(y_arr) else "NO_CENSORING_RESOLVED_ROWS",
            "resolved_n": int(len(y_arr)),
            "event_n": int(np.sum(y_arr)) if len(y_arr) else 0,
            "brier": brier,
            "brier_bootstrap_95ci": bootstrap_ci(lambda indices: float(np.mean((y_arr[indices] - p_arr[indices]) ** 2)), len(y_arr), n_boot=n_boot)
            if len(y_arr)
            else [None, None],
            "ece": ece,
            "ece_bootstrap_95ci": bootstrap_ci(lambda indices: ece_binary(y_arr[indices], p_arr[indices]), len(y_arr), n_boot=n_boot)
            if len(y_arr)
            else [None, None],
            "calibration": cal,
            "calibration_bins": calibration_curve_bins(y_arr, p_arr),
        }

    def aggregate_folds(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [m for m in fold_metrics if m.get("status") == "OK"]
        if not ok:
            return {"folds_ok": 0}
        all_auc_times = sorted({key for m in ok for key in m.get("time_dependent_auc", {})}, key=float)
        all_brier_times = sorted({key for m in ok for key in m.get("brier_by_time", {})}, key=float)
        return {
            "folds_ok": len(ok),
            "mean_c_index": float(np.mean([m["c_index"] for m in ok])),
            "mean_c_index_bootstrap_95ci_across_folds": bootstrap_ci_from_values([m.get("c_index") for m in ok], n_boot=n_boot),
            "mean_integrated_brier_score": float(np.mean([m["integrated_brier_score"] for m in ok if m.get("integrated_brier_score") is not None]))
            if any(m.get("integrated_brier_score") is not None for m in ok)
            else None,
            "mean_integrated_brier_score_bootstrap_95ci_across_folds": bootstrap_ci_from_values(
                [m.get("integrated_brier_score") for m in ok],
                n_boot=n_boot,
            ),
            "mean_dynamic_auc": float(np.mean([m["mean_dynamic_auc"] for m in ok if m.get("mean_dynamic_auc") is not None]))
            if any(m.get("mean_dynamic_auc") is not None for m in ok)
            else None,
            "mean_dynamic_auc_bootstrap_95ci_across_folds": bootstrap_ci_from_values(
                [m.get("mean_dynamic_auc") for m in ok],
                n_boot=n_boot,
            ),
            "mean_time_dependent_auc_by_time": {
                key: float(np.mean([m["time_dependent_auc"][key] for m in ok if key in m.get("time_dependent_auc", {})]))
                for key in all_auc_times
            },
            "mean_brier_by_time": {
                key: float(np.mean([m["brier_by_time"][key] for m in ok if key in m.get("brier_by_time", {})]))
                for key in all_brier_times
            },
        }

    def evaluate_folds(folds: list[dict[str, Any]], split_name: str, variant_name: str, spec: dict[str, Any]) -> dict[str, Any]:
        fold_metrics = []
        features = spec["features"]
        default_alpha = float(spec["default_alpha"])
        for fold in folds:
            train_rows = [by_hash[h] for h in fold["train_patient_hashes"] if h in by_hash]
            test_rows = [by_hash[h] for h in fold["test_patient_hashes"] if h in by_hash]
            if not train_rows or not test_rows:
                fold_metrics.append({"fold": fold["fold"], "split": split_name, "status": "EMPTY_FOLD"})
                continue
            alpha_selection = select_alpha(train_rows, features, default_alpha)
            selected_alpha = float(alpha_selection["selected_alpha"])
            x_train = make_x(train_rows, features)
            x_test = make_x(test_rows, features)
            y_train = make_y(train_rows)
            y_test = make_y(test_rows)
            pipeline = fit_pipeline(selected_alpha)
            try:
                pipeline.fit(x_train, y_train)
                risk = np.asarray(pipeline.predict(x_test), dtype=float)
                cindex = float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])
                max_allowed = min(float(np.max(y_train["time"])), float(np.max(y_test["time"]))) - 1e-3
                eval_times = requested_times[requested_times < max_allowed]
                auc_by_time: dict[str, float] = {}
                mean_dynamic_auc = None
                mean_dynamic_auc_ci = [None, None]
                ibs = None
                ibs_ci = [None, None]
                brier_by_time: dict[str, float] = {}
                fixed_by_horizon: dict[str, Any] = {}
                surv_matrix: np.ndarray | None = None
                auc_times = np.asarray(
                    [
                        t
                        for t in eval_times
                        if np.any(np.asarray(y_test["event"], dtype=bool) & (np.asarray(y_test["time"], dtype=float) <= t))
                        and np.any(np.asarray(y_test["time"], dtype=float) > t)
                    ],
                    dtype=float,
                )
                if len(auc_times):
                    aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, auc_times)
                    auc_by_time = {str(float(t)): float(a) for t, a in zip(auc_times, aucs)}
                    mean_dynamic_auc = float(mean_auc)
                    mean_dynamic_auc_ci = bootstrap_ci(
                        lambda indices: float(cumulative_dynamic_auc(y_train, y_test[indices], risk[indices], auc_times)[1]),
                        len(test_rows),
                        n_boot=n_boot,
                    )
                if len(eval_times):
                    surv_funcs = pipeline.predict_survival_function(x_test)
                    surv_matrix = np.vstack([fn(eval_times) for fn in surv_funcs])
                    _, brier_values = brier_score(y_train, y_test, surv_matrix, eval_times)
                    brier_by_time = {str(float(t)): float(v) for t, v in zip(eval_times, brier_values)}
                    if len(eval_times) >= 2:
                        ibs = float(integrated_brier_score(y_train, y_test, surv_matrix, eval_times))
                        ibs_ci = bootstrap_ci(
                            lambda indices: float(integrated_brier_score(y_train, y_test[indices], surv_matrix[indices, :], eval_times)),
                            len(test_rows),
                            n_boot=n_boot,
                        )
                    for horizon in requested_times:
                        key = str(float(horizon))
                        if horizon >= max_allowed:
                            fixed_by_horizon[key] = {
                                "status": "NOT_EVALUABLE_TIME_EXCEEDS_TRAIN_OR_TEST_RANGE",
                                "resolved_n": 0,
                            }
                            continue
                        fixed_surv = np.asarray([fn(float(horizon)) for fn in surv_funcs], dtype=float)
                        fixed_by_horizon[key] = fixed_horizon_metrics(float(horizon), y_test, test_rows, 1.0 - fixed_surv)
                cindex_ci = bootstrap_ci(
                    lambda indices: float(
                        concordance_index_censored(y_test["event"][indices], y_test["time"][indices], risk[indices])[0]
                    ),
                    len(test_rows),
                    n_boot=n_boot,
                )
                fold_metrics.append(
                    {
                        "fold": fold["fold"],
                        "heldout": fold.get("heldout", f"{split_name}_fold_{fold['fold']}"),
                        "split": split_name,
                        "status": "OK",
                        "n_train": len(train_rows),
                        "n_test": len(test_rows),
                        "train_events": int(np.sum(y_train["event"])),
                        "test_events": int(np.sum(y_test["event"])),
                        "feature_count": len(features),
                        "alpha_selection": alpha_selection,
                        "alpha": selected_alpha,
                        "c_index": cindex,
                        "c_index_bootstrap_95ci": cindex_ci,
                        "time_dependent_auc": auc_by_time,
                        "mean_dynamic_auc": mean_dynamic_auc,
                        "mean_dynamic_auc_bootstrap_95ci": mean_dynamic_auc_ci,
                        "integrated_brier_score": ibs,
                        "integrated_brier_score_bootstrap_95ci": ibs_ci,
                        "brier_by_time": brier_by_time,
                        "fixed_horizon_metrics": fixed_by_horizon,
                        "requested_evaluation_times_days": [float(v) for v in requested_times],
                        "evaluable_brier_times_days": [float(v) for v in eval_times],
                        "evaluable_auc_times_days": [float(v) for v in auc_times],
                        "not_evaluable_times_days": [float(v) for v in requested_times if v not in set(eval_times)],
                        "fixed_4_5y_resolved_n": fixed_by_horizon.get(str(float(requested_times[-1])), {}).get("resolved_n"),
                        "fixed_4_5y_brier": fixed_by_horizon.get(str(float(requested_times[-1])), {}).get("brier"),
                        "fixed_4_5y_ece": fixed_by_horizon.get(str(float(requested_times[-1])), {}).get("ece"),
                        "fixed_4_5y_calibration": fixed_by_horizon.get(str(float(requested_times[-1])), {}).get(
                            "calibration", {"slope": None, "intercept": None}
                        ),
                    }
                )
            except Exception as exc:
                fold_metrics.append(
                    {
                        "fold": fold["fold"],
                        "heldout": fold.get("heldout", f"{split_name}_fold_{fold['fold']}"),
                        "split": split_name,
                        "status": "FAILED",
                        "error": str(exc),
                        "feature_count": len(features),
                        "alpha_selection": alpha_selection,
                    }
                )
        return {
            "features": features,
            "default_alpha": default_alpha,
            "folds": fold_metrics,
            "aggregate": aggregate_folds(fold_metrics),
        }

    results: dict[str, Any] = {
        "status": "COMPLETED",
        "primary_metrics_valid": True,
        "library": "scikit-survival",
        "library_required": "scikit-survival==0.27.0",
        "model_family": "ridge_cox_ph",
        "nested_alpha_selection": {
            "enabled": True,
            "candidate_alphas": alpha_grid,
            "inner_cv_folds_requested": inner_cv_folds,
            "selection_metric": "inner_cv_c_index",
        },
        "requested_evaluation_times_days": [float(v) for v in requested_times],
        "validation_splits": {},
        "variants": {},
    }

    for variant_name, spec in variants.items():
        primary = evaluate_folds(split_artifact.get("primary_folds", []), "primary_leave_treatment_arm_out", variant_name, spec)
        secondary = evaluate_folds(split_artifact.get("secondary_folds", []), "secondary_pooled_5fold_subject_level", variant_name, spec)
        results["variants"][variant_name] = primary
        results["validation_splits"].setdefault("primary_leave_treatment_arm_out", {})[variant_name] = primary
        results["validation_splits"].setdefault("secondary_pooled_5fold_subject_level", {})[variant_name] = secondary
    return results


def confounding_audit(rows: list[dict[str, Any]], region_features: list[str]) -> dict[str, Any]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return {"status": "SKIPPED_DEPENDENCY_MISSING", "error": str(exc)}

    if not rows:
        return {"status": "NO_ROWS"}
    labels = np.array([1 if str(row["cohort"]) == "A4_treatment" else 0 for row in rows])
    if len(np.unique(labels)) < 2 or min(Counter(labels).values()) < 5:
        return {"status": "NOT_MEASURABLE_SINGLE_COHORT_OR_SMALL_CLASS"}
    features = ["global_tau_persi_mean", "global_tau_crus_mean", *region_features]
    x = pd.DataFrame([{feature: safe_float(row.get(feature)) for feature in features} for row in rows])
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=19)),
        ]
    )
    n_splits = min(5, min(Counter(labels).values()))
    probs = cross_val_predict(model, x, labels, cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=19), method="predict_proba")[:, 1]
    auc = float(roc_auc_score(labels, probs))
    return {
        "status": "COMPLETED",
        "probe_target": "A4_treatment_vs_A4_placebo",
        "probe_feature_set": "tau_suvr_global_plus_regional",
        "auc": auc,
        "gate_ceiling": 0.65,
        "gate_passed": auc <= 0.65,
        "interpretation": "AUC above 0.65 means treatment-arm identity is recoverable from tau features and image/deep claims must be downgraded.",
    }


def build_source_inventory(repo_root: Path, data_root: Path, config: dict[str, Any], tau_audit: dict[str, Any]) -> dict[str, Any]:
    ds = config["data_sources"]
    files = {
        "tau_suvr_csv": data_root / ds["tau_suvr_csv"],
        "cdr_csv": data_root / ds["cdr_csv"],
        "pacc_csv": data_root / ds["pacc_csv"],
        "subjinfo_csv": data_root / ds["subjinfo_csv"],
        "clinical_datadic_csv": data_root / ds["clinical_datadic_csv"],
        "derived_datadic_csv": data_root / ds["derived_datadic_csv"],
    }
    inventory = {
        "status": "COMPLETED",
        "data_root_role": "read_only_raw_source",
        "repo_root": str(repo_root).replace("\\", "/"),
        "raw_paths_exported": False,
        "raw_identifiers_exported": False,
        "files": {name: count_csv(path) for name, path in files.items()},
        "ftp_image_dir_exists": (data_root / ds["ftp_image_dir"]).exists(),
        "tau_suvr_audit": tau_audit,
        "amyloid_method_y_boundaries": {
            "non_fbp_decision": load_json_if_exists(repo_root / ds["amyloid_non_fbp_decision"]),
            "s1_protocol_v2": load_json_if_exists(repo_root / ds["amyloid_s1_protocol_v2"]),
        },
    }
    return inventory


def audit_death_disposition(subj_rows: list[dict[str, str]], data_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    ds = config["data_sources"]
    candidate_pattern = re.compile(
        r"(death|deceas|died|mort|discont|withdraw|terminat|status|exit|lost|drop|complete|reason|consent|screen)",
        re.IGNORECASE,
    )
    header = list(subj_rows[0].keys()) if subj_rows else []
    subj_candidates = [col for col in header if candidate_pattern.search(col)]
    column_summaries: dict[str, Any] = {}
    for col in subj_candidates:
        values = [norm(row.get(col)) for row in subj_rows if norm(row.get(col))]
        column_summaries[col] = {
            "nonmissing": len(values),
            "unique_values": len(set(values)),
            "top_values": dict(Counter(values).most_common(10)),
        }

    datadic_hits: dict[str, Any] = {}
    for key in ("clinical_datadic_csv", "derived_datadic_csv"):
        path = data_root / ds[key]
        hits = []
        if path.exists():
            for row in read_csv_rows(path):
                text = " ".join(norm(v) for v in row.values())
                if candidate_pattern.search(text):
                    hits.append({k: norm(v) for k, v in row.items() if k and norm(v)})
        datadic_hits[key] = {
            "exists": path.exists(),
            "candidate_rows": len(hits),
            "sample_without_raw_ids": hits[:25],
        }

    return {
        "status": "COMPLETED_PROVENANCE_NOT_FROZEN_FOR_CENSORING",
        "censoring_model_used": "right_censor_at_followup_window_or_last_post_index_cdr_visit",
        "death_or_competing_risk_used_in_labels": False,
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
        "subjinfo_candidate_columns": column_summaries,
        "dictionary_candidate_rows": datadic_hits,
        "decision": "Do not alter censoring with death/disposition until a human-reviewed provenance rule maps explicit death or study-exit fields to post-index follow-up.",
    }


def audit_scanner_site_linkage(sidecar_audit: dict[str, Any]) -> dict[str, Any]:
    non_ftp_hits = sidecar_audit.get("non_ftp_term_hits", {})
    mixed_tracer_terms = {k: v for k, v in non_ftp_hits.items() if v}
    status = "BLOCKED_MIXED_TRACER_SIDECARS_NOT_LINKED_TO_SELECTED_SCANS" if mixed_tracer_terms else "AGGREGATE_QC_PASSED_LINKAGE_STILL_REQUIRED"
    return {
        "status": status,
        "manufacturer_counts_aggregate_only": sidecar_audit.get("manufacturer_counts", {}),
        "mixed_non_ftp_terms": mixed_tracer_terms,
        "scanner_site_holdout_status": "NOT_RUN_SELECTED_SCAN_LINKAGE_REQUIRED",
        "per_selected_scan_sidecar_linkage": False,
        "ftp_only_image_manifest_status": "NOT_BUILT",
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
        "decision": "Image cache, CNN, and scanner/site holdout remain blocked until selected analyzed FTP scans are linked to FTP-only sidecars.",
    }


def tau_added_value_audit(survival_metrics: dict[str, Any], source_inventory: dict[str, Any]) -> dict[str, Any]:
    comparisons = []
    split_results = survival_metrics.get("validation_splits", {})
    pairs = [
        ("regional_tau_clinical", "clinical_cognition", "tau_plus_clinical_vs_clinical_cognition"),
        ("regional_tau_clinical", "clinical_only", "tau_plus_clinical_vs_clinical_only"),
        ("tau_global", "clinical_only", "global_tau_vs_clinical_only"),
        ("regional_tau", "clinical_only", "regional_tau_vs_clinical_only"),
    ]
    for split_name, variants in split_results.items():
        for tau_name, base_name, comparison_name in pairs:
            tau_item = variants.get(tau_name, {})
            base_item = variants.get(base_name, {})
            tau_agg = tau_item.get("aggregate", {})
            base_agg = base_item.get("aggregate", {})
            tau_c = tau_agg.get("mean_c_index")
            base_c = base_agg.get("mean_c_index")
            tau_b = tau_agg.get("mean_integrated_brier_score")
            base_b = base_agg.get("mean_integrated_brier_score")
            delta_c = float(tau_c - base_c) if tau_c is not None and base_c is not None else None
            delta_b = float(tau_b - base_b) if tau_b is not None and base_b is not None else None
            comparisons.append(
                {
                    "split": split_name,
                    "comparison": comparison_name,
                    "tau_variant": tau_name,
                    "baseline_variant": base_name,
                    "delta_c_index": delta_c,
                    "delta_integrated_brier_score": delta_b,
                    "interpretation": "positive delta_c_index and negative delta_integrated_brier_score favor tau variant",
                }
            )
    primary = next(
        (
            item
            for item in comparisons
            if item["split"] == "primary_leave_treatment_arm_out"
            and item["comparison"] == "tau_plus_clinical_vs_clinical_cognition"
        ),
        None,
    )
    demonstrated = bool(
        primary
        and primary.get("delta_c_index") is not None
        and primary.get("delta_integrated_brier_score") is not None
        and float(primary["delta_c_index"]) > 0.0
        and float(primary["delta_integrated_brier_score"]) < 0.0
    )
    tau_audit = source_inventory.get("tau_suvr_audit", {})
    return {
        "status": "TAU_INCREMENTAL_VALUE_DEMONSTRATED" if demonstrated else "TAU_INCREMENTAL_VALUE_NOT_DEMONSTRATED",
        "primary_decision_basis": primary,
        "comparisons": comparisons,
        "regional_policy_status": "RAW_REGION_LEVEL_NO_FROZEN_BRAAK_OR_META_ROI_POLICY"
        if not tau_audit.get("has_explicit_braak_region_names") and not tau_audit.get("has_explicit_meta_roi_names")
        else "EXPLICIT_REGION_POLICY_AVAILABLE",
        "suvr_sensitivity_status": "GLOBAL_CRUS_MEAN_INCLUDED_SCALAR_SENSITIVITY_ONLY",
        "claim_decision": "Do not claim tau PET adds predictive value over clinical+cognition unless the added-value audit passes.",
    }


def calibration_report(survival_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for split_name, variants in survival_metrics.get("validation_splits", {}).items():
        for variant_name, item in variants.items():
            for fold in item.get("folds", []):
                for horizon, metrics in fold.get("fixed_horizon_metrics", {}).items():
                    rows.append(
                        {
                            "split": split_name,
                            "variant": variant_name,
                            "fold": fold.get("fold"),
                            "heldout": fold.get("heldout"),
                            "horizon_days": horizon,
                            "status": metrics.get("status"),
                            "resolved_n": metrics.get("resolved_n"),
                            "event_n": metrics.get("event_n"),
                            "brier": metrics.get("brier"),
                            "ece": metrics.get("ece"),
                            "calibration_slope": metrics.get("calibration", {}).get("slope") if isinstance(metrics.get("calibration"), dict) else None,
                            "calibration_intercept": metrics.get("calibration", {}).get("intercept") if isinstance(metrics.get("calibration"), dict) else None,
                        }
                    )
    completed = [row for row in rows if row.get("status") == "OK"]
    return {
        "status": "COMPLETED" if completed else "PARTIAL_NO_COMPLETED_FIXED_HORIZON_CALIBRATION",
        "rows": rows,
        "limitations": [
            "Rows censored before a fixed horizon are excluded from fixed-horizon binary calibration.",
            "Calibration slope/intercept are null when resolved labels do not contain both classes.",
        ],
    }


def build_stage_gate_summary(
    endpoint_audit: dict[str, Any],
    split_artifact: dict[str, Any],
    sidecar_audit: dict[str, Any],
    scanner_site_audit: dict[str, Any],
    death_audit: dict[str, Any],
    survival_metrics: dict[str, Any],
    confound: dict[str, Any],
    added_value: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    stop = endpoint_audit.get("stop_conditions", {})
    s1_core_pass = bool(
        endpoint_audit.get("status") == "COMPLETED"
        and stop.get("total_event_gate_passed")
        and stop.get("heldout_arm_event_gate_passed")
        and split_artifact.get("subject_level_only")
        and endpoint_audit.get("event_flag_columns_used_as_predictors") is False
        and confound.get("gate_passed") is True
    )
    secondary = survival_metrics.get("validation_splits", {}).get("secondary_pooled_5fold_subject_level", {})
    secondary_completed = bool(secondary) and all(item.get("aggregate", {}).get("folds_ok", 0) >= 2 for item in secondary.values())
    primary_completed = survival_metrics.get("primary_metrics_valid") is True
    s3_blocked = str(sidecar_audit.get("image_cache_status", "")).startswith("IMAGE_CACHE_BLOCKED") or str(
        scanner_site_audit.get("status", "")
    ).startswith("BLOCKED")
    s5_partial_reasons = [
        "public_bundle_not_updated_by_this_generator",
        "full_repo_test_suite_not_run_by_this_generator",
    ]
    return {
        "overall_status": "PARTIAL_TAU_S1_S2_UPDATED_S3_BLOCKED_S5_PARTIAL",
        "S1_tau": {
            "status": "PASS_WITH_CENSORING_AND_SCANNER_LINKAGE_LIMITATIONS" if s1_core_pass else "BLOCKED",
            "endpoint_event_floor_passed": bool(stop.get("total_event_gate_passed") and stop.get("heldout_arm_event_gate_passed")),
            "subject_level_split": bool(split_artifact.get("subject_level_only")),
            "forbidden_label_sources_not_used": endpoint_audit.get("forbidden_label_sources_not_used", []),
            "treatment_probe_passed": confound.get("gate_passed"),
            "scanner_site_linkage_status": scanner_site_audit.get("status"),
            "death_disposition_status": death_audit.get("status"),
        },
        "S2_tau": {
            "status": "COMPLETED" if primary_completed and secondary_completed else "PARTIAL",
            "primary_leave_treatment_arm_out_completed": primary_completed,
            "secondary_pooled_5fold_completed": secondary_completed,
            "nested_alpha_selection": survival_metrics.get("nested_alpha_selection"),
        },
        "S3_tau": {
            "status": "BLOCKED_PENDING_SIDECAR_QC" if s3_blocked else "READY_FOR_IMAGE_CACHE_BUILD",
            "image_cache_status": sidecar_audit.get("image_cache_status"),
            "per_selected_scan_sidecar_linkage": scanner_site_audit.get("per_selected_scan_sidecar_linkage"),
            "ftp_only_image_manifest_status": scanner_site_audit.get("ftp_only_image_manifest_status"),
        },
        "S4_tau": {
            "status": added_value.get("status"),
            "regional_policy_status": added_value.get("regional_policy_status"),
            "claim_decision": added_value.get("claim_decision"),
        },
        "S5_tau": {
            "status": "PARTIAL",
            "calibration_report_status": calibration.get("status"),
            "model_card_generated": True,
            "tripod_ai_summary_generated": True,
            "remaining_reasons": s5_partial_reasons,
        },
    }


def write_claim_boundary(path: Path, config: dict[str, Any], endpoint_audit: dict[str, Any], confound: dict[str, Any]) -> None:
    allowed = config["claim_wording"]["allowed"]
    lines = [
        "# Tau-AD Progression Claim Boundary\n\n",
        "## Allowed Claim\n\n",
        f"{allowed}\n\n",
        "## Required Limitations\n\n",
    ]
    for item in config["claim_wording"]["required_limitations"]:
        lines.append(f"- {item}\n")
    lines.extend(
        [
            "\n## Repo-Truth Constraints\n\n",
            f"- eligible_rows: `{endpoint_audit.get('eligible_rows')}`\n",
            f"- sustained_events: `{endpoint_audit.get('events')}`\n",
            f"- cohort_counts: `{endpoint_audit.get('cohort_counts')}`\n",
            f"- treatment_arm_probe_auc: `{confound.get('auc')}`\n",
            "- amyloid Method Y S1 artifacts are not overwritten by this branch.\n",
        ]
    )
    atomic_write_text(path, "".join(lines))


def write_model_card(path: Path, report: dict[str, Any], added_value: dict[str, Any], stage_gates: dict[str, Any]) -> None:
    endpoint = report["endpoint_audit"]
    metrics = report["survival_metrics"]
    variants = metrics.get("variants", {})
    best_name = None
    best_c = -1.0
    for name, item in variants.items():
        c_index = item.get("aggregate", {}).get("mean_c_index")
        if c_index is not None and float(c_index) > best_c:
            best_name = name
            best_c = float(c_index)
    lines = [
        "# Method Y Tau-Progression Model Card\n\n",
        "## Intended Use\n\n",
        report["final_allowed_claim"] + "\n\n",
        "This branch estimates progression risk; it is not an Alzheimer disease diagnostic tool.\n\n",
        "## Data And Endpoint\n\n",
        f"- Eligible subjects: `{endpoint.get('eligible_rows')}`\n",
        f"- Sustained CDR-Global progression events: `{endpoint.get('events')}`\n",
        f"- Censoring: `{endpoint.get('censored')}`\n",
        f"- Endpoint: `{endpoint.get('event_definition')}`\n\n",
        "## Current Model Selection\n\n",
        f"- Best primary split variant by mean C-index: `{best_name}`\n",
        f"- Best primary split mean C-index: `{best_c if best_name else None}`\n",
        f"- Tau added-value status: `{added_value.get('status')}`\n\n",
        "## Stage Gates\n\n",
    ]
    for stage, item in stage_gates.items():
        if stage == "overall_status":
            continue
        lines.append(f"- `{stage}`: `{item.get('status')}`\n")
    lines.extend(
        [
            "\n## Forbidden Claims\n\n",
            "- Do not claim Alzheimer disease diagnosis.\n",
            "- Do not claim amyloid cross-tracer validation.\n",
            "- Do not claim FTP/flortaucipir resolves Method Y amyloid S1.\n",
            "- Do not claim LEARN tau-PET validation until analyzed LEARN FTP is verified.\n",
        ]
    )
    atomic_write_text(path, "".join(lines))


def write_tripod_summary(path: Path, report: dict[str, Any], calibration: dict[str, Any], added_value: dict[str, Any]) -> None:
    endpoint = report["endpoint_audit"]
    lines = [
        "# TRIPOD-AI Style Summary: Method Y Tau-Progression\n\n",
        "## Prediction Target\n\n",
        "Risk of AD-related clinical/cognitive progression, operationalized as sustained CDR-Global progression after baseline FTP/tau PET.\n\n",
        "## Population\n\n",
        f"Cognitively unimpaired A4 tau-PET participants with baseline CDR-Global 0, linked baseline clinical packet, and post-index CDR follow-up. Eligible rows: `{endpoint.get('eligible_rows')}`.\n\n",
        "## Predictors\n\n",
        "Baseline non-future clinical/cognitive covariates and baseline FTP tau SUVR features. Raw subject identifiers, raw paths, future outcomes, follow-up length, site identifiers, and amyloid PET images are forbidden as model inputs.\n\n",
        "## Validation\n\n",
        "Primary validation is leave-treatment-arm-out. Secondary validation is pooled subject-level 5-fold CV. LEARN tau-PET validation is blocked pending analyzed LEARN FTP evidence.\n\n",
        "## Performance And Calibration\n\n",
        f"Survival metrics status: `{report['survival_metrics'].get('status')}`. Calibration report status: `{calibration.get('status')}`.\n\n",
        "## Added Value Of Tau\n\n",
        f"`{added_value.get('status')}`. The final claim must not state tau improves prediction unless this gate passes.\n\n",
        "## Limitations\n\n",
        "- Image/CNN modeling remains blocked pending per-selected-scan sidecar QC.\n",
        "- Death/disposition competing-risk provenance is not frozen for censoring.\n",
        "- Braak/meta-ROI policy is not frozen; current tau modeling uses raw region-level SUVR features.\n",
        "- This is not external validation and not an Alzheimer disease diagnosis model.\n",
    ]
    atomic_write_text(path, "".join(lines))


def report_markdown(report: dict[str, Any]) -> str:
    endpoint = report["endpoint_audit"]
    metrics = report["survival_metrics"]
    stage_gates = report.get("stage_gates", {})
    added_value = report.get("tau_added_value_audit", {})
    lines = [
        "# Tau-AD Progression Report v1\n\n",
        f"- status: `{report['status']}`\n",
        f"- eligible_rows: `{endpoint.get('eligible_rows')}`\n",
        f"- sustained_events: `{endpoint.get('events')}`\n",
        f"- cohort_counts: `{endpoint.get('cohort_counts')}`\n",
        f"- event_counts_by_cohort: `{endpoint.get('event_counts_by_cohort')}`\n",
        f"- LEARN tau validation: `{report['learn_tau_validation_status']}`\n",
        f"- image/CNN status: `{report['image_model_status']}`\n\n",
        "## Primary Survival Models\n\n",
        "| variant | folds_ok | mean_c_index | mean_integrated_brier_score |\n",
        "|---|---:|---:|---:|\n",
    ]
    for name, item in metrics.get("variants", {}).items():
        agg = item.get("aggregate", {})
        lines.append(
            f"| {name} | {agg.get('folds_ok')} | {agg.get('mean_c_index')} | {agg.get('mean_integrated_brier_score')} |\n"
        )
    lines.extend(["\n## Stage Gates\n\n", "| stage | status |\n", "|---|---|\n"])
    for stage, item in stage_gates.items():
        if stage == "overall_status":
            continue
        lines.append(f"| {stage} | {item.get('status')} |\n")
    lines.extend(
        [
            "\n## Tau Added Value\n\n",
            f"- status: `{added_value.get('status')}`\n",
            f"- claim decision: `{added_value.get('claim_decision')}`\n",
        ]
    )
    lines.extend(
        [
            "\n## Claim Boundary\n\n",
            report["final_allowed_claim"] + "\n\n",
            "This workflow is not a diagnostic tool for Alzheimer disease, does not validate amyloid cross-tracer performance, and does not resolve the blocked Method Y amyloid S1 gate.\n",
        ]
    )
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Method Y tau-progression protocol, labels, splits, audits, and Cox baselines.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--data_root", default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = Path(args.data_root or config["data_sources"]["data_root"])
    artifacts = {name: repo_root / rel for name, rel in config["artifacts"].items() if name != "root"}
    artifact_root = repo_root / config["artifacts"]["root"]
    artifact_root.mkdir(parents=True, exist_ok=True)

    ds = config["data_sources"]
    tau_path = data_root / ds["tau_suvr_csv"]
    cdr_path = data_root / ds["cdr_csv"]
    pacc_path = data_root / ds["pacc_csv"]
    subj_path = data_root / ds["subjinfo_csv"]

    first_tau, tau_audit = load_first_tau_scans(tau_path, config)
    subj_rows = read_csv_rows(subj_path)
    cdr_rows = read_csv_rows(cdr_path)
    pacc_rows = read_csv_rows(pacc_path)
    label_rows, endpoint_audit, region_features = build_label_rows(first_tau, subj_rows, cdr_rows, pacc_rows, config)
    split_artifact = build_splits(label_rows, config)
    source_inventory = build_source_inventory(repo_root, data_root, config, tau_audit)
    sidecar_audit = audit_ftp_sidecars(data_root / ds["ftp_image_dir"])
    scanner_site_audit = audit_scanner_site_linkage(sidecar_audit)
    death_audit = audit_death_disposition(subj_rows, data_root, config)
    survival_metrics = run_survival_models(label_rows, region_features, split_artifact, config)
    confound = confounding_audit(label_rows, region_features)
    added_value = tau_added_value_audit(survival_metrics, source_inventory)
    calibration = calibration_report(survival_metrics)
    stage_gates = build_stage_gate_summary(
        endpoint_audit,
        split_artifact,
        sidecar_audit,
        scanner_site_audit,
        death_audit,
        survival_metrics,
        confound,
        added_value,
        calibration,
    )

    base_fields = [
        "patient_hash",
        "cohort",
        "treatment_arm",
        "baseline_gap_days_abs",
        "index_source",
        "event",
        "time_days",
        "censoring_rule",
        "followup_cdr_visits",
        "single_visit_cdglobal_ge_0_5",
        "baseline_cdglobal",
        "baseline_cdrsb",
        "age",
        "sex_binary",
        "education_years",
        "apoe4_carrier",
        "baseline_amyloid_centiloid",
        "baseline_amyloid_status_cl24",
        "baseline_pacc_raw",
        "baseline_pacc_change",
        "baseline_fctotal96",
        "baseline_ldeltotal",
        "baseline_digittotal",
        "baseline_mmscore",
        "global_tau_persi_mean",
        "global_tau_crus_mean",
        "tau_region_count",
        "uses_raw_identifier",
        "uses_raw_path",
    ]
    write_csv(artifacts["labels_csv"], label_rows, base_fields + region_features)

    protocol = {
        "schema_version": config["schema_version"],
        "protocol_name": config["protocol_name"],
        "status": config["status"],
        "scope": config["scope"],
        "endpoint": config["endpoint"],
        "inputs": config["inputs"],
        "validation": config["validation"],
        "stages": config["stages"],
        "claim_wording": config["claim_wording"],
    }
    atomic_write_json(artifacts["protocol_json"], json_safe(protocol))
    atomic_write_json(artifacts["source_inventory_json"], json_safe(source_inventory))
    atomic_write_json(artifacts["endpoint_audit_json"], json_safe(endpoint_audit))
    atomic_write_json(artifacts["sidecar_audit_json"], json_safe(sidecar_audit))
    atomic_write_json(artifacts["scanner_site_linkage_audit_json"], json_safe(scanner_site_audit))
    atomic_write_json(artifacts["death_disposition_audit_json"], json_safe(death_audit))
    atomic_write_json(artifacts["split_json"], json_safe(split_artifact))

    learn_count = endpoint_audit.get("cohort_counts", {}).get("LEARN", 0)
    image_status = sidecar_audit.get("image_cache_status", "UNKNOWN")
    report = {
        "schema_version": 1,
        "status": "COMPLETED_WITH_IMAGE_CNN_BLOCKED_PENDING_QC"
        if str(image_status).startswith("IMAGE_CACHE_BLOCKED")
        else "COMPLETED",
        "config": str(config_path).replace("\\", "/"),
        "source_inventory": str(artifacts["source_inventory_json"].relative_to(repo_root)).replace("\\", "/"),
        "labels": str(artifacts["labels_csv"].relative_to(repo_root)).replace("\\", "/"),
        "splits": str(artifacts["split_json"].relative_to(repo_root)).replace("\\", "/"),
        "endpoint_audit": endpoint_audit,
        "ftp_sidecar_audit": sidecar_audit,
        "scanner_site_linkage_audit": scanner_site_audit,
        "death_disposition_audit": death_audit,
        "confounding_audit": confound,
        "survival_metrics": survival_metrics,
        "tau_added_value_audit": added_value,
        "calibration_report": calibration,
        "stage_gates": stage_gates,
        "learn_tau_validation_status": "BLOCKED_NO_ANALYZED_LEARN_FTP_IN_CURRENT_AUDIT" if learn_count == 0 else "AVAILABLE",
        "image_model_status": image_status,
        "final_allowed_claim": config["claim_wording"]["allowed"],
        "forbidden_claims_enforced": {
            "diagnostic_tool_for_alzheimer_disease": False,
            "amyloid_cross_tracer_validation": False,
            "method_y_amyloid_s1_resolved_by_ftp": False,
        },
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
    }
    atomic_write_json(artifacts["stage_gate_summary_json"], json_safe(stage_gates))
    atomic_write_json(artifacts["tau_added_value_audit_json"], json_safe(added_value))
    atomic_write_json(artifacts["calibration_report_json"], json_safe(calibration))
    atomic_write_json(artifacts["report_json"], json_safe(report))
    atomic_write_text(artifacts["report_md"], report_markdown(report))
    write_claim_boundary(artifacts["claim_boundary_md"], config, endpoint_audit, confound)
    write_model_card(artifacts["model_card_md"], report, added_value, stage_gates)
    write_tripod_summary(artifacts["tripod_summary_md"], report, calibration, added_value)

    raw_text_scan = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in [
            artifacts["protocol_json"],
            artifacts["source_inventory_json"],
            artifacts["endpoint_audit_json"],
            artifacts["sidecar_audit_json"],
            artifacts["scanner_site_linkage_audit_json"],
            artifacts["death_disposition_audit_json"],
            artifacts["split_json"],
            artifacts["stage_gate_summary_json"],
            artifacts["tau_added_value_audit_json"],
            artifacts["calibration_report_json"],
            artifacts["report_json"],
            artifacts["report_md"],
            artifacts["claim_boundary_md"],
            artifacts["model_card_md"],
            artifacts["tripod_summary_md"],
        ]
    )
    if contains_raw_identifier(raw_text_scan):
        raise RuntimeError("Raw identifier pattern detected in generated aggregate outputs.")

    print(
        json.dumps(
            {
                "status": report["status"],
                "eligible_rows": endpoint_audit["eligible_rows"],
                "events": endpoint_audit["events"],
                "learn_tau_validation_status": report["learn_tau_validation_status"],
                "image_model_status": report["image_model_status"],
                "stage_gate_overall_status": stage_gates["overall_status"],
                "tau_added_value_status": added_value["status"],
                "report": str(artifacts["report_json"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
