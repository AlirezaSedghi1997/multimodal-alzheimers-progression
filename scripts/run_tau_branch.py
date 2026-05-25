from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_mri_features import MRI_FEATURES
from src.data.a4_schema import REQUIRED_RELATIVE_FILES, T1_NAME_PATTERN, resolve_data_root, resolve_required_file
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import redact_path, stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


TABULAR_FEATURES = ["age", "education_years", "apoe4_carrier", "mmse_base"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def norm_visit(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        text = value.strip()
        return text if text and text.upper() != "NA" else None


def apoe4_carrier(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return 1 if "4" in value else 0


def mmse_from_subject_row(row: dict[str, str]) -> float | None:
    mmse = to_float(row.get("MMSETSV6"))
    if mmse is None:
        mmse = to_float(row.get("MMSETSV1"))
    return mmse


def subject_features_from_subjinfo(path: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    if not path.exists():
        return out
    for row in read_csv_rows(path):
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        if not substudy or not bid:
            continue
        out[stable_hash(f"{substudy}|{bid}", prefix="pt")] = {
            "age": to_float(row.get("AGEYR")),
            "sex": row.get("SEX") or None,
            "education_years": to_float(row.get("EDCCNTU")),
            "apoe4_carrier": apoe4_carrier(row.get("APOEGN")),
            "mmse_base": mmse_from_subject_row(row),
        }
    return out


def subject_features_from_catalog(path: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    if not path.exists():
        return out
    for row in read_csv_rows(path):
        patient_hash = row.get("patient_hash") or ""
        if not patient_hash:
            continue
        out[patient_hash] = {
            "age": to_float(row.get("age")),
            "sex": row.get("sex") or None,
            "education_years": to_float(row.get("education_years")),
            "apoe4_carrier": to_float(row.get("apoe4_carrier")),
            "mmse_base": to_float(row.get("mmse_base")),
        }
    return out


def load_subject_features(data_root: Path) -> tuple[dict[str, dict[str, object]], str]:
    subj_path = resolve_required_file(data_root, "subjinfo")
    subj = subject_features_from_subjinfo(subj_path)
    if subj:
        return subj, "subjinfo"
    catalog_path = Path("data/processed/master_catalog.csv")
    subj = subject_features_from_catalog(catalog_path)
    if subj:
        return subj, "locked_master_catalog"
    raise FileNotFoundError(
        f"Missing subject info ({subj_path}) and fallback catalog ({catalog_path}) has no usable subject features."
    )


def visit_days(path: Path) -> dict[tuple[str, str, str], float]:
    out = {}
    for row in read_csv_rows(path):
        key = ((row.get("SUBSTUDY") or "").upper(), row.get("BID") or "", norm_visit(row.get("VISITCD")))
        if not key[0] or not key[1] or key[2] is None:
            continue
        day = to_float(row.get("SVUSEDTC_DAYS_T0"))
        if day is None:
            day = to_float(row.get("SVSTDTC_DAYS_T0"))
        if day is not None:
            out[key] = day
    return out


def t1_by_patient_hash_from_visits(
    root: Path, visits: dict[tuple[str, str, str], float]
) -> dict[str, list[dict[str, object]]]:
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    out: dict[str, list[dict[str, object]]] = defaultdict(list)
    for p in (root / "T1").rglob("*"):
        if not p.is_file() or not (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        substudy = m.group("substudy").upper()
        bid = m.group("bid")
        scan = str(int(m.group("scan")))
        day = visits.get((substudy, bid, scan))
        if day is not None:
            patient_hash = stable_hash(f"{substudy}|{bid}", prefix="pt")
            out[patient_hash].append({"day": day, "t1_file_hash": stable_hash(p.name, prefix="t1")})
    for vals in out.values():
        vals.sort(key=lambda x: float(x["day"]))
    return out


def t1_by_patient_hash_from_catalog(path: Path) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str, float]] = set()
    if not path.exists():
        return out
    for row in read_csv_rows(path):
        patient_hash = row.get("patient_hash") or ""
        t1_hash = row.get("t1_file_hash") or ""
        day = to_float(row.get("t1_day_t0"))
        if not patient_hash or not t1_hash or day is None:
            continue
        key = (patient_hash, t1_hash, day)
        if key in seen:
            continue
        seen.add(key)
        out[patient_hash].append({"day": day, "t1_file_hash": t1_hash})
    for vals in out.values():
        vals.sort(key=lambda x: float(x["day"]))
    return out


def load_t1_candidates(data_root: Path) -> tuple[dict[str, list[dict[str, object]]], str]:
    visits_path = resolve_required_file(data_root, "visits")
    if visits_path.exists():
        return t1_by_patient_hash_from_visits(data_root, visit_days(visits_path)), "sv_visit_timing"
    catalog_path = Path("data/processed/master_catalog.csv")
    candidates = t1_by_patient_hash_from_catalog(catalog_path)
    if candidates:
        return candidates, "locked_master_catalog"
    raise FileNotFoundError(
        f"Missing visits file ({visits_path}) and fallback catalog ({catalog_path}) has no usable T1 candidates."
    )


def nearest_t1(tau_day: float | None, candidates: list[dict[str, object]], max_gap: float) -> dict[str, object] | None:
    if tau_day is None or not candidates:
        return None
    best = min(candidates, key=lambda c: abs(float(c["day"]) - tau_day))
    gap = abs(float(best["day"]) - tau_day)
    if gap > max_gap:
        return None
    return {**best, "gap": gap}


def choose_tau_target_column(tau_rows: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
    candidates = ["suvr_cer", "suvr_persi", "suvr_crus"]
    coverage = {
        col: sum(
            1
            for row in tau_rows
            if (row.get("scan_analyzed") or "").strip().lower() in {"yes", "y", "1", "true"}
            and to_float(row.get(col)) is not None
        )
        for col in candidates
    }
    best = max(candidates, key=lambda col: coverage[col])
    if coverage[best] == 0:
        raise RuntimeError("No usable Tau SUVR target column found.")
    return best, coverage


def build_tau_dataset(data_root: Path, max_gap: float) -> tuple[list[dict[str, object]], str, dict[str, int], str, str]:
    t1_index, t1_source = load_t1_candidates(data_root)
    subj, subject_source = load_subject_features(data_root)
    tau_rows = read_csv_rows(resolve_required_file(data_root, "tau"))
    target_col, target_coverage = choose_tau_target_column(tau_rows)
    scans: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in tau_rows:
        if (row.get("scan_analyzed") or "").strip().lower() not in {"yes", "y", "1", "true"}:
            continue
        value = to_float(row.get(target_col))
        tau_day = to_float(row.get("scan_date_DAYS_T0"))
        if value is None or tau_day is None:
            continue
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        key = (substudy, bid, row.get("VISCODE") or "", row.get("scan_date_DAYS_T0") or "")
        rec = scans.setdefault(
            key,
            {"substudy": substudy, "bid": bid, "tau_day": tau_day, "sum": 0.0, "n": 0},
        )
        rec["sum"] = float(rec["sum"]) + value
        rec["n"] = int(rec["n"]) + 1
    rows: list[dict[str, object]] = []
    for rec in scans.values():
        patient_hash = stable_hash(f"{rec['substudy']}|{rec['bid']}", prefix="pt")
        nearest = nearest_t1(float(rec["tau_day"]), t1_index.get(patient_hash, []), max_gap)
        if nearest is None:
            continue
        srow = subj.get(patient_hash, {})
        target = float(rec["sum"]) / int(rec["n"])
        rows.append(
            {
                "tau_sample_hash": stable_hash(f"{rec['substudy']}|{rec['bid']}|{rec['tau_day']}", prefix="tau"),
                "patient_hash": patient_hash,
                "substudy": rec["substudy"],
                "tau_target_value": target,
                "tau_pet_day_t0": rec["tau_day"],
                "t1_tau_gap_days": nearest["gap"],
                "t1_file_hash": nearest["t1_file_hash"],
                "age": srow.get("age"),
                "sex": srow.get("sex"),
                "education_years": srow.get("education_years"),
                "apoe4_carrier": srow.get("apoe4_carrier"),
                "mmse_base": srow.get("mmse_base"),
                "tau_n_rois_aggregated": rec["n"],
            }
        )
    return rows, target_col, target_coverage, t1_source, subject_source


def read_mri_features(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["t1_file_hash"]: row for row in read_csv_rows(path) if row.get("mri_feature_status") == "OK"}


def safe_float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def feature_matrix(
    rows: list[dict[str, object]], mri: dict[str, dict[str, str]]
) -> tuple[list[list[float]], list[float], list[str]]:
    x = []
    y = []
    groups = []
    for row in rows:
        frow = mri.get(str(row["t1_file_hash"]))
        if frow is None:
            continue
        sex = row.get("sex", "")
        sex_val = 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0
        vals = [safe_float(row.get(f)) for f in TABULAR_FEATURES] + [sex_val]
        vals.extend(safe_float(frow.get(f)) for f in MRI_FEATURES)
        x.append(vals)
        y.append(safe_float(row["tau_target_value"]))
        groups.append(str(row["patient_hash"]))
    return x, y, groups


def summarize_values(values: list[float]) -> dict[str, float]:
    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
    else:
        variance = 0.0

    def quantile(p: float) -> float:
        if n == 1:
            return vals[0]
        pos = (n - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return vals[lo]
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": vals[0],
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "p75": quantile(0.75),
        "max": vals[-1],
    }


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    n = len(y_true)
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    mean_y = sum(y_true) / n
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"n": n, "rmse": math.sqrt(mse), "mae": mae, "r2": r2}


def write_tau_skip(out_dir: Path, data_root: Path, missing_path: Path, max_gap_days: float) -> None:
    reason = "Tau target file is unavailable in the mounted data root; exploratory tau regression skipped."
    summary = {
        "schema_version": 2,
        "mode": "SKIP_WITH_JUSTIFICATION",
        "target": None,
        "target_column": None,
        "target_coverage": {},
        "t1_candidate_source": None,
        "subject_feature_source": None,
        "target_summary": {"n": 0},
        "rows_aligned": 0,
        "rows_used": 0,
        "rows_missing_mri_features": 0,
        "mri_feature_join_rate": 0.0,
        "unique_patients_used": 0,
        "max_gap_days": max_gap_days,
        "cv_strategy": None,
        "mean_baseline": None,
        "mean_baseline_folds": [],
        "aggregate": {"n": 0, "rmse": None, "mae": None, "r2": None},
        "folds": [],
        "delta_vs_mean_baseline": {"rmse_reduction": None, "r2_delta": None},
        "uses_tau_as_input": False,
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "skip_reason": reason,
        "missing_required_files": [redact_path(missing_path)],
        "interpretation": "Exploratory secondary analysis; does not alter primary AB pipeline.",
        "interpretation_detail": reason,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "tau_regression_metrics.json", summary)
    lines = [
        "# Tau Regression Metrics\n\n",
        "- mode: `SKIP_WITH_JUSTIFICATION`\n",
        f"- skip_reason: `{reason}`\n",
        f"- missing_required_files: `{summary['missing_required_files']}`\n",
        "- uses_tau_as_input: `False`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
        "- interpretation: `Exploratory secondary analysis; does not alter primary AB pipeline.`\n",
    ]
    atomic_write_text(out_dir / "tau_regression_metrics.md", "".join(lines))
    StateManager().append_event("tau", "SKIPPED_WITH_JUSTIFICATION", reason)
    write_run_manifest(
        "results/logs/tau_manifest.json",
        stage="tau",
        data_root=str(data_root),
        extra={"mode": "SKIP_WITH_JUSTIFICATION", "schema_version": 2, "missing": summary["missing_required_files"]},
    )
    print(json.dumps({"mode": "SKIP_WITH_JUSTIFICATION", "skip_reason": reason}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--mri_features", default="data/processed/mri_features.csv")
    ap.add_argument("--out_dir", default="results/final_results/tau")
    ap.add_argument("--max_gap_days", type=float, default=120.0)
    args = ap.parse_args()

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GroupKFold
    except Exception as exc:
        raise RuntimeError("run_tau_branch requires scikit-learn") from exc

    data_root = resolve_data_root(args.data_root)
    tau_path = resolve_required_file(data_root, "tau")
    if not tau_path.exists():
        write_tau_skip(Path(args.out_dir), data_root, tau_path, args.max_gap_days)
        return 0
    rows, target_col, target_coverage, t1_source, subject_source = build_tau_dataset(data_root, args.max_gap_days)
    mri = read_mri_features(Path(args.mri_features))
    x, y, groups = feature_matrix(rows, mri)
    if len(y) < 200:
        raise RuntimeError("Tau FULL mode requires at least 200 usable rows.")
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 3:
        raise RuntimeError("Tau grouped CV requires at least 3 unique patients.")
    n_splits = min(3, len(unique_groups))
    kf = GroupKFold(n_splits=n_splits)
    preds = [0.0 for _ in y]
    baseline_preds = [0.0 for _ in y]
    folds = []
    baseline_folds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(x, y, groups)):
        model = RandomForestRegressor(n_estimators=120, min_samples_leaf=5, random_state=53 + fold, n_jobs=-1)
        x_train = [x[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        x_test = [x[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        mean_train = sum(y_train) / len(y_train)
        mean_pred = [mean_train for _ in test_idx]
        for idx, val in zip(test_idx, mean_pred):
            baseline_preds[idx] = val
        model.fit(x_train, y_train)
        p = [float(v) for v in model.predict(x_test)]
        for idx, val in zip(test_idx, p):
            preds[idx] = val
        baseline_folds.append({"fold": fold, **regression_metrics(y_test, mean_pred)})
        folds.append({"fold": fold, **regression_metrics(y_test, p)})
    aggregate = regression_metrics(y, preds)
    mean_baseline = regression_metrics(y, baseline_preds)
    rmse_delta = mean_baseline["rmse"] - aggregate["rmse"]
    r2_delta = aggregate["r2"] - mean_baseline["r2"]
    mri_join_rate = len(y) / len(rows) if rows else 0.0
    interpretation_detail = (
        "No out-of-fold predictive signal over the grouped mean baseline; report as a negative exploratory result."
        if aggregate["r2"] <= 0 or rmse_delta <= 0
        else "Exploratory signal exceeds the grouped mean baseline; still secondary to the primary AB pipeline."
    )
    summary = {
        "schema_version": 2,
        "mode": "FULL",
        "target": f"tau_mean_{target_col}_across_regions",
        "target_column": target_col,
        "target_coverage": target_coverage,
        "t1_candidate_source": t1_source,
        "subject_feature_source": subject_source,
        "target_summary": summarize_values(y),
        "rows_aligned": len(rows),
        "rows_used": len(y),
        "rows_missing_mri_features": len(rows) - len(y),
        "mri_feature_join_rate": mri_join_rate,
        "unique_patients_used": len(unique_groups),
        "max_gap_days": args.max_gap_days,
        "cv_strategy": f"GroupKFold(n_splits={n_splits}, group=patient_hash)",
        "mean_baseline": mean_baseline,
        "mean_baseline_folds": baseline_folds,
        "aggregate": aggregate,
        "folds": folds,
        "delta_vs_mean_baseline": {"rmse_reduction": rmse_delta, "r2_delta": r2_delta},
        "uses_tau_as_input": False,
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
        "interpretation": "Exploratory secondary analysis; does not alter primary AB pipeline.",
        "interpretation_detail": interpretation_detail,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "tau_regression_metrics.json", summary)
    lines = [
        "# Tau Regression Metrics\n\n",
        f"- mode: `{summary['mode']}`\n",
        f"- target: `{summary['target']}`\n",
        f"- t1_candidate_source: `{summary['t1_candidate_source']}`\n",
        f"- subject_feature_source: `{summary['subject_feature_source']}`\n",
        f"- rows_aligned: `{summary['rows_aligned']}`\n",
        f"- rows_used: `{summary['rows_used']}`\n",
        f"- rows_missing_mri_features: `{summary['rows_missing_mri_features']}`\n",
        f"- mri_feature_join_rate: `{summary['mri_feature_join_rate']:.4f}`\n",
        f"- unique_patients_used: `{summary['unique_patients_used']}`\n",
        f"- cv_strategy: `{summary['cv_strategy']}`\n",
        f"- r2: `{aggregate['r2']:.4f}`\n",
        f"- rmse: `{aggregate['rmse']:.4f}`\n",
        f"- mae: `{aggregate['mae']:.4f}`\n",
        f"- mean_baseline_r2: `{mean_baseline['r2']:.4f}`\n",
        f"- mean_baseline_rmse: `{mean_baseline['rmse']:.4f}`\n",
        f"- rmse_reduction_vs_mean_baseline: `{rmse_delta:.4f}`\n",
        "- uses_tau_as_input: `False`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
        "- interpretation: `Exploratory secondary analysis; does not alter primary AB pipeline.`\n",
        f"- interpretation_detail: `{interpretation_detail}`\n",
    ]
    atomic_write_text(out_dir / "tau_regression_metrics.md", "".join(lines))
    StateManager().append_event("tau", "COMPLETED", "Exploratory tau regression completed.")
    write_run_manifest(
        "results/logs/tau_manifest.json",
        stage="tau",
        data_root=str(data_root),
        extra={"aggregate": aggregate, "mean_baseline": mean_baseline, "schema_version": 2},
    )
    print(
        json.dumps(
            {
                "rows_used": len(y),
                "r2": aggregate["r2"],
                "rmse": aggregate["rmse"],
                "mae": aggregate["mae"],
                "mean_baseline_r2": mean_baseline["r2"],
                "mean_baseline_rmse": mean_baseline["rmse"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
