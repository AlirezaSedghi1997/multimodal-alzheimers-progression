#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

FORBIDDEN_FEATURE_TERMS = (
    "pet", "suvr", "centiloid", "amyloid", "tau", "fbp", "ftp", "tracer",
    "site", "scanner", "scan_date", "date_day", "days_t0", "gap",
)
DEFAULT_TABULAR = ["age", "sex", "education_years", "apoe4_carrier", "mmse_base"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def normalize_label_value(x: Any) -> Optional[int]:
    s = str(x).strip().lower()
    if s in {"ab_pos", "pos", "positive", "1", "true", "yes"}:
        return 1
    if s in {"ab_neg", "neg", "negative", "0", "false", "no"}:
        return 0
    return None


def find_key_col(df: pd.DataFrame) -> str:
    for c in ["sample_hash", "sample_id", "id_sample", "hash_id"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find sample key column: expected sample_hash/sample_id/id_sample/hash_id.")


def find_group_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["patient_hash", "patient_id_hash", "patient_id", "subject_hash", "id_patient"]:
        if c in df.columns:
            return c
    return None


def safe_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.dtype == object:
            # Basic categorical mapping for common values.
            lowered = s.astype(str).str.strip().str.lower()
            if set(lowered.dropna().unique()).issubset({"m", "f", "male", "female", "0", "1", "nan", ""}):
                out[c] = lowered.map({"m": 1.0, "male": 1.0, "f": 0.0, "female": 0.0, "1": 1.0, "0": 0.0})
            else:
                out[c] = pd.to_numeric(s, errors="coerce")
        else:
            out[c] = pd.to_numeric(s, errors="coerce")
    return out


def load_catalog(path: Path, label_col: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
    df = pd.read_csv(path)
    key_col = find_key_col(df)
    group_col = find_group_col(df)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}.")
    y = df[label_col].map(normalize_label_value)
    df = df[y.notna()].copy()
    df["_y"] = y[y.notna()].astype(int).values
    if "substudy" not in df.columns:
        df["substudy"] = "UNKNOWN"
    return df, key_col, group_col


def load_mri_features(path: Path, key_col_catalog: str) -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(path)
    key_col = key_col_catalog if key_col_catalog in df.columns else find_key_col(df)
    numeric_cols = []
    for c in df.columns:
        if c == key_col:
            continue
        lower = c.lower()
        if any(term in lower for term in FORBIDDEN_FEATURE_TERMS):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().any():
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError(f"No usable MRI feature columns found in {path}.")
    out = df[[key_col] + numeric_cols].copy()
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out, key_col, numeric_cols


def join_features(cat: pd.DataFrame, key_col: str, mri: Optional[pd.DataFrame], mri_key_col: Optional[str]) -> pd.DataFrame:
    if mri is None:
        return cat.copy()
    return cat.merge(mri, left_on=key_col, right_on=mri_key_col, how="left", suffixes=("", "_mri"))


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    out: Dict[str, Any] = {"n": int(len(y_true)), "n_pos": int((y_true == 1).sum()), "n_neg": int((y_true == 0).sum())}
    if len(y_true) == 0:
        return {**out, "status": "EMPTY"}
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    if len(np.unique(y_true)) < 2:
        out.update({"roc_auc": None, "pr_auc": None, "status": "SINGLE_CLASS_NO_DISCRIMINATION_METRIC"})
    else:
        out.update({
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "status": "OK",
        })
    return out


def get_feature_columns(df: pd.DataFrame, mode: str, mri_cols: List[str]) -> List[str]:
    cols: List[str] = []
    if mode in {"tabular_only", "tabular_plus_mri_features"}:
        cols += [c for c in DEFAULT_TABULAR if c in df.columns]
    if mode in {"mri_features_only", "tabular_plus_mri_features", "random_feature_control"}:
        cols += [c for c in mri_cols if c in df.columns]
    # Keep forbidden terms out even if they slipped into input columns.
    return [c for c in cols if not any(term in c.lower() for term in FORBIDDEN_FEATURE_TERMS)]


def cv_splits(df: pd.DataFrame, group_col: Optional[str], n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    y = df["_y"].astype(int).values
    if len(np.unique(y)) < 2:
        raise ValueError("Cannot run CV with a single class.")
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str).values
        # Reduce split count if a class is too small.
        n_splits_eff = min(n_splits, int(pd.Series(y).value_counts().min()))
        n_splits_eff = max(2, n_splits_eff)
        splitter = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
        yield from splitter.split(df, y, groups)
    else:
        n_splits_eff = min(n_splits, int(pd.Series(y).value_counts().min()))
        n_splits_eff = max(2, n_splits_eff)
        splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
        yield from splitter.split(df, y)


def fit_predict_cv(df: pd.DataFrame, feature_cols: List[str], group_col: Optional[str], n_splits: int, seed: int) -> Dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    if not feature_cols:
        return {"status": "NO_FEATURES", "feature_cols": []}

    y_all, p_all, fold_rows = [], [], []
    for fold, (tr, te) in enumerate(cv_splits(df, group_col=group_col, n_splits=n_splits, seed=seed)):
        train = df.iloc[tr]
        test = df.iloc[te]
        if len(np.unique(train["_y"].values)) < 2 or len(np.unique(test["_y"].values)) < 2:
            fold_rows.append({"fold": int(fold), "status": "SKIPPED_SINGLE_CLASS"})
            continue

        x_train = safe_numeric_frame(train, feature_cols)
        x_test = safe_numeric_frame(test, feature_cols)
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
        )
        model.fit(x_train, train["_y"].astype(int).values)
        prob = model.predict_proba(x_test)[:, 1]
        y = test["_y"].astype(int).values
        y_all.extend(y.tolist())
        p_all.extend(prob.tolist())
        row = metric_bundle(y, prob)
        row["fold"] = int(fold)
        fold_rows.append(row)

    overall = metric_bundle(np.asarray(y_all), np.asarray(p_all)) if y_all else {"status": "NO_VALID_FOLDS"}
    overall["folds"] = fold_rows
    overall["feature_count"] = int(len(feature_cols))
    overall["feature_cols"] = feature_cols
    return overall


def permute_columns(df: pd.DataFrame, cols: List[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    if not cols:
        return out
    perm = rng.permutation(len(out))
    out.loc[:, cols] = out.iloc[perm][cols].to_numpy()
    return out


def make_random_features(df: pd.DataFrame, cols: List[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in cols:
        out[c] = rng.normal(size=len(out))
    return out


def shuffle_labels_global(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["_y"] = rng.permutation(out["_y"].astype(int).values)
    return out


def substudy_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vals = out["substudy"].astype(str).str.upper()
    # A4 vs not-A4 is the most useful artifact-proxy for this project.
    out["_y"] = (vals == "A4").astype(int)
    return out


def classify_verdict(fusion: Dict[str, Any], perm_mean: Optional[float], substudy: Dict[str, Any], label_shuffle: Dict[str, Any]) -> Dict[str, Any]:
    fusion_auc = fusion.get("roc_auc")
    delta = None if fusion_auc is None or perm_mean is None else float(fusion_auc - perm_mean)
    substudy_auc = substudy.get("roc_auc")
    shuffle_auc = label_shuffle.get("roc_auc")

    verdict = {
        "mri_permutation_delta_auc": delta,
        "substudy_prediction_auc": substudy_auc,
        "global_label_shuffle_auc": shuffle_auc,
        "mri_signal_status": "UNKNOWN",
        "cohort_artifact_status": "UNKNOWN",
        "label_shuffle_status": "UNKNOWN",
        "final_claim_recommendation": "HOLD_CLAIM_PENDING_REVIEW",
    }

    if delta is not None:
        verdict["mri_signal_status"] = "MRI_FEATURES_ADD_MEASURABLE_SIGNAL" if delta >= 0.03 else "MRI_FEATURES_ADD_WEAK_OR_NO_SIGNAL"
    if substudy_auc is not None:
        if substudy_auc >= 0.85:
            verdict["cohort_artifact_status"] = "STRONG_COHORT_ARTIFACT_SIGNAL"
        elif substudy_auc >= 0.70:
            verdict["cohort_artifact_status"] = "MODERATE_COHORT_ARTIFACT_SIGNAL"
        else:
            verdict["cohort_artifact_status"] = "LIMITED_COHORT_ARTIFACT_SIGNAL"
    if shuffle_auc is not None:
        verdict["label_shuffle_status"] = "POSSIBLE_LEAKAGE_OR_SPLIT_ARTIFACT" if shuffle_auc >= 0.60 else "NO_OBVIOUS_LABEL_SHUFFLE_LEAKAGE"

    if (
        verdict["mri_signal_status"] == "MRI_FEATURES_ADD_MEASURABLE_SIGNAL"
        and verdict["cohort_artifact_status"] != "STRONG_COHORT_ARTIFACT_SIGNAL"
        and verdict["label_shuffle_status"] == "NO_OBVIOUS_LABEL_SHUFFLE_LEAKAGE"
    ):
        verdict["final_claim_recommendation"] = "CLAIM_ALLOWED_WITH_PSEUDO_EXTERNAL_CAVEAT"
    else:
        verdict["final_claim_recommendation"] = "CLAIM_MUST_BE_LIMITED_OR_REFRAMED"
    return verdict


def write_md(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Method X Validity Audit — v3.6-validity-fixes\n\n",
        "This report audits whether current performance may be driven by MRI signal, tabular signal, cohort/protocol artifacts, or leakage-like shortcuts.\n\n",
        "## Verdict\n\n",
    ]
    for k, v in report.get("verdict", {}).items():
        lines.append(f"- **{k}**: `{v}`\n")
    lines.append("\n## Core tests\n\n")
    for name, res in report.get("tests", {}).items():
        lines.append(f"### {name}\n\n")
        if isinstance(res, dict):
            for k in ["status", "n", "n_pos", "n_neg", "roc_auc", "pr_auc", "brier", "feature_count", "delta_auc_vs_fusion"]:
                if k in res:
                    lines.append(f"- {k}: `{res[k]}`\n")
        lines.append("\n")
    lines.append("\n## Required reporting language\n\n")
    lines.append(report.get("required_reporting_language", "") + "\n")
    atomic_write_text(path, "".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--mri_features", default="data/processed/mri_features.csv")
    ap.add_argument("--label_col", default="ab_label")
    ap.add_argument("--out_dir", default="results/tables/validity_audit")
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--permutation_repeats", type=int, default=10)
    args = ap.parse_args()

    catalog_path = PROJECT_ROOT / args.catalog
    mri_path = PROJECT_ROOT / args.mri_features
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cat, key_col, group_col = load_catalog(catalog_path, args.label_col)
    mri = None
    mri_key = None
    mri_cols: List[str] = []
    if mri_path.exists():
        mri, mri_key, mri_cols = load_mri_features(mri_path, key_col)
    else:
        warnings.warn(f"MRI features file not found: {mri_path}. MRI-related tests will be marked NO_FEATURES.")

    df = join_features(cat, key_col, mri, mri_key)

    results: Dict[str, Any] = {}
    for mode in ["tabular_only", "mri_features_only", "tabular_plus_mri_features"]:
        cols = get_feature_columns(df, mode, mri_cols)
        results[mode] = fit_predict_cv(df, cols, group_col, args.n_splits, args.seed)

    fusion_cols = get_feature_columns(df, "tabular_plus_mri_features", mri_cols)
    mri_only_cols = get_feature_columns(df, "mri_features_only", mri_cols)

    perm_results = []
    if mri_only_cols:
        for i in range(args.permutation_repeats):
            df_perm = permute_columns(df, mri_only_cols, seed=args.seed + 100 + i)
            perm_results.append(fit_predict_cv(df_perm, fusion_cols, group_col, args.n_splits, args.seed))
    perm_aucs = [r.get("roc_auc") for r in perm_results if r.get("roc_auc") is not None]
    perm_mean = float(np.mean(perm_aucs)) if perm_aucs else None
    results["mri_permutation_mean"] = {
        "status": "OK" if perm_mean is not None else "NO_VALID_PERMUTATION",
        "roc_auc": perm_mean,
        "delta_auc_vs_fusion": None if perm_mean is None or results["tabular_plus_mri_features"].get("roc_auc") is None else float(results["tabular_plus_mri_features"]["roc_auc"] - perm_mean),
        "n_repeats": int(len(perm_results)),
    }

    df_sub = substudy_target(df)
    results["substudy_prediction_from_mri"] = fit_predict_cv(df_sub, mri_only_cols, group_col, args.n_splits, args.seed) if mri_only_cols else {"status": "NO_MRI_FEATURES"}

    df_shuffle = shuffle_labels_global(df, seed=args.seed + 2026)
    results["global_label_shuffle"] = fit_predict_cv(df_shuffle, fusion_cols, group_col, args.n_splits, args.seed) if fusion_cols else {"status": "NO_FEATURES"}

    df_rand = make_random_features(df, mri_only_cols, seed=args.seed + 404) if mri_only_cols else df
    results["random_feature_control"] = fit_predict_cv(df_rand, fusion_cols, group_col, args.n_splits, args.seed) if fusion_cols else {"status": "NO_FEATURES"}

    report = {
        "version": "v3.6-validity-fixes",
        "method": "X",
        "inputs_checked": {
            "catalog": str(catalog_path),
            "mri_features": str(mri_path),
            "key_col": key_col,
            "group_col": group_col,
            "n_rows_after_label_filter": int(len(df)),
            "mri_feature_count": int(len(mri_cols)),
        },
        "tests": results,
        "verdict": classify_verdict(
            fusion=results.get("tabular_plus_mri_features", {}),
            perm_mean=perm_mean,
            substudy=results.get("substudy_prediction_from_mri", {}),
            label_shuffle=results.get("global_label_shuffle", {}),
        ),
        "required_reporting_language": (
            "The A4/LEARN catalog may exhibit strong cohort-label coupling. Method X results must therefore be "
            "reported as biomarker-status prediction under a locked pseudo-external evaluation setting, not as "
            "independent clinical diagnosis or pathology-only generalization. MRI feature permutation, modality "
            "ablation, substudy prediction, and global label-shuffle controls are required before strong claims."
        ),
    }

    atomic_write_text(out_dir / "validity_audit_summary.json", json.dumps(report, indent=2, ensure_ascii=False))
    write_md(report, out_dir / "validity_audit_summary.md")
    print(f"Wrote {out_dir / 'validity_audit_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
