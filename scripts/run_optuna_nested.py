#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FEATURE_TERMS = ("pet", "suvr", "centiloid", "amyloid", "tau", "fbp", "ftp", "tracer", "site", "scanner", "gap")
DEFAULT_TABULAR = ["age", "sex", "education_years", "apoe4_carrier", "mmse_base"]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def normalize_label(x: Any) -> Optional[int]:
    s = str(x).strip().lower()
    if s in {"ab_pos", "positive", "pos", "1", "true"}:
        return 1
    if s in {"ab_neg", "negative", "neg", "0", "false"}:
        return 0
    return None


def find_key_col(df: pd.DataFrame) -> str:
    for c in ["sample_hash", "sample_id", "id_sample", "hash_id"]:
        if c in df.columns:
            return c
    raise ValueError("No sample key column found.")


def find_group_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["patient_hash", "patient_id_hash", "patient_id", "subject_hash", "id_patient"]:
        if c in df.columns:
            return c
    return None


def load_data(catalog_path: Path, mri_features_path: Optional[Path], label_col: str, mode: str) -> tuple[pd.DataFrame, str, Optional[str], List[str]]:
    df = pd.read_csv(catalog_path)
    key_col = find_key_col(df)
    group_col = find_group_col(df)
    y = df[label_col].map(normalize_label)
    df = df[y.notna()].copy()
    df["_y"] = y[y.notna()].astype(int).values

    feature_cols: List[str] = []
    if mode in {"tabular_only", "tabular_plus_mri_features"}:
        feature_cols += [c for c in DEFAULT_TABULAR if c in df.columns]

    if mode in {"mri_features_only", "tabular_plus_mri_features"}:
        if not mri_features_path or not mri_features_path.exists():
            raise FileNotFoundError(f"MRI features file required for mode={mode}: {mri_features_path}")
        mf = pd.read_csv(mri_features_path)
        mf_key = key_col if key_col in mf.columns else find_key_col(mf)
        numeric = []
        for c in mf.columns:
            if c == mf_key:
                continue
            if any(term in c.lower() for term in FORBIDDEN_FEATURE_TERMS):
                continue
            if pd.to_numeric(mf[c], errors="coerce").notna().any():
                numeric.append(c)
                mf[c] = pd.to_numeric(mf[c], errors="coerce")
        df = df.merge(mf[[mf_key] + numeric], left_on=key_col, right_on=mf_key, how="inner")
        feature_cols += numeric

    feature_cols = [c for c in feature_cols if not any(term in c.lower() for term in FORBIDDEN_FEATURE_TERMS)]
    return df, key_col, group_col, feature_cols


def exclude_external(df: pd.DataFrame, key_col: str, external_holdout_path: Optional[Path]) -> pd.DataFrame:
    if not external_holdout_path or not external_holdout_path.exists():
        return df.copy()
    ext = pd.read_csv(external_holdout_path)
    ext_key = key_col if key_col in ext.columns else find_key_col(ext)
    ext_keys = set(ext[ext_key].astype(str))
    return df[~df[key_col].astype(str).isin(ext_keys)].copy()


def numeric_x(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].dtype == object:
            s = df[c].astype(str).str.strip().str.lower()
            if set(s.dropna().unique()).issubset({"m", "f", "male", "female", "0", "1", "", "nan"}):
                out[c] = s.map({"m": 1.0, "male": 1.0, "f": 0.0, "female": 0.0, "1": 1.0, "0": 0.0})
            else:
                out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def metrics(y, p) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
    if len(np.unique(y)) < 2:
        out.update({"roc_auc": None, "pr_auc": None, "brier": float(brier_score_loss(y, p)), "status": "SINGLE_CLASS"})
    else:
        out.update({"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "status": "OK"})
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: List[str], params: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty=str(params.get("penalty", "l2")),
            solver="liblinear",
            class_weight=params.get("class_weight", "balanced"),
            max_iter=1000,
        ),
    )
    model.fit(numeric_x(train, cols), train["_y"].astype(int).values)
    p = model.predict_proba(numeric_x(test, cols))[:, 1]
    return test["_y"].astype(int).values, p


def param_candidates(n_trials: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        yield {
            "C": float(10 ** rng.uniform(-3, 2)),
            "penalty": "l2",
            "class_weight": "balanced" if rng.random() < 0.7 else None,
        }


def inner_score(train_outer: pd.DataFrame, cols: List[str], group_col: Optional[str], params: Dict[str, Any], seed: int, inner_splits: int) -> float:
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    y = train_outer["_y"].astype(int).values
    if len(np.unique(y)) < 2:
        return float("nan")
    n_splits_eff = max(2, min(inner_splits, int(pd.Series(y).value_counts().min())))
    if group_col and group_col in train_outer.columns:
        splitter = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
        split_iter = splitter.split(train_outer, y, train_outer[group_col].astype(str).values)
    else:
        splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
        split_iter = splitter.split(train_outer, y)
    scores = []
    for tr, va in split_iter:
        tr_df, va_df = train_outer.iloc[tr], train_outer.iloc[va]
        if len(np.unique(tr_df["_y"])) < 2 or len(np.unique(va_df["_y"])) < 2:
            continue
        yy, pp = fit_predict(tr_df, va_df, cols, params)
        m = metrics(yy, pp)
        if m.get("roc_auc") is not None:
            scores.append(m["roc_auc"])
    return float(np.mean(scores)) if scores else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--mri_features", default="data/processed/mri_features.csv")
    ap.add_argument("--external_holdout", default="splits/external_holdout.csv")
    ap.add_argument("--label_col", default="ab_label")
    ap.add_argument("--mode", choices=["tabular_only", "mri_features_only", "tabular_plus_mri_features"], default="tabular_plus_mri_features")
    ap.add_argument("--outer_splits", type=int, default=5)
    ap.add_argument("--inner_splits", type=int, default=3)
    ap.add_argument("--n_trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/final_results/optuna_nested")
    args = ap.parse_args()

    catalog_path = PROJECT_ROOT / args.catalog
    mf_path = PROJECT_ROOT / args.mri_features
    ext_path = PROJECT_ROOT / args.external_holdout
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, key_col, group_col, cols = load_data(catalog_path, mf_path, args.label_col, args.mode)
    df = exclude_external(df, key_col, ext_path)
    if not cols:
        raise RuntimeError("No valid features for nested HPO after forbidden-term filtering.")

    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    y = df["_y"].astype(int).values
    n_splits_eff = max(2, min(args.outer_splits, int(pd.Series(y).value_counts().min())))
    if group_col and group_col in df.columns:
        splitter = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(df, y, df[group_col].astype(str).values)
    else:
        splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(df, y)

    folds = []
    for fold, (tr_idx, te_idx) in enumerate(split_iter):
        train_outer, test_outer = df.iloc[tr_idx], df.iloc[te_idx]
        best = {"score": -np.inf, "params": None}
        for i, params in enumerate(param_candidates(args.n_trials, args.seed + fold * 1000)):
            score = inner_score(train_outer, cols, group_col, params, args.seed + i, args.inner_splits)
            if np.isfinite(score) and score > best["score"]:
                best = {"score": score, "params": params}
        if best["params"] is None:
            raise RuntimeError(f"No valid inner HPO candidate for outer fold {fold}.")
        yy, pp = fit_predict(train_outer, test_outer, cols, best["params"])
        row = metrics(yy, pp)
        row.update({
            "outer_fold": int(fold),
            "inner_best_auc": float(best["score"]),
            "best_params": best["params"],
            "outer_test_used_for_tuning": False,
            "external_used_for_tuning": False,
            "calibration_fit_inside_objective": False,
        })
        folds.append(row)

    aucs = [f["roc_auc"] for f in folds if f.get("roc_auc") is not None]
    report = {
        "version": "v3.6-validity-fixes",
        "method": "X",
        "mode": args.mode,
        "feature_count": int(len(cols)),
        "outer_splits": int(len(folds)),
        "inner_splits": int(args.inner_splits),
        "n_trials_requested": int(args.n_trials),
        "external_used_for_tuning": False,
        "outer_test_used_for_tuning": False,
        "folds": folds,
        "summary": {
            "mean_outer_roc_auc": float(np.mean(aucs)) if aucs else None,
            "std_outer_roc_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
        },
    }

    atomic_write_text(out_dir / "optuna_nested_summary.json", json.dumps(report, indent=2, ensure_ascii=False))
    lines = ["# Nested HPO Summary — v3.6-validity-fixes\n\n", f"- mode: `{args.mode}`\n", f"- external_used_for_tuning: `False`\n", f"- outer_test_used_for_tuning: `False`\n\n", "| fold | inner_best_auc | outer_roc_auc | pr_auc | brier |\n|---:|---:|---:|---:|---:|\n"]
    for f in folds:
        lines.append(f"| {f['outer_fold']} | {f['inner_best_auc']} | {f.get('roc_auc')} | {f.get('pr_auc')} | {f.get('brier')} |\n")
    atomic_write_text(out_dir / "optuna_nested_summary.md", "".join(lines))
    print(f"Wrote {out_dir / 'optuna_nested_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
