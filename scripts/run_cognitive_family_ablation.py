#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_TABULAR = ["age", "sex", "education_years", "apoe4_carrier", "mmse_base"]
FAMILIES = {
    "core_cognitive": ["pacc", "fctotal96", "ldeltotal", "digittotal", "mmscore"],
    "clinical_severity": ["cdglobal", "cdrsb", "cdr_memory", "cdr_orientation", "cdr_judgment", "cdr_community", "cdr_home", "cdr_care"],
    "complaints": ["cfipttotal", "cfspttotal"],
    "cogstate": ["c3comp", "learnwmcomp", "oclonbcomp", "psychattncomp", "attncomp", "c3abrcomp"],
    "mood_anxiety": ["gdtotal", "staitotal", "cadcncrn", "iescore"],
    "gdtotal_only": ["gdtotal"],
    "staitotal_only": ["staitotal"],
    "cadcncrn_only": ["cadcncrn"],
    "iescore_only": ["iescore"],
}
VARIANTS = [
    ("fixed_tabular_only", []),
    ("fixed_tabular_plus_core_cognitive", ["core_cognitive"]),
    ("fixed_tabular_plus_clinical_severity", ["clinical_severity"]),
    ("fixed_tabular_plus_complaints", ["complaints"]),
    ("fixed_tabular_plus_cogstate", ["cogstate"]),
    ("fixed_tabular_plus_mood_anxiety", ["mood_anxiety"]),
    ("fixed_tabular_plus_gdtotal_only", ["gdtotal_only"]),
    ("fixed_tabular_plus_staitotal_only", ["staitotal_only"]),
    ("fixed_tabular_plus_cadcncrn_only", ["cadcncrn_only"]),
    ("fixed_tabular_plus_iescore_only", ["iescore_only"]),
    ("fixed_tabular_plus_all_cognitive", ["core_cognitive", "clinical_severity", "complaints", "cogstate", "mood_anxiety"]),
]


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


def x_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "n_pos": int((y==1).sum()), "n_neg": int((y==0).sum()), "brier": float(brier_score_loss(y, p))}
    if len(np.unique(y)) < 2:
        out.update({"roc_auc": None, "pr_auc": None, "status": "SINGLE_CLASS"})
    else:
        out.update({"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "status": "OK"})
    return out


def evaluate(df: pd.DataFrame, cols: List[str], group_col: Optional[str], n_splits: int, seed: int) -> Dict[str, Any]:
    if not cols:
        return {"status": "NO_FEATURES", "feature_cols": []}
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    y = df["_y"].astype(int).values
    n_splits_eff = max(2, min(n_splits, int(pd.Series(y).value_counts().min())))
    if group_col and group_col in df.columns:
        splitter = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
        split_iter = splitter.split(df, y, df[group_col].astype(str).values)
    else:
        splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
        split_iter = splitter.split(df, y)

    yy, pp = [], []
    for tr, te in split_iter:
        train, test = df.iloc[tr], df.iloc[te]
        if len(np.unique(train["_y"])) < 2 or len(np.unique(test["_y"])) < 2:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"))
        model.fit(x_numeric(train, cols), train["_y"].astype(int).values)
        pred = model.predict_proba(x_numeric(test, cols))[:, 1]
        yy.extend(test["_y"].astype(int).tolist())
        pp.extend(pred.tolist())
    out = metrics(np.asarray(yy), np.asarray(pp)) if yy else {"status": "NO_VALID_FOLDS"}
    out["feature_count"] = int(len(cols))
    out["feature_cols"] = cols
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--cognitive_features", default="data/processed/cognitive_features.csv")
    ap.add_argument("--label_col", default="ab_label")
    ap.add_argument("--out_dir", default="results/final_results/method_x_cog")
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cat = pd.read_csv(PROJECT_ROOT / args.catalog)
    key_col = find_key_col(cat)
    y = cat[args.label_col].map(normalize_label)
    cat = cat[y.notna()].copy()
    cat["_y"] = y[y.notna()].astype(int).values

    cog = pd.read_csv(PROJECT_ROOT / args.cognitive_features)
    cog_key = key_col if key_col in cog.columns else find_key_col(cog)
    df = cat.merge(cog, left_on=key_col, right_on=cog_key, how="inner", suffixes=("", "_cog"))
    group_col = find_group_col(df)

    results = []
    for name, fams in VARIANTS:
        cols = [c for c in FIXED_TABULAR if c in df.columns]
        for fam in fams:
            cols += [c for c in FAMILIES[fam] if c in df.columns]
        # Remove duplicates while preserving order.
        cols = list(dict.fromkeys(cols))
        row = evaluate(df, cols, group_col, args.n_splits, args.seed)
        row.update({"variant": name, "families": fams})
        results.append(row)

    base_auc = next((r.get("roc_auc") for r in results if r["variant"] == "fixed_tabular_only"), None)
    for r in results:
        r["auc_gain_vs_fixed_tabular"] = None if base_auc is None or r.get("roc_auc") is None else float(r["roc_auc"] - base_auc)

    report = {
        "version": "v3.6-validity-fixes",
        "method": "X-Cog",
        "status": "exploratory",
        "n_rows": int(len(df)),
        "variants": results,
        "interpretation_policy": "Report family-level contribution. Do not attribute gains to core cognition unless the family ablation supports it.",
    }

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_dir / "method_x_cog_family_ablation.json", json.dumps(report, indent=2, ensure_ascii=False))
    lines = ["# Method X-Cog Family Ablation — v3.6-validity-fixes\n\n", "Status: exploratory.\n\n", "| variant | ROC-AUC | PR-AUC | Brier | AUC gain vs fixed tabular | feature_count |\n|---|---:|---:|---:|---:|---:|\n"]
    for r in results:
        lines.append(f"| {r['variant']} | {r.get('roc_auc')} | {r.get('pr_auc')} | {r.get('brier')} | {r.get('auc_gain_vs_fixed_tabular')} | {r.get('feature_count')} |\n")
    atomic_write_text(out_dir / "method_x_cog_family_ablation.md", "".join(lines))
    print(f"Wrote {out_dir / 'method_x_cog_family_ablation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
