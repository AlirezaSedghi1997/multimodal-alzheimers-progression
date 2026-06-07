#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAP_CANDIDATES = [
    "abs_pet_t1_gap_days",
    "days_gap_pet_t1",
    "pet_t1_gap_days",
    "abs_gap_pet_t1",
    "gap_pet_t1",
    "t1_pet_abs_gap_days",
]
LABEL_CANDIDATES = ["ab_label", "label", "y"]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def derive_gap_col(df: pd.DataFrame) -> tuple[pd.Series, str]:
    col = find_col(df, GAP_CANDIDATES)
    if col:
        return pd.to_numeric(df[col], errors="coerce").abs(), col

    pairs = [
        ("ab_pet_day_t0", "t1_day_t0"),
        ("amyloid_pet_day_t0", "t1_day_t0"),
        ("pet_day_t0", "t1_day_t0"),
        ("scan_date_DAYS_T0", "t1_day_t0"),
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for a, b in pairs:
        if a.lower() in lower_map and b.lower() in lower_map:
            aa = pd.to_numeric(df[lower_map[a.lower()]], errors="coerce")
            bb = pd.to_numeric(df[lower_map[b.lower()]], errors="coerce")
            return (aa - bb).abs(), f"derived:{lower_map[a.lower()]}-{lower_map[b.lower()]}"
    raise ValueError("No PET-T1 gap column found and gap cannot be derived from day columns.")


def label_counts(df: pd.DataFrame, label_col: Optional[str]) -> Dict[str, int]:
    if label_col is None or label_col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[label_col].astype(str).value_counts(dropna=False).to_dict().items()}


def substudy_counts(df: pd.DataFrame) -> Dict[str, int]:
    if "substudy" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["substudy"].astype(str).value_counts(dropna=False).to_dict().items()}


def normalize_two_class_status(df: pd.DataFrame, label_col: Optional[str]) -> str:
    if label_col is None or label_col not in df.columns:
        return "LABEL_COLUMN_MISSING"
    unique = set(df[label_col].astype(str).str.lower().str.strip().unique())
    pos = any(x in unique for x in {"ab_pos", "positive", "pos", "1", "true"})
    neg = any(x in unique for x in {"ab_neg", "negative", "neg", "0", "false"})
    return "BINARY_OK" if pos and neg else "SINGLE_CLASS_OR_NONSTANDARD_LABELS"


def make_variant(df: pd.DataFrame, gap: pd.Series, window: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if window == 0:
        out = df.copy()
        policy = "NO_FILTER_ALL_ALIGNED_ROWS"
    else:
        out = df[gap.notna() & (gap <= float(window))].copy()
        policy = f"ABS_GAP_LE_{window}_DAYS"

    label_col = find_col(out, LABEL_CANDIDATES)
    summary = {
        "window_days": int(window),
        "policy": policy,
        "n_before": int(len(df)),
        "n_after": int(len(out)),
        "excluded_by_gap": int(len(df) - len(out)),
        "missing_gap_count_before": int(gap.isna().sum()),
        "label_counts_after": label_counts(out, label_col),
        "substudy_counts_after": substudy_counts(out),
        "two_class_status": normalize_two_class_status(out, label_col),
    }
    return out, summary


def write_md(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Temporal Window Sensitivity — v3.6-validity-fixes\n\n",
        "This report creates catalog variants for PET–T1 temporal-window sensitivity. It does **not** overwrite `master_catalog.csv`.\n\n",
        f"- gap source: `{report.get('gap_source')}`\n",
        f"- input catalog: `{report.get('input_catalog')}`\n\n",
        "| window_days | n_after | excluded_by_gap | two_class_status | label_counts | substudy_counts |\n",
        "|---:|---:|---:|---|---|---|\n",
    ]
    for row in report["windows"]:
        lines.append(
            f"| {row['window_days']} | {row['n_after']} | {row['excluded_by_gap']} | "
            f"{row['two_class_status']} | `{row['label_counts_after']}` | `{row['substudy_counts_after']}` |\n"
        )
    lines += [
        "\n## Governance decision required\n\n",
        "Record in `DECISIONS.md` whether the mainline thesis claim uses all rows or a strict temporal window. "
        "Do not silently replace the main catalog.\n",
    ]
    atomic_write_text(path, "".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out_catalog_dir", default="data/processed/catalog_variants")
    ap.add_argument("--out_json", default="results/tables/temporal_window_sensitivity.json")
    ap.add_argument("--out_md", default="results/tables/temporal_window_sensitivity.md")
    ap.add_argument("--windows", default="0,60,90,120")
    args = ap.parse_args()

    catalog_path = PROJECT_ROOT / args.catalog
    out_dir = PROJECT_ROOT / args.out_catalog_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(catalog_path)
    gap, source = derive_gap_col(df)
    df = df.copy()
    df["abs_pet_t1_gap_days_v36"] = gap

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    report: Dict[str, Any] = {
        "version": "v3.6-validity-fixes",
        "method": "X",
        "input_catalog": str(catalog_path),
        "gap_source": source,
        "main_catalog_overwritten": False,
        "windows": [],
    }

    for w in windows:
        variant, summary = make_variant(df, gap, w)
        name = "all" if w == 0 else f"gap{w}"
        out_csv = out_dir / f"master_catalog_{name}.csv"
        variant.to_csv(out_csv, index=False)
        summary["output_catalog"] = str(out_csv)
        report["windows"].append(summary)

    atomic_write_text(PROJECT_ROOT / args.out_json, json.dumps(report, indent=2, ensure_ascii=False))
    write_md(report, PROJECT_ROOT / args.out_md)
    print(f"Wrote {PROJECT_ROOT / args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
