#!/usr/bin/env python3
"""Extract FLAIR white matter hyperintensity (WMH) features.

WMH is a strong AD biomarker reflecting small vessel disease.
Features: WMH volume, WMH/ICV ratio, WMH corrected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"H:\A4 Learn")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load_flair_wmh() -> pd.DataFrame:
    path = DATA_ROOT / "Clinical" / "External Data" / "imaging_FLAIR_WMH_QC.csv"
    if not path.exists():
        print(f"WARNING: FLAIR WMH file not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["WMHvol_masked"] = pd.to_numeric(df["WMHvol_masked"], errors="coerce")
    df["ICV"] = pd.to_numeric(df["ICV"], errors="coerce")
    df["WMH_corrected"] = pd.to_numeric(df["WMH_corrected"], errors="coerce")
    baseline = df[df["QC"] == "pass"].copy()
    baseline = baseline.sort_values("VISCODE").groupby("BID").first().reset_index()
    baseline["WMH_ICV_ratio"] = baseline["WMHvol_masked"] / baseline["ICV"].replace(0, float("nan"))
    return baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()

    print("Loading FLAIR WMH data...")
    wmh = load_flair_wmh()
    print(f"  FLAIR WMH: {len(wmh)} subjects with QC pass")

    if wmh.empty:
        print("ERROR: No FLAIR WMH data found")
        return 1

    wmh["patient_hash"] = wmh.apply(
        lambda row: "pt_" + stable_hash(f"{row['SUBSTUDY']}_{row['BID']}"), axis=1
    )
    wmh["sample_hash"] = wmh.apply(
        lambda row: "s_" + stable_hash(f"{row['SUBSTUDY']}_{row['BID']}"), axis=1
    )

    features = wmh[["sample_hash", "patient_hash", "SUBSTUDY", "BID",
                     "WMHvol_masked", "ICV", "WMH_corrected", "WMH_ICV_ratio"]].copy()
    for col in ["WMHvol_masked", "ICV", "WMH_corrected", "WMH_ICV_ratio"]:
        features[col] = features[col].fillna(features[col].median())

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "flair_wmh_features.csv", index=False)

    summary = {
        "n_subjects": len(features),
        "wmh_volume_mean": float(features["WMHvol_masked"].mean()),
        "wmh_volume_std": float(features["WMHvol_masked"].std()),
        "wmh_icv_ratio_mean": float(features["WMH_ICV_ratio"].mean()),
        "feature_columns": ["WMHvol_masked", "ICV", "WMH_corrected", "WMH_ICV_ratio"],
    }
    (out_dir / "flair_wmh_features_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
