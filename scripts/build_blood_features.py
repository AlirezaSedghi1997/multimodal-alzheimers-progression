#!/usr/bin/env python3
"""Extract blood biomarker features from A4/LEARN clinical data."""
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


def create_sample_hashes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["patient_hash"] = df.apply(
        lambda row: "pt_" + stable_hash(f"{row['SUBSTUDY']}_{row['BID']}"), axis=1
    )
    df["sample_hash"] = df.apply(
        lambda row: "s_" + stable_hash(f"{row['SUBSTUDY']}_{row['BID']}"), axis=1
    )
    return df


def load_ptau217() -> pd.DataFrame:
    path = DATA_ROOT / "Clinical" / "External Data" / "biomarker_pTau217.csv"
    if not path.exists():
        print(f"WARNING: pTau217 file not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["ORRES"] = pd.to_numeric(df["ORRES"], errors="coerce")
    df["ORRESRAW"] = pd.to_numeric(df["ORRESRAW"], errors="coerce")
    df["value"] = df["ORRES"].fillna(df["ORRESRAW"])
    baseline = df[df["VISCODE"].astype(str).isin(["6", "006", "0"])].copy()
    if baseline.empty:
        baseline = df.sort_values("VISCODE").groupby("BID").first().reset_index()
    baseline = baseline[["SUBSTUDY", "BID", "value"]].dropna(subset=["value"])
    baseline = baseline.rename(columns={"value": "pTau217"})
    return baseline


def load_plasma_biomarkers() -> pd.DataFrame:
    path = DATA_ROOT / "Clinical" / "External Data" / "biomarker_Plasma_Roche_Results.csv"
    if not path.exists():
        print(f"WARNING: Plasma Roche file not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["LABRESN"] = pd.to_numeric(df["LABRESN"], errors="coerce")
    df["LBTESTCD"] = df["LBTESTCD"].str.strip()
    baseline = df[df["VISCODE"].astype(str).isin(["001", "1", "0"])].copy()
    if baseline.empty:
        baseline = df.sort_values("VISCODE").groupby(["BID", "LBTESTCD"]).first().reset_index()
    pivot = baseline.pivot_table(index=["SUBSTUDY", "BID"], columns="LBTESTCD", values="LABRESN", aggfunc="first")
    pivot = pivot.reset_index()
    return pivot


def load_ab_ratio() -> pd.DataFrame:
    path = DATA_ROOT / "Clinical" / "External Data" / "biomarker_AB_Test.csv"
    if not path.exists():
        print(f"WARNING: AB Test file not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["LBORRES"] = pd.to_numeric(df["LBORRES"], errors="coerce")
    baseline = df[df["VISCODE"].astype(str).isin(["003", "3", "0"])].copy()
    if baseline.empty:
        baseline = df.sort_values("VISCODE").groupby(["BID", "LBTESTCD"]).first().reset_index()
    pivot = baseline.pivot_table(index=["SUBSTUDY", "BID"], columns="LBTESTCD", values="LBORRES", aggfunc="first")
    pivot = pivot.reset_index()
    return pivot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()

    print("Loading pTau217...")
    ptau = load_ptau217()
    print(f"  pTau217: {len(ptau)} subjects")

    print("Loading plasma biomarkers...")
    plasma = load_plasma_biomarkers()
    print(f"  Plasma: {len(plasma)} subjects")

    print("Loading AB ratio...")
    ab = load_ab_ratio()
    print(f"  AB ratio: {len(ab)} subjects")

    merged = ptau[["SUBSTUDY", "BID", "pTau217"]].copy()
    if not plasma.empty:
        keep_cols = ["SUBSTUDY", "BID"]
        for col in ["GFAP", "AMYLB40", "AMYLB42", "APOE4"]:
            if col in plasma.columns:
                keep_cols.append(col)
        merged = merged.merge(plasma[keep_cols], on=["SUBSTUDY", "BID"], how="outer")
    if not ab.empty:
        ab_cols = ["SUBSTUDY", "BID"]
        for col in ab.columns:
            if col not in ["SUBSTUDY", "BID"]:
                ab_cols.append(col)
        merged = merged.merge(ab[ab_cols], on=["SUBSTUDY", "BID"], how="outer")

    if "AMYLB42" in merged.columns and "AMYLB40" in merged.columns:
        merged["AB42_40_ratio"] = merged["AMYLB42"] / merged["AMYLB40"].replace(0, float("nan"))

    merged = create_sample_hashes(merged)
    merged = merged.dropna(subset=["pTau217"])

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = ["pTau217"]
    if "GFAP" in merged.columns:
        feature_cols.append("GFAP")
    if "AB42_40_ratio" in merged.columns:
        feature_cols.append("AB42_40_ratio")
    if "APOE4" in merged.columns:
        feature_cols.append("APOE4")

    features = merged[["sample_hash", "patient_hash", "SUBSTUDY", "BID"] + feature_cols].copy()
    for col in feature_cols:
        features[col] = features[col].fillna(features[col].median())

    features.to_csv(out_dir / "blood_features.csv", index=False)

    summary = {
        "n_subjects": len(features),
        "feature_columns": feature_cols,
    }
    for col in feature_cols:
        summary[f"{col}_mean"] = float(features[col].mean())
        summary[f"{col}_std"] = float(features[col].std())

    (out_dir / "blood_features_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
