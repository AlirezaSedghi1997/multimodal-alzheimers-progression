#!/usr/bin/env python3
"""Merge real and synthetic MRI caches for training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_cache(cache_path: Path) -> dict:
    import h5py
    data = {}
    with h5py.File(cache_path, "r") as h5:
        data["image"] = h5["image"][:]
        data["label"] = h5["label"][:]
        data["sample_hash"] = [v.decode() if isinstance(v, bytes) else v for v in h5["sample_hash"][:]]
        data["tabular"] = h5["tabular"][:]
        data["status"] = [v.decode() if isinstance(v, bytes) else v for v in h5["status"][:]]
    return data


def merge_caches(real: dict, synthetic: dict) -> dict:
    merged = {
        "image": np.concatenate([real["image"], synthetic["image"]], axis=0),
        "label": np.concatenate([real["label"], synthetic["label"]], axis=0),
        "tabular": np.concatenate([real["tabular"], synthetic["tabular"]], axis=0),
        "sample_hash": real["sample_hash"] + synthetic["sample_hash"],
        "status": real["status"] + ["OK"] * len(synthetic["label"]),
    }
    return merged


def save_cache(data: dict, output_path: Path) -> None:
    import h5py
    n_real = data.get("n_real", 0)
    n_syn = data.get("n_synthetic", 0)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("image", data=data["image"], dtype="float16", compression="gzip")
        h5.create_dataset("label", data=data["label"], dtype="int8")
        h5.create_dataset("tabular", data=data["tabular"], dtype="float32")
        h5.create_dataset("sample_hash", data=[s.encode() for s in data["sample_hash"]])
        h5.create_dataset("status", data=[s.encode() for s in data["status"]])
        h5.attrs["contains_synthetic"] = True
        h5.attrs["n_real"] = n_real
        h5.attrs["n_synthetic"] = n_syn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/cached/t1_volume_cache_128.h5")
    ap.add_argument("--synthetic", default="data/cached/t1_synthetic_128.h5")
    ap.add_argument("--out", default="data/cached/t1_merged_128.h5")
    args = ap.parse_args()

    real_path = PROJECT_ROOT / args.real
    syn_path = PROJECT_ROOT / args.synthetic
    out_path = PROJECT_ROOT / args.out

    print(f"Loading real cache: {real_path}")
    real = load_cache(real_path)
    print(f"  Real samples: {len(real['label'])}")

    print(f"Loading synthetic cache: {syn_path}")
    synthetic = load_cache(syn_path)
    print(f"  Synthetic samples: {len(synthetic['label'])}")

    print("Merging caches...")
    merged = merge_caches(real, synthetic)
    print(f"  Total samples: {len(merged['label'])}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_cache(merged, out_path)
    print(f"Saved to: {out_path}")

    summary = {
        "n_real": len(real["label"]),
        "n_synthetic": len(synthetic["label"]),
        "total": len(merged["label"]),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
