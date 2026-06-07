from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import T1_NAME_PATTERN, resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.quarantine_paths import LEAKY_MRI_SURROGATE_REL, ensure_quarantine_readme
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


MRI_FEATURES = [
    "mri_shape_x",
    "mri_shape_y",
    "mri_shape_z",
    "mri_voxel_x",
    "mri_voxel_y",
    "mri_voxel_z",
    "mri_mean",
    "mri_std",
    "mri_p01",
    "mri_p10",
    "mri_p50",
    "mri_p90",
    "mri_p99",
    "mri_nonzero_fraction",
]


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_t1_hash_index(data_root: Path) -> dict[str, Path]:
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    out: dict[str, Path] = {}
    for p in (data_root / "T1").rglob("*"):
        if not p.is_file() or not (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            continue
        if pattern.match(p.name):
            out[stable_hash(p.name, prefix="t1")] = p
    return out


def extract_features(path: Path) -> dict[str, float]:
    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:
        raise RuntimeError("extract_mri_features requires nibabel and numpy") from exc

    img = nib.load(str(path))
    shape = img.shape[:3]
    zooms = img.header.get_zooms()[:3]
    data = img.get_fdata(dtype=np.float32)
    finite = np.isfinite(data)
    if not bool(finite.any()):
        raise ValueError("MRI contains no finite voxels")
    vals = data[finite]
    nonzero = vals[np.abs(vals) > 1e-6]
    stat_vals = nonzero if nonzero.size >= 100 else vals
    p01, p10, p50, p90, p99 = np.percentile(stat_vals, [1, 10, 50, 90, 99])
    return {
        "mri_shape_x": float(shape[0]),
        "mri_shape_y": float(shape[1]),
        "mri_shape_z": float(shape[2]),
        "mri_voxel_x": float(zooms[0]),
        "mri_voxel_y": float(zooms[1]),
        "mri_voxel_z": float(zooms[2]),
        "mri_mean": float(np.mean(stat_vals)),
        "mri_std": float(np.std(stat_vals)),
        "mri_p01": float(p01),
        "mri_p10": float(p10),
        "mri_p50": float(p50),
        "mri_p90": float(p90),
        "mri_p99": float(p99),
        "mri_nonzero_fraction": float(nonzero.size / vals.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out", default=str(LEAKY_MRI_SURROGATE_REL))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_root = resolve_data_root(args.data_root)
    rows = read_catalog(Path(args.catalog))
    if args.limit > 0:
        rows = rows[: args.limit]
    t1_index = build_t1_hash_index(data_root)
    ensure_quarantine_readme()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_hash",
        "patient_hash",
        "ab_label",
        "substudy",
        "t1_file_hash",
        *MRI_FEATURES,
        "mri_feature_status",
    ]
    ok = 0
    missing = 0
    errors = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {
                "sample_hash": row["sample_hash"],
                "patient_hash": row["patient_hash"],
                "ab_label": row["ab_label"],
                "substudy": row["substudy"],
                "t1_file_hash": row["t1_file_hash"],
            }
            path = t1_index.get(row["t1_file_hash"])
            if path is None:
                missing += 1
                out_row.update({k: "" for k in MRI_FEATURES})
                out_row["mri_feature_status"] = "MISSING_T1"
            else:
                try:
                    out_row.update(extract_features(path))
                    out_row["mri_feature_status"] = "OK"
                    ok += 1
                except Exception:
                    errors += 1
                    out_row.update({k: "" for k in MRI_FEATURES})
                    out_row["mri_feature_status"] = "ERROR"
            writer.writerow(out_row)

    summary = {
        "rows": len(rows),
        "ok": ok,
        "missing": missing,
        "errors": errors,
        "output": str(out_path),
        "contains_raw_images": False,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "quarantined_negative_control_only": True,
        "headline_model_use_forbidden": True,
        "features": MRI_FEATURES,
    }
    atomic_write_json("results/tables/mri_feature_extraction_summary.json", summary)
    md = [
        "# MRI Feature Extraction Summary\n\n",
        f"- rows: `{len(rows)}`\n",
        f"- ok: `{ok}`\n",
        f"- missing: `{missing}`\n",
        f"- errors: `{errors}`\n",
        f"- output: `{out_path}`\n",
        "- contains_raw_images: `False`\n",
        "- contains_pet_input: `False`\n",
        "- contains_raw_identifiers: `False`\n",
        f"- features: `{MRI_FEATURES}`\n",
    ]
    atomic_write_text("results/tables/mri_feature_extraction_summary.md", "".join(md))
    write_run_manifest("results/logs/mri_features_manifest.json", stage="mri_features", data_root=str(data_root), extra={"summary": summary})
    StateManager().append_event("mri_features", "COMPLETED" if errors == 0 and missing == 0 else "FAILED", "MRI-derived feature extraction completed.")
    print(json.dumps({"rows": len(rows), "ok": ok, "missing": missing, "errors": errors}, indent=2))
    return 0 if errors == 0 and missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

