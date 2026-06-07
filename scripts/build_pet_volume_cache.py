#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.utils.pet_spatial_qc import atomic_write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracer", default="fbp")
    ap.add_argument("--preprocessed_root", default="data/preprocessed_pet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--require_qc_pass", action="store_true", default=True)
    args = ap.parse_args()
    try:
        import nibabel as nib
        import h5py
    except Exception as e:
        raise RuntimeError("nibabel and h5py are required to build Method Y PET cache.") from e
    in_dir = PROJECT_ROOT / args.preprocessed_root / args.tracer / "normalized"
    qc_dir = PROJECT_ROOT / args.preprocessed_root / args.tracer / "qc"
    files = sorted(in_dir.glob("*.nii.gz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No normalized PET NIfTI files found in {in_dir}")
    out_path = Path(args.out) if args.out else PROJECT_ROOT / "data" / "cached_method_y_pet" / f"{args.tracer}_pet_96.h5"
    if "data/cached/" in out_path.as_posix():
        raise RuntimeError("Do not write Method Y PET cache inside Method X data/cached/ root.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays, sample_hashes, included, excluded = [], [], [], []
    for fp in files:
        sample_hash = fp.name.replace(".nii.gz", "")
        qc_path = qc_dir / f"{sample_hash}.json"
        if args.require_qc_pass:
            if not qc_path.exists():
                excluded.append({"sample_hash": sample_hash, "reason": "MISSING_QC_JSON"})
                continue
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            if qc.get("status") != "PASS":
                excluded.append({"sample_hash": sample_hash, "reason": "QC_NOT_PASS"})
                continue
        img = nib.load(str(fp))
        arr = img.get_fdata(dtype="float32")
        if arr.ndim != 3:
            excluded.append({"sample_hash": sample_hash, "reason": "NOT_3D"})
            continue
        target = np.zeros((96, 96, 96), dtype=np.float16)
        src = arr.astype(np.float32)
        src_shape = np.array(src.shape); dst_shape = np.array(target.shape)
        crop = np.minimum(src_shape, dst_shape)
        src_start = ((src_shape - crop) // 2).astype(int); dst_start = ((dst_shape - crop) // 2).astype(int)
        target[dst_start[0]:dst_start[0]+crop[0], dst_start[1]:dst_start[1]+crop[1], dst_start[2]:dst_start[2]+crop[2]] = src[src_start[0]:src_start[0]+crop[0], src_start[1]:src_start[1]+crop[1], src_start[2]:src_start[2]+crop[2]].astype(np.float16)
        arrays.append(target[None, ...]); sample_hashes.append(sample_hash); included.append(sample_hash)
    if not arrays:
        raise RuntimeError("No QC-pass PET volumes available for cache.")
    import h5py
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("pet_volume", data=np.stack(arrays, axis=0), chunks=(1,1,96,96,96), compression="lzf", dtype="float16")
        h5.create_dataset("sample_hash", data=np.array(sample_hashes, dtype="S64"))
        h5.attrs["method"] = "Y"
        h5.attrs["pet_input_allowed"] = True
        h5.attrs["not_method_x_cache"] = True
        h5.attrs["requires_qc_pass"] = True
    summary = {"stage": "method_y_pet_cache", "version": "v3.8-literature-aligned", "tracer": args.tracer, "n": len(included), "excluded": excluded[:50], "cache_path_redacted": "data/cached_method_y_pet/<tracer>_pet_96.h5", "method_x_cache": False, "dataset_key": "pet_volume"}
    atomic_write_json(PROJECT_ROOT / "results" / "tables" / "method_y_pet_cache_summary.json", summary)
    print(f"Wrote Method Y PET cache: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
