from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import T1_NAME_PATTERN, resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_t1_hash_index(data_root: Path) -> dict[str, Path]:
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    out: dict[str, Path] = {}
    for p in (data_root / "T1").rglob("*"):
        if not p.is_file() or not (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            continue
        if not pattern.match(p.name):
            continue
        out[stable_hash(p.name, prefix="t1")] = p
    return out


def read_nifti_header(path: Path) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as f:
        hdr = f.read(348)
    if len(hdr) < 348:
        raise ValueError("NIfTI header too short")
    sizeof_hdr = struct.unpack("<i", hdr[:4])[0]
    endian = "<" if sizeof_hdr == 348 else ">"
    if struct.unpack(endian + "i", hdr[:4])[0] != 348:
        raise ValueError("Invalid NIfTI header size")
    dim = struct.unpack(endian + "8h", hdr[40:56])
    pixdim = struct.unpack(endian + "8f", hdr[76:108])
    shape = tuple(int(x) for x in dim[1:4])
    voxel = tuple(float(x) for x in pixdim[1:4])
    if any(x <= 0 for x in shape):
        raise ValueError("Invalid NIfTI shape")
    return shape, voxel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out", default="data/cached/t1_header_cache.h5")
    ap.add_argument("--limit", type=int, default=0, help="Optional development limit; 0 means all rows.")
    args = ap.parse_args()

    try:
        import h5py
        import numpy as np
    except Exception as exc:
        raise RuntimeError("preprocess_mri requires h5py and numpy in the active environment") from exc

    data_root = resolve_data_root(args.data_root)
    catalog = read_catalog(Path(args.catalog))
    if args.limit and args.limit > 0:
        catalog = catalog[: args.limit]
    t1_index = build_t1_hash_index(data_root)

    sample_hashes: list[str] = []
    patient_hashes: list[str] = []
    labels: list[int] = []
    shapes: list[tuple[int, int, int]] = []
    voxels: list[tuple[float, float, float]] = []
    qc_status: list[str] = []
    missing = 0
    errors = 0
    shape_counts: Counter[str] = Counter()

    for row in catalog:
        sample_hashes.append(row["sample_hash"])
        patient_hashes.append(row["patient_hash"])
        labels.append(int(row["ab_label"]))
        path = t1_index.get(row["t1_file_hash"])
        if path is None:
            missing += 1
            shapes.append((0, 0, 0))
            voxels.append((0.0, 0.0, 0.0))
            qc_status.append("MISSING_T1")
            continue
        try:
            shape, voxel = read_nifti_header(path)
            shapes.append(shape)
            voxels.append(voxel)
            qc_status.append("OK")
            shape_counts["x".join(str(x) for x in shape)] += 1
        except Exception:
            errors += 1
            shapes.append((0, 0, 0))
            voxels.append((0.0, 0.0, 0.0))
            qc_status.append("HEADER_ERROR")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out, "w") as h5:
        h5.attrs["method"] = "X"
        h5.attrs["contains_raw_images"] = False
        h5.attrs["contains_pet_input"] = False
        h5.attrs["contains_raw_identifiers"] = False
        h5.create_dataset("sample_hash", data=np.array(sample_hashes, dtype=object), dtype=str_dtype)
        h5.create_dataset("patient_hash", data=np.array(patient_hashes, dtype=object), dtype=str_dtype)
        h5.create_dataset("ab_label", data=np.array(labels, dtype=np.int8))
        h5.create_dataset("mri_shape", data=np.array(shapes, dtype=np.int32))
        h5.create_dataset("mri_voxel_size", data=np.array(voxels, dtype=np.float32))
        h5.create_dataset("mri_qc_status", data=np.array(qc_status, dtype=object), dtype=str_dtype)

    status_counts = dict(Counter(qc_status))
    summary = {
        "rows": len(catalog),
        "cache_path": str(out),
        "contains_raw_images": False,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "missing_t1": missing,
        "header_errors": errors,
        "qc_status_counts": status_counts,
        "top_shapes": dict(shape_counts.most_common(10)),
    }
    atomic_write_json("results/tables/mri_preprocess_qc_summary.json", summary)
    md = [
        "# MRI Preprocess QC Summary\n\n",
        f"- rows: `{len(catalog)}`\n",
        f"- cache_path: `{out}`\n",
        "- contains_raw_images: `False`\n",
        "- contains_pet_input: `False`\n",
        "- contains_raw_identifiers: `False`\n",
        f"- missing_t1: `{missing}`\n",
        f"- header_errors: `{errors}`\n",
        f"- qc_status_counts: `{status_counts}`\n",
        f"- top_shapes: `{dict(shape_counts.most_common(10))}`\n",
    ]
    atomic_write_text("results/tables/mri_preprocess_qc_summary.md", "".join(md))
    write_run_manifest("results/logs/preprocess_manifest.json", stage="preprocess", data_root=str(data_root), extra={"summary": summary})
    StateManager().append_event("preprocess", "COMPLETED", "T1 header-only QC cache generated.")
    print(f"preprocess_ready rows={len(catalog)} ok={status_counts.get('OK', 0)} missing={missing} errors={errors}")
    return 0 if errors == 0 and missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

