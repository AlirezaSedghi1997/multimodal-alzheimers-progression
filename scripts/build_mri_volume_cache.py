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
from src.utils.privacy import stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


TABULAR_FEATURES = ["age", "education_years", "apoe4_carrier", "mmse_base"]
REQUIRED_CACHE_KEYS = {"image", "label", "sample_hash", "patient_hash", "tabular", "status"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def iter_t1_files(data_root: Path):
    t1_root = data_root / "T1"
    roots = [t1_root] if t1_root.exists() else []
    if not roots and data_root.exists():
        roots = [p for p in data_root.rglob("*") if p.is_dir() and p.name.lower() == "t1"]
    for root in roots:
        for p in root.rglob("*"):
            name_lower = p.name.lower()
            if p.is_file() and (name_lower.endswith(".nii") or name_lower.endswith(".nii.gz")):
                yield p


def build_t1_hash_index(data_root: Path) -> dict[str, Path]:
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    out: dict[str, Path] = {}
    for p in iter_t1_files(data_root):
        if pattern.match(p.name):
            out[stable_hash(p.name, prefix="t1")] = p
    return out


def strip_nifti_suffix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return path.stem


def build_preprocessed_hash_index(preprocessed_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not preprocessed_dir.exists():
        return out
    for p in preprocessed_dir.rglob("*"):
        if "_scratch" in p.parts:
            continue
        lower = p.name.lower()
        if p.is_file() and (lower.endswith(".nii") or lower.endswith(".nii.gz")):
            out[strip_nifti_suffix(p)] = p
    return out


def write_cache_summary(summary: dict[str, object]) -> None:
    atomic_write_json("results/tables/mri_volume_cache_summary.json", summary)
    md = [
        "# MRI Volume Cache Summary\n\n",
        f"- rows: `{summary['rows']}`\n",
        f"- ok: `{summary['ok']}`\n",
        f"- missing: `{summary['missing']}`\n",
        f"- errors: `{summary['errors']}`\n",
        f"- resolution: `{summary['resolution']}`\n",
        f"- indexed_t1: `{summary.get('indexed_t1')}`\n",
        f"- catalog_t1_hashes: `{summary.get('catalog_t1_hashes')}`\n",
        f"- catalog_t1_hash_overlap: `{summary.get('catalog_t1_hash_overlap')}`\n",
        f"- source_mode: `{summary.get('source_mode')}`\n",
        f"- preprocessed_dir: `{summary.get('preprocessed_dir')}`\n",
        f"- indexed_preprocessed: `{summary.get('indexed_preprocessed')}`\n",
        f"- catalog_preprocessed_overlap: `{summary.get('catalog_preprocessed_overlap')}`\n",
        f"- cache_path: `{summary['cache_path']}`\n",
        "- contains_pet_input: `False`\n",
        "- contains_raw_identifiers: `False`\n",
        "- contains_processed_t1: `True`\n",
    ]
    atomic_write_text("results/tables/mri_volume_cache_summary.md", "".join(md))


def sex_value(row: dict[str, str]) -> float:
    value = str(row.get("sex", "")).strip()
    return 1.0 if value in {"1", "M", "Male", "male"} else 0.0


def tabular_vector(row: dict[str, str]) -> list[float]:
    return [safe_float(row.get(f)) for f in TABULAR_FEATURES] + [sex_value(row)]


def resolve_torch_device(device: str):
    import torch

    requested = device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for MRI cache generation, but torch.cuda.is_available() is false")
    return torch.device(requested)


def load_volume(path: Path, resolution: int, device: str | object = "cpu"):
    try:
        import nibabel as nib
        import numpy as np
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("build_mri_volume_cache requires nibabel, numpy, and torch") from exc

    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D MRI volume, got shape {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(data) & (data != 0)
    if not np.any(mask):
        raise ValueError("No finite brain voxels")
    brain = data[mask]
    lo, hi = np.percentile(brain, [1, 99])
    if hi > lo:
        data[mask] = np.clip(brain, lo, hi)
    brain = data[mask]
    mean = float(brain.mean())
    std = float(brain.std())
    if std < 1e-6:
        std = 1.0
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (data[mask] - mean) / std
    data = normalized
    torch_device = resolve_torch_device(str(device))
    tensor = torch.from_numpy(data).float()[None, None].to(torch_device)
    tensor = F.interpolate(tensor, size=(resolution, resolution, resolution), mode="trilinear", align_corners=False)
    return tensor[0].detach().cpu().numpy().astype(np.float16)


def validate_cache(path: Path) -> None:
    import h5py

    with h5py.File(path, "r") as h5:
        missing = REQUIRED_CACHE_KEYS.difference(h5.keys())
        if missing:
            raise RuntimeError(f"MRI volume cache missing required keys: {sorted(missing)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out", default="data/cached/t1_volume_cache_64.h5")
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device for volume resampling.")
    ap.add_argument(
        "--preprocessed_dir",
        default="",
        help="Optional directory of full-preprocessed sample_hash.nii.gz files to cache instead of raw T1 files.",
    )
    args = ap.parse_args()

    try:
        import h5py
        import numpy as np
    except Exception as exc:
        raise RuntimeError("build_mri_volume_cache requires h5py and numpy") from exc

    rows = read_rows(Path(args.catalog))
    if args.limit > 0:
        rows = rows[: args.limit]
    preprocessed_dir = Path(args.preprocessed_dir) if args.preprocessed_dir else None
    data_root = resolve_data_root(args.data_root) if preprocessed_dir is None else None
    t1_index = build_t1_hash_index(data_root) if data_root is not None else {}
    preprocessed_index = build_preprocessed_hash_index(preprocessed_dir) if preprocessed_dir is not None else {}
    source_index = preprocessed_index if preprocessed_dir is not None else t1_index
    source_mode = "preprocessed_mri" if preprocessed_dir is not None else "raw_t1"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    n = len(rows)
    catalog_t1_hashes = {str(r.get("t1_file_hash", "")) for r in rows if r.get("t1_file_hash")}
    overlap = catalog_t1_hashes.intersection(t1_index.keys())
    catalog_sample_hashes = {str(r.get("sample_hash", "")) for r in rows if r.get("sample_hash")}
    preprocessed_overlap = catalog_sample_hashes.intersection(preprocessed_index.keys())
    base_summary = {
        "rows": n,
        "ok": 0,
        "missing": n,
        "errors": 0,
        "resolution": args.resolution,
        "device_requested": args.device,
        "cache_path": str(out_path),
        "source_mode": source_mode,
        "preprocessed_dir": str(preprocessed_dir) if preprocessed_dir else None,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "contains_processed_t1": True,
        "intensity_normalization": "nonzero_brain_mask_1_99_percentile_clip_zscore",
        "tabular_features": [*TABULAR_FEATURES, "sex_binary"],
        "published_cache": False,
        "indexed_t1": len(t1_index),
        "catalog_t1_hashes": len(catalog_t1_hashes),
        "catalog_t1_hash_overlap": len(overlap),
        "indexed_preprocessed": len(preprocessed_index),
        "catalog_preprocessed_overlap": len(preprocessed_overlap),
        "t1_dir_exists": (data_root / "T1").exists() if data_root is not None else None,
    }
    if not source_index or (preprocessed_dir is None and not overlap) or (preprocessed_dir is not None and not preprocessed_overlap):
        write_cache_summary(base_summary)
        write_run_manifest(
            "results/logs/mri_volume_cache_manifest.json",
            stage="mri_volume_cache",
            data_root=str(data_root) if data_root is not None else None,
            extra={"summary": base_summary},
        )
        StateManager().append_event("mri_volume_cache", "FAILED", "No matching MRI files were visible for 3D cache generation.")
        print(json.dumps(base_summary, indent=2))
        return 2

    str_dtype = h5py.string_dtype(encoding="utf-8")
    ok = 0
    missing = 0
    errors = 0
    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["method"] = "X"
        h5.attrs["contains_processed_t1"] = True
        h5.attrs["contains_pet_input"] = False
        h5.attrs["contains_raw_identifiers"] = False
        h5.attrs["resolution"] = args.resolution
        h5.attrs["source_mode"] = source_mode
        images = h5.create_dataset(
            "image",
            shape=(n, 1, args.resolution, args.resolution, args.resolution),
            dtype="float16",
            chunks=(1, 1, args.resolution, args.resolution, args.resolution),
            compression="gzip",
            compression_opts=1,
        )
        h5.create_dataset("label", data=np.array([int(float(r["ab_label"])) for r in rows], dtype=np.int8))
        h5.create_dataset("sample_hash", data=np.array([r["sample_hash"] for r in rows], dtype=object), dtype=str_dtype)
        h5.create_dataset("patient_hash", data=np.array([r["patient_hash"] for r in rows], dtype=object), dtype=str_dtype)
        h5.create_dataset("tabular", data=np.array([tabular_vector(r) for r in rows], dtype=np.float32))
        status = h5.create_dataset("status", shape=(n,), dtype=str_dtype)
        for i, row in enumerate(rows):
            source_key = row["sample_hash"] if preprocessed_dir is not None else row["t1_file_hash"]
            path = source_index.get(source_key)
            if path is None:
                missing += 1
                images[i] = np.zeros((1, args.resolution, args.resolution, args.resolution), dtype=np.float16)
                status[i] = "MISSING_T1"
                continue
            try:
                images[i] = load_volume(path, args.resolution, device=args.device)
                status[i] = "OK"
                ok += 1
            except Exception:
                errors += 1
                images[i] = np.zeros((1, args.resolution, args.resolution, args.resolution), dtype=np.float16)
                status[i] = "ERROR"
        h5.flush()

    status_ok = errors == 0 and missing == 0
    if status_ok:
        validate_cache(tmp_path)
        tmp_path.replace(out_path)

    summary = {
        "rows": n,
        "ok": ok,
        "missing": missing,
        "errors": errors,
        "resolution": args.resolution,
        "device_requested": args.device,
        "cache_path": str(out_path),
        "source_mode": source_mode,
        "preprocessed_dir": str(preprocessed_dir) if preprocessed_dir else None,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "contains_processed_t1": True,
        "intensity_normalization": "nonzero_brain_mask_1_99_percentile_clip_zscore",
        "tabular_features": [*TABULAR_FEATURES, "sex_binary"],
        "published_cache": status_ok,
        "indexed_t1": len(t1_index),
        "catalog_t1_hashes": len(catalog_t1_hashes),
        "catalog_t1_hash_overlap": len(overlap),
        "indexed_preprocessed": len(preprocessed_index),
        "catalog_preprocessed_overlap": len(preprocessed_overlap),
        "t1_dir_exists": (data_root / "T1").exists() if data_root is not None else None,
    }
    write_cache_summary(summary)
    write_run_manifest(
        "results/logs/mri_volume_cache_manifest.json",
        stage="mri_volume_cache",
        data_root=str(data_root) if data_root is not None else None,
        extra={"summary": summary},
    )
    StateManager().append_event("mri_volume_cache", "COMPLETED" if status_ok else "FAILED", "Voxel-level T1 cache generated for 3D training.")
    print(json.dumps({"rows": n, "ok": ok, "missing": missing, "errors": errors, "resolution": args.resolution}, indent=2))
    return 0 if status_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
