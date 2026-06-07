from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_mri_volume_cache import build_preprocessed_hash_index
from scripts.build_mri_volume_cache import load_volume, read_rows, tabular_vector, validate_cache
from scripts.spm_completion_guard import MANIFEST_PATH, PREPROCESSED_DIR, manifest_counts, read_json
from src.utils.io_helpers import atomic_write_json, atomic_write_text


DEFAULT_CATALOG = Path("data/processed/master_catalog.csv")
DEFAULT_SUMMARY_JSON = Path("results/tables/mri_volume_cache_smoke_summary.json")
DEFAULT_SUMMARY_MD = Path("results/tables/mri_volume_cache_smoke_summary.md")
DEFAULT_MANIFEST_OUT = Path("results/logs/mri_volume_cache_smoke_manifest.json")
DEFAULT_KEEP_PATH = Path("results/scratch/mri_volume_cache_smoke/t1_volume_cache_smoke_96.h5")

FORBIDDEN_OFFICIAL_OUTPUTS = {
    "data/cached/t1_volume_cache_64.h5",
    "data/cached/t1_volume_cache_96.h5",
    "results/tables/mri_volume_cache_summary.json",
    "results/tables/mri_volume_cache_summary.md",
    "results/logs/mri_volume_cache_manifest.json",
}


def normalized_posix(path: Path) -> str:
    return path.as_posix().lower().lstrip("./")


def ensure_smoke_output_path_allowed(path: Path) -> Path:
    normalized = normalized_posix(path)
    parts = Path(normalized).parts
    if normalized in FORBIDDEN_OFFICIAL_OUTPUTS:
        raise ValueError(f"Refusing to write smoke output over official pipeline artifact: {path}")
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "cached":
        raise ValueError(f"Refusing to write smoke HDF5 under data/cached: {path}")
    return path


def done_sample_hashes(manifest: dict[str, Any] | None) -> set[str]:
    if not manifest:
        return set()
    items = manifest.get("items")
    if not isinstance(items, dict):
        return set()
    return {
        str(sample_hash)
        for sample_hash, item in items.items()
        if isinstance(item, dict) and item.get("status") == "DONE"
    }


def build_smoke_preprocessed_hash_index(preprocessed_dir: Path) -> dict[str, Path]:
    base_index = build_preprocessed_hash_index(preprocessed_dir)
    return {
        sample_hash: path
        for sample_hash, path in base_index.items()
        if not any("_scratch" in part.lower() for part in path.parts)
    }


def collect_smoke_inputs(
    *,
    catalog_path: Path = DEFAULT_CATALOG,
    manifest_path: Path = MANIFEST_PATH,
    preprocessed_dir: Path = PREPROCESSED_DIR,
    limit: int = 8,
) -> tuple[list[dict[str, str]], dict[str, Path], dict[str, int]]:
    manifest = read_json(manifest_path)
    done_hashes = done_sample_hashes(manifest)
    done_count, manifest_total = manifest_counts(manifest)
    rows = read_rows(catalog_path) if catalog_path.exists() else []
    preprocessed_index = build_smoke_preprocessed_hash_index(preprocessed_dir)
    candidates = [
        row
        for row in rows
        if row.get("sample_hash") in done_hashes and row.get("sample_hash") in preprocessed_index
    ]
    selected = candidates[:limit] if limit > 0 else candidates
    counts = {
        "catalog_rows": len(rows),
        "manifest_done_count": int(done_count or 0),
        "manifest_total": int(manifest_total or 0),
        "indexed_preprocessed": len(preprocessed_index),
        "candidate_rows": len(candidates),
        "limit": int(limit),
    }
    return selected, preprocessed_index, counts


def check_runtime_dependencies(device: str) -> str | None:
    missing = []
    for module_name in ["h5py", "nibabel", "numpy", "torch"]:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        return "mri_volume_cache_smoke missing runtime dependencies: " + ", ".join(missing)
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            return "CUDA requested for smoke conversion, but torch.cuda.is_available() is false"
    return None


def build_public_smoke_summary(
    *,
    counts: dict[str, int],
    rows_written: int,
    errors: int,
    resolution: int,
    device: str,
    hdf5_persisted: bool,
    dependency_error: str | None = None,
) -> dict[str, Any]:
    status = "SMOKE_ONLY_NOT_FOR_TRAINING" if rows_written > 0 and errors == 0 and dependency_error is None else "SMOKE_FAILED_NOT_FOR_TRAINING"
    return {
        "status": status,
        "purpose": "dependency_and_volume_loading_smoke_test_only",
        "official_cache_written": False,
        "training_usable": False,
        "published_cache": False,
        "subject_level_outputs_exported": False,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "raw_paths_exported": False,
        "hdf5_persisted": hdf5_persisted,
        "hdf5_location_disclosed": False,
        "gpu_policy": "device=auto uses CUDA when torch.cuda.is_available(); device=cuda fails fast if CUDA is unavailable",
        "dependency_check_passed": dependency_error is None,
        "dependency_error": dependency_error,
        "device_requested": device,
        "resolution": resolution,
        "rows_written": rows_written,
        "errors": errors,
        **counts,
    }


def write_public_smoke_summary(summary: dict[str, Any], *, summary_json: Path, summary_md: Path) -> None:
    ensure_smoke_output_path_allowed(summary_json)
    ensure_smoke_output_path_allowed(summary_md)
    atomic_write_json(summary_json, summary)
    md = [
        "# MRI Volume Cache Smoke Summary\n\n",
        "This is a smoke-only dependency and image-loading check. It is not a Method X training cache.\n\n",
        f"- status: `{summary['status']}`\n",
        f"- official_cache_written: `{summary['official_cache_written']}`\n",
        f"- training_usable: `{summary['training_usable']}`\n",
        f"- dependency_check_passed: `{summary['dependency_check_passed']}`\n",
        f"- hdf5_persisted: `{summary['hdf5_persisted']}`\n",
        f"- rows_written: `{summary['rows_written']}`\n",
        f"- errors: `{summary['errors']}`\n",
        f"- manifest_done_count: `{summary['manifest_done_count']}`\n",
        f"- manifest_total: `{summary['manifest_total']}`\n",
        f"- indexed_preprocessed: `{summary['indexed_preprocessed']}`\n",
        f"- candidate_rows: `{summary['candidate_rows']}`\n",
        f"- resolution: `{summary['resolution']}`\n",
        f"- device_requested: `{summary['device_requested']}`\n",
        "- contains_pet_input: `False`\n",
        "- contains_raw_identifiers: `False`\n",
        "- raw_paths_exported: `False`\n",
        "- subject_level_outputs_exported: `False`\n",
    ]
    atomic_write_text(summary_md, "".join(md))


def write_smoke_hdf5(
    *,
    rows: list[dict[str, str]],
    preprocessed_index: dict[str, Path],
    out_path: Path,
    resolution: int,
    device: str,
) -> tuple[int, int]:
    import h5py
    import numpy as np

    ensure_smoke_output_path_allowed(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    rows_written = 0
    errors = 0
    with h5py.File(out_path, "w") as h5:
        h5.attrs["method"] = "X"
        h5.attrs["smoke_only_not_for_training"] = True
        h5.attrs["official_cache_written"] = False
        h5.attrs["contains_pet_input"] = False
        h5.attrs["contains_raw_identifiers"] = False
        h5.attrs["resolution"] = resolution
        h5.attrs["source_mode"] = "preprocessed_mri_smoke"
        images = h5.create_dataset(
            "image",
            shape=(len(rows), 1, resolution, resolution, resolution),
            dtype="float16",
            chunks=(1, 1, resolution, resolution, resolution),
            compression="gzip",
            compression_opts=1,
        )
        h5.create_dataset("label", data=np.array([int(float(row["ab_label"])) for row in rows], dtype=np.int8))
        h5.create_dataset("sample_hash", data=np.array([row["sample_hash"] for row in rows], dtype=object), dtype=str_dtype)
        h5.create_dataset("patient_hash", data=np.array([row["patient_hash"] for row in rows], dtype=object), dtype=str_dtype)
        h5.create_dataset("tabular", data=np.array([tabular_vector(row) for row in rows], dtype=np.float32))
        status = h5.create_dataset("status", shape=(len(rows),), dtype=str_dtype)
        for idx, row in enumerate(rows):
            source_path = preprocessed_index.get(row["sample_hash"])
            try:
                if source_path is None:
                    raise FileNotFoundError(row["sample_hash"])
                images[idx] = load_volume(source_path, resolution, device=device)
                status[idx] = "OK"
                rows_written += 1
            except Exception:
                images[idx] = np.zeros((1, resolution, resolution, resolution), dtype=np.float16)
                status[idx] = "ERROR"
                errors += 1
        h5.flush()
    validate_cache(out_path)
    return rows_written, errors


def run_smoke(
    *,
    catalog_path: Path,
    manifest_path: Path,
    preprocessed_dir: Path,
    limit: int,
    resolution: int,
    device: str,
    keep_hdf5: bool,
    out_path: Path,
    summary_json: Path,
    summary_md: Path,
    manifest_out: Path,
) -> int:
    selected, preprocessed_index, counts = collect_smoke_inputs(
        catalog_path=catalog_path,
        manifest_path=manifest_path,
        preprocessed_dir=preprocessed_dir,
        limit=limit,
    )
    rows_written = 0
    errors = 0
    dependency_error = None
    if selected:
        dependency_error = check_runtime_dependencies(device)
        if dependency_error:
            errors = len(selected)
        else:
            if keep_hdf5:
                ensure_smoke_output_path_allowed(out_path)
                rows_written, errors = write_smoke_hdf5(
                    rows=selected,
                    preprocessed_index=preprocessed_index,
                    out_path=out_path,
                    resolution=resolution,
                    device=device,
                )
            else:
                with tempfile.TemporaryDirectory(prefix="mri_volume_cache_smoke_") as temp_dir:
                    temp_out = Path(temp_dir) / "t1_volume_cache_smoke.h5"
                    rows_written, errors = write_smoke_hdf5(
                        rows=selected,
                        preprocessed_index=preprocessed_index,
                        out_path=temp_out,
                        resolution=resolution,
                        device=device,
                    )
    summary = build_public_smoke_summary(
        counts=counts,
        rows_written=rows_written,
        errors=errors,
        resolution=resolution,
        device=device,
        hdf5_persisted=keep_hdf5 and bool(selected) and dependency_error is None,
        dependency_error=dependency_error,
    )
    write_public_smoke_summary(summary, summary_json=summary_json, summary_md=summary_md)
    ensure_smoke_output_path_allowed(manifest_out)
    atomic_write_json(
        manifest_out,
        {
            "stage": "mri_volume_cache_smoke",
            "status": summary["status"],
            "official_cache_written": False,
            "training_usable": False,
            "summary": summary,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if selected and errors == 0 and dependency_error is None else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--preprocessed_dir", default=str(PREPROCESSED_DIR))
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=96)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--keep_hdf5", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_KEEP_PATH))
    ap.add_argument("--summary_json", default=str(DEFAULT_SUMMARY_JSON))
    ap.add_argument("--summary_md", default=str(DEFAULT_SUMMARY_MD))
    ap.add_argument("--manifest_out", default=str(DEFAULT_MANIFEST_OUT))
    args = ap.parse_args()

    return run_smoke(
        catalog_path=Path(args.catalog),
        manifest_path=Path(args.manifest),
        preprocessed_dir=Path(args.preprocessed_dir),
        limit=args.limit,
        resolution=args.resolution,
        device=args.device,
        keep_hdf5=args.keep_hdf5,
        out_path=Path(args.out),
        summary_json=Path(args.summary_json),
        summary_md=Path(args.summary_md),
        manifest_out=Path(args.manifest_out),
    )


if __name__ == "__main__":
    raise SystemExit(main())