from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.spm_completion_guard import spm_completion_ready, spm_completion_status


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def inspect_cache(cache_path: Path, catalog_hashes: set[str]) -> dict[str, object]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        keys = set(h5.keys())
        base_required = {"image", "label", "sample_hash", "tabular", "status"}
        supplementary_required = {"source", "source_fold", "source_train_only"}
        missing_base = sorted(base_required - keys)
        has_patient_hash = "patient_hash" in keys
        has_supplementary_metadata = supplementary_required.issubset(keys)
        cache_hashes = {decode_h5_string(x) for x in h5["sample_hash"][:]} if "sample_hash" in keys else set()
        if has_supplementary_metadata:
            source_values = [decode_h5_string(x) for x in h5["source"][:]]
            sample_values = [decode_h5_string(x) for x in h5["sample_hash"][:]]
            real_hashes = {sample_hash for sample_hash, source in zip(sample_values, source_values) if source == "real"}
        else:
            real_hashes = cache_hashes
        overlap = len(catalog_hashes & real_hashes)
        cache_n = int(h5["image"].shape[0]) if "image" in keys else 0
        cache_res = tuple(int(x) for x in h5["image"].shape[1:]) if "image" in keys else None
        image_shape_ok = cache_res is not None and len(cache_res) == 4 and cache_res[0] == 1
        metadata_ok = has_patient_hash or has_supplementary_metadata
        return {
            "ok": not missing_base and image_shape_ok and metadata_ok,
            "n": cache_n,
            "resolution": cache_res,
            "overlap": overlap,
            "real_n": len(real_hashes),
            "catalog_n": len(catalog_hashes),
            "mode": "supplementary_combined" if has_supplementary_metadata else "real_catalog",
            "missing": missing_base + ([] if metadata_ok else ["patient_hash_or_supplementary_metadata"]),
            "has_patient_hash": has_patient_hash,
            "has_supplementary_metadata": has_supplementary_metadata,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-flight checks before GPU 3D training/ablation.")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_128.h5")
    ap.add_argument("--splits", default="splits/split_summary.json")
    ap.add_argument("--label_freeze", default="results/governance/label_freeze.json")
    ap.add_argument("--out", default="results/governance/gpu_preflight_check.json")
    ap.add_argument("--require_gpu", action="store_true", help="Fail if CUDA is not available.")
    args = ap.parse_args()

    status = {"schema_version": 1, "status": "PASS", "checks": [], "failures": []}

    def add(name: str, ok: bool, note: str) -> None:
        status["checks"].append({"check": name, "status": "PASS" if ok else "FAIL", "note": note})
        if not ok:
            status["failures"].append(name)

    catalog_path = Path(args.catalog)
    catalog_hashes = set()
    if catalog_path.exists():
        with catalog_path.open("r", encoding="utf-8", newline="") as f:
            catalog_hashes = {r["sample_hash"] for r in csv.DictReader(f)}
    add("master_catalog_exists", catalog_path.exists(), f"{len(catalog_hashes)} catalog rows")

    cache_path = Path(args.cache)
    cache_ok = False
    cache_n = 0
    cache_res = None
    cache_mode = "missing"
    cache_missing: list[str] = []
    if cache_path.exists():
        try:
            cache_info = inspect_cache(cache_path, catalog_hashes)
            cache_ok = bool(cache_info["ok"])
            cache_n = int(cache_info["n"])
            cache_res = cache_info["resolution"]
            cache_mode = str(cache_info["mode"])
            cache_missing = list(cache_info["missing"])
            overlap = int(cache_info["overlap"])
            real_n = int(cache_info["real_n"])
            catalog_n = int(cache_info["catalog_n"])
            if cache_mode == "supplementary_combined":
                # A QC-gated canonical cache intentionally omits HOLD/FAIL rows.
                # Every retained real row must come from the catalog; requiring
                # all catalog rows would incorrectly reject that valid policy.
                overlap_ok = overlap == real_n and 0 < real_n <= catalog_n
                overlap_note = f"{overlap}/{real_n} retained real rows in catalog ({catalog_n} total catalog rows)"
            else:
                overlap_ok = overlap == catalog_n
                overlap_note = f"{overlap}/{catalog_n} catalog samples in cache"
            add("cache_catalog_overlap", overlap_ok, overlap_note)
        except Exception as exc:
            add("cache_readable", False, str(exc))
    add(
        "cache_exists_and_valid",
        cache_ok,
        f"mode={cache_mode}, n={cache_n}, resolution={cache_res}, missing={cache_missing}",
    )

    spm = spm_completion_status()
    add("spm_complete", spm_completion_ready(spm), json.dumps(spm))

    freeze = Path(args.label_freeze)
    freeze_ok = False
    if freeze.exists():
        obj = json.loads(freeze.read_text(encoding="utf-8"))
        freeze_ok = obj.get("status") == "FROZEN_BEFORE_MODELING"
    add("label_frozen", freeze_ok, str(freeze))

    splits_ok = Path(args.splits).exists()
    add("splits_locked", splits_ok, args.splits)

    cuda_ok = False
    cuda_name = None
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            cuda_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        add("cuda_available", False, str(exc))
    else:
        add("cuda_available", cuda_ok, cuda_name or "no device")

    if status["failures"]:
        status["status"] = "BLOCKED"
    elif args.require_gpu and not cuda_ok:
        status["status"] = "BLOCKED"
        status["failures"].append("cuda_available")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
