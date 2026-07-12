#!/usr/bin/env python3
"""Delete only explicitly superseded cache artifacts and write an audit manifest."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json


SUPERSEDED = {
    "data/cached/flair_volume_cache_96.h5": "unused legacy 96^3 FLAIR cache; no active code/config consumer",
    "data/cached/t1_combined_128.h5": "global supplementary cache superseded by canonical fold-safe cache",
    "data/cached/t1_combined_foldlocal_4000_128.h5": "June cache stale against July-11 MRI and current five-fold splits",
    "data/cached/t1_foldlocal_aug_128.h5": "June augmentation cache stale and lineage-invalid for current splits",
    "data/cached/t1_header_cache.h5": "legacy header-only preprocessing cache; no active training consumer",
    "data/cached/t1_synthetic_128.h5": "legacy global synthetic cache superseded by materialized fold-local augmentation",
    "data/cached/t1_volume_cache_64.h5": "legacy 64^3 development cache superseded by final 128^3 cache",
    "data/cached/t1_volume_cache_96.h5": "legacy 96^3 development cache superseded by final 128^3 cache",
    "caches/method_y/pet_mni_128.h5": "legacy per-scan-zscore PET cache prohibited for final biomarker work",
    "caches/method_y/pet_native.h5": "unreferenced legacy native PET intermediate",
    "caches/method_y/pet_native_128_broad_s1_fbp_only.h5": "unreferenced non-HQ native PET intermediate",
    "caches/method_y/pet_native_128_broad_s1_fbp_only_hq_v1.h5": "unreferenced native intermediate; validated HQ MNI cache retained",
    "caches/method_y/pet_native_suvr_preserved_v1.h5": "unreferenced native intermediate; diagnostic MNI variant retained",
}

REQUIRED_KEEP = {
    "data/cached/t1_volume_cache_128.h5": "canonical final real-plus-foldlocal-augmentation MRI cache",
    "data/cached/archive/t1_volume_cache_128_real_20260712T203222Z_26fff22aa7de.h5": "immutable July-11 real-only source/provenance backup and Method Y MRI input",
    "caches/method_y/pet_mni_128_broad_s1_fbp_only_hq_v1.h5": "current claim-bearing FBP-only HQ Method Y PET cache",
    "caches/method_y/pet_mni_128_broad_s1_fbp_only.h5": "still referenced by an explicit baseline config",
    "caches/method_y/pet_mni_128_suvr_preserved_v1.h5": "still referenced by diagnostic/ablation configs",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", default="results/tables/cache_cleanup_manifest.json")
    args = parser.parse_args()

    canonical = PROJECT_ROOT / "data/cached/t1_volume_cache_128.h5"
    summary_path = PROJECT_ROOT / "results/tables/fold_local_mri_augmentation_summary.json"
    if not canonical.exists() or not summary_path.exists():
        raise RuntimeError("Canonical cache and validated augmentation summary must exist before cleanup.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PUBLISHED_CANONICAL" or summary.get("validation", {}).get("status") != "PASS":
        raise RuntimeError("Refusing cleanup until the canonical cache has a published PASS validation.")

    root = PROJECT_ROOT.resolve()
    deleted: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    candidates: list[dict[str, object]] = []
    for relative, reason in SUPERSEDED.items():
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Refusing path outside project: {path}")
        if not path.exists():
            missing.append({"path": relative, "reason": reason})
            continue
        record = {
            "path": relative,
            "reason": reason,
            "bytes": int(path.stat().st_size),
            "last_write_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        candidates.append(record)
        if args.execute:
            path.unlink()
            deleted.append(record)

    kept = []
    for relative, reason in REQUIRED_KEEP.items():
        path = PROJECT_ROOT / relative
        kept.append({"path": relative, "reason": reason, "exists": path.exists(), "bytes": int(path.stat().st_size) if path.exists() else 0})
    report = {
        "status": "COMPLETED" if args.execute else "DRY_RUN",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "canonical_cache": "data/cached/t1_volume_cache_128.h5",
        "candidates": candidates,
        "deleted": deleted,
        "deleted_count": len(deleted),
        "deleted_bytes": sum(int(row["bytes"]) for row in deleted),
        "already_missing": missing,
        "kept": kept,
    }
    atomic_write_json(PROJECT_ROOT / args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
