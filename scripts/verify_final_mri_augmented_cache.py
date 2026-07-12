#!/usr/bin/env python3
"""Verify the published MRI cache, July-11 source lineage, geometry, and cleanup."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_fold_local_mri_augmented_cache import cache_manifest_digest, decode
from src.utils.io_helpers import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/cached/t1_volume_cache_128.h5")
    parser.add_argument("--real_backup", default="data/cached/archive/t1_volume_cache_128_real_20260712T203222Z_26fff22aa7de.h5")
    parser.add_argument("--output", default="results/tables/final_mri_augmented_cache_verification.json")
    args = parser.parse_args()

    import h5py

    cache = PROJECT_ROOT / args.cache
    backup = PROJECT_ROOT / args.real_backup
    augmentation_summary_path = PROJECT_ROOT / "results/tables/fold_local_mri_augmentation_summary.json"
    geometry_path = PROJECT_ROOT / "results/tables/mri_128_geometry_audit.json"
    cleanup_path = PROJECT_ROOT / "results/tables/cache_cleanup_manifest.json"
    for path in (cache, backup, augmentation_summary_path, geometry_path, cleanup_path):
        if not path.exists():
            raise RuntimeError(f"Required verification artifact is missing: {path}")

    source_digest = cache_manifest_digest(backup)
    augmentation_summary = json.loads(augmentation_summary_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))

    with h5py.File(backup, "r") as source:
        source_hashes = {decode(value) for value in source["sample_hash"][:]}
        source_status_counts = Counter(decode(value) for value in source["status"][:])
        source_rows = int(source["image"].shape[0])

    with h5py.File(cache, "r+") as h5:
        h5.attrs["source_nifti_orientation"] = next(iter(geometry["orientation_counts"]))
        h5.attrs["source_nifti_unique_affine_count"] = int(geometry["unique_affine_count"])
        h5.attrs["source_geometry_audit_status"] = str(geometry["status"])
        h5.attrs["axis_flips_applied"] = False
        rows = int(h5["image"].shape[0])
        statuses = [decode(value) for value in h5["status"][:]]
        sources = [decode(value) for value in h5["source"][:]]
        hashes = [decode(value) for value in h5["sample_hash"][:]]
        parent_hashes = [decode(value) for value in h5["source_sample_hash"][:]]
        source_folds = [int(value) for value in h5["source_fold"][:]]
        attr_source_digest = decode(h5.attrs.get("source_cache_manifest_sha256", ""))
        attr_split_folds = int(h5.attrs.get("split_folds", -1))
        attr_resolution = int(h5.attrs.get("resolution", -1))
        image_shape = list(h5["image"].shape)

    real_rows = sum(source == "real" for source in sources)
    augmented_rows = rows - real_rows
    augmented_parents = [parent for source, parent in zip(sources, parent_hashes, strict=True) if source != "real"]
    checks = {
        "exact_cache_name": cache.name == "t1_volume_cache_128.h5",
        "source_digest_matches_july11_backup": attr_source_digest == source_digest,
        "source_backup_rows_1699": source_rows == 1699,
        "source_backup_status_contract": dict(source_status_counts) == {"OK": 1669, "MRI_QC_HOLD": 28, "MRI_QC_FAIL": 2},
        "final_rows_6213": rows == 6213,
        "eligible_real_rows_1669": real_rows == 1669,
        "augmented_rows_4544": augmented_rows == 4544,
        "all_final_status_ok": set(statuses) == {"OK"},
        "unique_sample_hashes": len(hashes) == len(set(hashes)),
        "all_augmented_parents_in_july11_source": bool(augmented_parents) and set(augmented_parents).issubset(source_hashes),
        "five_fold_contract": attr_split_folds == 5 and set(source_folds) == {-1, 0, 1, 2, 3, 4},
        "resolution_and_shape": attr_resolution == 128 and image_shape[1:] == [1, 128, 128, 128],
        "full_data_validation_pass": augmentation_summary.get("validation", {}).get("status") == "PASS" and augmentation_summary.get("validation", {}).get("checked_images") == rows,
        "geometry_audit_pass": geometry.get("status") == "PASS" and geometry.get("axis_consistent") is True and geometry.get("affine_consistent") is True,
        "cache_cleanup_completed": cleanup.get("status") == "COMPLETED" and cleanup.get("deleted_count") == 13,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "verified_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cache": args.cache,
        "cache_bytes": int(cache.stat().st_size),
        "real_backup": args.real_backup,
        "real_backup_last_write_utc": datetime.fromtimestamp(backup.stat().st_mtime, tz=timezone.utc).isoformat(),
        "source_cache_manifest_sha256": source_digest,
        "rows": rows,
        "real_rows": real_rows,
        "augmented_rows": augmented_rows,
        "source_fold_counts": dict(sorted(Counter(source_folds).items())),
        "checks": checks,
    }
    atomic_write_json(PROJECT_ROOT / args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
