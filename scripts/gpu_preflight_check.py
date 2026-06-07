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


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-flight checks before GPU 3D training/ablation.")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--splits", default="splits/split_summary.json")
    ap.add_argument("--label_freeze", default="results/governance/label_freeze.json")
    ap.add_argument("--out", default="results/governance/gpu_preflight_check.json")
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
    if cache_path.exists():
        try:
            import h5py

            with h5py.File(cache_path, "r") as h5:
                required = {"image", "label", "sample_hash", "patient_hash", "tabular", "status"}
                cache_ok = required.issubset(set(h5.keys()))
                cache_n = int(h5["image"].shape[0])
                cache_res = tuple(int(x) for x in h5["image"].shape[1:])
                cache_hashes = {
                    x.decode() if isinstance(x, bytes) else str(x) for x in h5["sample_hash"][:]
                }
                overlap = len(catalog_hashes & cache_hashes)
                add("cache_catalog_overlap", overlap == len(catalog_hashes), f"{overlap}/{len(catalog_hashes)} catalog samples in cache")
        except Exception as exc:
            add("cache_readable", False, str(exc))
    add("cache_exists_and_valid", cache_ok, f"n={cache_n}, resolution={cache_res}")

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

    if status["failures"]:
        status["status"] = "BLOCKED"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
