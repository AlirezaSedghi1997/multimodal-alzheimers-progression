#!/usr/bin/env python3
"""Build a miniature SYNTHETIC Method X cache + splits for pipeline smoke tests.

NON-EVIDENTIARY. This data contains no human measurements (pure RNG volumes,
random labels, shuffled tabular). It exists ONLY to prove execution mechanics
(parse -> backbone build -> train step -> checkpoint -> metrics -> outer eval)
on a new machine (e.g. Colab GPU) before any real-data run. NEVER point an
official battery/validity run at this output, NEVER merge its artifacts into
results/method_x_registry or the battery output root, and NEVER report numbers
from it as scientific evidence.

Cache schema mirrors data/cached/t1_volume_cache_128.h5 minimally:
sample_hash / status / image / tabular. All hashes are "s_smokeNNNN" (real by
sample_source, i.e. no supplementary prefix), all statuses "OK".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAB_COLS = ("age", "sex", "education_years", "apoe4_carrier", "mmse_base")


def build(out_root: Path, n: int, volume: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out_root.mkdir(parents=True, exist_ok=True)
    hashes = [f"s_smoke{i:04d}" for i in range(n)]
    labels = [(i % 2) for i in range(n)]
    order = rng.permutation(n)
    hashes = [hashes[i] for i in order]
    labels = [labels[i] for i in order]
    substudy = ["A4" if i % 2 == 0 else "LEARN" for i in range(n)]
    images = rng.standard_normal((n, 1, volume, volume, volume), dtype=np.float32)
    tabular = np.stack(
        [
            rng.uniform(60.0, 90.0, n).astype(np.float32),  # age
            rng.integers(1, 3, n).astype(np.float32),  # sex
            rng.uniform(6.0, 20.0, n).astype(np.float32),  # education_years
            rng.integers(0, 2, n).astype(np.float32),  # apoe4_carrier
            rng.uniform(20.0, 30.0, n).astype(np.float32),  # mmse_base
        ],
        axis=1,
    )
    cache_path = out_root / "synth_cache.h5"
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(cache_path, "w") as h5:
        h5.create_dataset("sample_hash", data=np.array(hashes, dtype=str_dt))
        h5.create_dataset("status", data=np.array(["OK"] * n, dtype=str_dt))
        h5.create_dataset("image", data=images)
        h5.create_dataset("tabular", data=tabular)
        h5.attrs["synthetic"] = True
        h5.attrs["non_evidentiary"] = True
        h5.attrs["tabular_columns"] = list(TAB_COLS)
        h5.attrs["seed"] = seed
    digest = hashlib.sha256()
    with cache_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    splits_dir = out_root / "splits_synth"
    splits_dir.mkdir(exist_ok=True)
    folds: list[list[int]] = [[] for _ in range(5)]
    for pos, idx in enumerate(range(n)):
        folds[pos % 5].append(idx)
    header = [
        "patient_hash", "sample_hash", "substudy", "ab_label", "pet_day_t0",
        "t1_day_t0", "abs_pet_t1_gap_days", "t1_file_hash", "t1_scan_number",
        "t1_match_strategy", "age", "sex", "education_years", "apoe4_carrier",
        "mmse_base", "has_tabular",
    ]
    for fold in range(5):
        test_idx = set(folds[fold])
        for split_name, members in (
            ("test", sorted(test_idx)),
            ("train", sorted(set(range(n)) - test_idx)),
        ):
            with (splits_dir / f"fold_{fold}_{split_name}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                for i in members:
                    writer.writerow([
                        f"pt_synth{i:04d}", hashes[i], substudy[i], labels[i],
                        0.0, 0.0, 0.0, "t1_synth", 1, "synthetic",
                        float(tabular[i][0]), float(tabular[i][1]),
                        float(tabular[i][2]), float(tabular[i][3]),
                        float(tabular[i][4]), True,
                    ])
    manifest = {
        "synthetic": True,
        "non_evidentiary": True,
        "never_merge_into_battery_registry": True,
        "n": n,
        "volume": volume,
        "seed": seed,
        "cache_sha256": digest.hexdigest(),
        "cache_path": cache_path.relative_to(ROOT).as_posix(),
        "splits_dir": splits_dir.relative_to(ROOT).as_posix(),
        "tabular_columns": list(TAB_COLS),
    }
    (out_root / "synth_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT / "tmp/mx03_synth")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--volume", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    manifest = build(args.out_root.resolve(), int(args.n), int(args.volume), int(args.seed))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
