from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_take(rows: list[dict[str, str]], fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["ab_label"]].append(row)
    taken: list[dict[str, str]] = []
    remaining: list[dict[str, str]] = []
    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        n_take = max(1, round(len(shuffled) * fraction))
        taken.extend(shuffled[:n_take])
        remaining.extend(shuffled[n_take:])
    rng.shuffle(taken)
    rng.shuffle(remaining)
    return taken, remaining


def make_folds(rows: list[dict[str, str]], k: int, seed: int) -> list[list[dict[str, str]]]:
    rng = random.Random(seed)
    folds: list[list[dict[str, str]]] = [[] for _ in range(k)]
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["ab_label"]].append(row)
    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        for i, row in enumerate(shuffled):
            folds[i % k].append(row)
    for fold in folds:
        rng.shuffle(fold)
    return folds


def count(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter(row.get(key, "") for row in rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out_dir", default="splits")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--external_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260510)
    args = ap.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise RuntimeError("Missing master_catalog.csv; run catalog stage first.")
    rows = read_catalog(catalog_path)
    if not rows:
        raise RuntimeError("master_catalog.csv is empty.")
    patient_ids = [row["patient_hash"] for row in rows]
    if len(patient_ids) != len(set(patient_ids)):
        raise RuntimeError("Patient-level split requires one row per patient_hash in current scaffold.")

    external, dev = stratified_take(rows, args.external_fraction, args.seed)
    folds = make_folds(dev, args.folds, args.seed + 1)
    out_dir = Path(args.out_dir)
    fieldnames = list(rows[0].keys())
    write_rows(out_dir / "external_holdout.csv", external, fieldnames)
    for i, test_rows in enumerate(folds):
        test_ids = {row["patient_hash"] for row in test_rows}
        train_rows = [row for row in dev if row["patient_hash"] not in test_ids]
        write_rows(out_dir / f"fold_{i}_test.csv", test_rows, fieldnames)
        write_rows(out_dir / f"fold_{i}_train.csv", train_rows, fieldnames)

    label_by_substudy = Counter((row["substudy"], row["ab_label"]) for row in rows)
    cohort_label_confound = all(
        len({row["ab_label"] for row in rows if row["substudy"] == substudy}) == 1
        for substudy in {row["substudy"] for row in rows}
    )
    summary = {
        "rows_total": len(rows),
        "patients_total": len(set(patient_ids)),
        "external_rows": len(external),
        "dev_rows": len(dev),
        "folds": args.folds,
        "external_label_counts": count(external, "ab_label"),
        "dev_label_counts": count(dev, "ab_label"),
        "external_substudy_counts": count(external, "substudy"),
        "dev_substudy_counts": count(dev, "substudy"),
        "cohort_label_confound": cohort_label_confound,
        "label_by_substudy": {f"{k[0]}|{k[1]}": v for k, v in label_by_substudy.items()},
        "external_holdout_type": "locked_pseudo_external_random_patient_holdout",
        "raw_identifiers_written": False,
    }
    atomic_write_json(out_dir / "split_summary.json", summary)
    md = [
        "# Split Summary\n\n",
        f"- rows_total: `{summary['rows_total']}`\n",
        f"- patients_total: `{summary['patients_total']}`\n",
        f"- external_rows: `{summary['external_rows']}`\n",
        f"- dev_rows: `{summary['dev_rows']}`\n",
        f"- external_label_counts: `{summary['external_label_counts']}`\n",
        f"- dev_label_counts: `{summary['dev_label_counts']}`\n",
        f"- cohort_label_confound: `{summary['cohort_label_confound']}`\n",
        f"- external_holdout_type: `{summary['external_holdout_type']}`\n",
        "\n## Interpretation\n\n",
        "The local catalog has strong cohort-label coupling: A4 is positive and LEARN is negative. ",
        "The external holdout is therefore pseudo-external and must not be described as independent.\n",
    ]
    atomic_write_text(out_dir / "split_summary.md", "".join(md))
    write_run_manifest("results/logs/splits_manifest.json", stage="splits", data_root=None, extra={"summary": summary})
    StateManager().append_event("splits", "COMPLETED", "Patient-level locked pseudo-external split generated.")
    print(f"splits_ready external={len(external)} dev={len(dev)} folds={args.folds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

