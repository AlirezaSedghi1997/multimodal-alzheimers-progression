#!/usr/bin/env python3
"""PET-guided KD trainer: MRI-only student (teacher checkpoint optional; gated on Y3)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_mri_method_y_paired import H5MRIDataset, cache_index, catalog_labels, split_indices_paired
from scripts.train_pet_method_y import evaluate, predict, resolve_training_device, run_epoch
from src.models.backbones.lightweight_3d import Lightweight3DEncoder
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest


def build_student():
    import torch.nn as nn

    class _Student(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Lightweight3DEncoder(in_channels=1, base_channels=8, out_dim=64)
            self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(64, 1))

        def forward(self, x):
            return self.head(self.encoder(x)).squeeze(-1)

    return _Student()


def load_paired_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["sample_hash"] for row in csv.DictReader(handle) if row.get("sample_hash")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", default="results/final_results/method_y/late_fusion_training/method_y_late_fusion_fold0.pt")
    ap.add_argument("--mri_cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--paired_csv", default="data/processed/catalogs/method_y_paired_primary_feasible_90.csv")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out_dir", default="results/final_results/pet_guided_kd")
    ap.add_argument("--kd_lambda", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow_no_teacher", action="store_true", help="CPU smoke: train MRI student without KD term.")
    args = ap.parse_args()

    teacher_path = PROJECT_ROOT / args.teacher_ckpt
    blockers: list[str] = []
    if not teacher_path.exists():
        blockers.append("TEACHER_CHECKPOINT_MISSING")
    mri_cache = PROJECT_ROOT / args.mri_cache
    if not mri_cache.exists():
        blockers.append("MRI_CACHE_MISSING")
    if blockers and not args.allow_no_teacher:
        report = {
            "stage": "pet_guided_kd_train",
            "status": "BLOCKED",
            "blockers": blockers,
            "note": "Run Y3 late fusion first or pass --allow_no_teacher for MRI-only smoke.",
        }
        out = PROJECT_ROOT / args.out_dir
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / "pet_guided_kd_train_summary.json", report)
        print(json.dumps(report, indent=2))
        return 0

    import torch
    from torch.utils.data import DataLoader

    paired = load_paired_hashes(PROJECT_ROOT / args.paired_csv)
    labels = catalog_labels(PROJECT_ROOT / args.catalog)
    sample_map = cache_index(mri_cache)
    device = resolve_training_device(args.device)
    kd_active = not blockers
    mode = "FULL_KD" if device.type == "cuda" and kd_active else "CPU_DEBUG_ONLY"
    epochs = args.epochs if mode == "FULL_KD" else 1
    folds = args.folds if mode == "FULL_KD" else 1
    limit = 0 if mode == "FULL_KD" else 32

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    all_y: list[int] = []
    all_p: list[float] = []
    for fold in range(folds):
        train_idx, train_y = split_indices_paired(
            Path(args.splits) / f"fold_{fold}_train.csv", sample_map, labels, paired, limit=limit
        )
        test_idx, test_y = split_indices_paired(
            Path(args.splits) / f"fold_{fold}_test.csv", sample_map, labels, paired, limit=max(limit // 4, 16) if limit else 0
        )
        if not train_idx or not test_idx:
            continue
        train_loader = DataLoader(H5MRIDataset(mri_cache, train_idx, train_y), batch_size=1, shuffle=True)
        test_loader = DataLoader(H5MRIDataset(mri_cache, test_idx, test_y), batch_size=1, shuffle=False)
        pos_weight = torch.tensor([max(sum(1 for v in train_y if v == 0), 1) / max(sum(train_y), 1)], device=device)
        student = build_student().to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
        for _ in range(epochs):
            run_epoch(student, train_loader, optimizer, device, pos_weight)
        y_true, y_prob = predict(student, test_loader, device)
        metrics = evaluate(y_true, y_prob)
        metrics.update({"fold": fold, "train_n": len(train_idx), "test_n": len(test_idx), "kd_applied": kd_active})
        fold_results.append(metrics)
        all_y.extend(y_true)
        all_p.extend(y_prob)

    aggregate = evaluate(all_y, all_p) if all_y else {"roc_auc": None, "n": 0}
    summary = {
        "stage": "pet_guided_kd_train",
        "status": "COMPLETED" if mode == "FULL_KD" else "CPU_DEBUG_ONLY",
        "mode": mode,
        "blockers": blockers,
        "kd_lambda": args.kd_lambda if kd_active else 0.0,
        "aggregate": aggregate,
        "folds": fold_results,
        "inference": "strictly_PET_free",
        "claim_level": "supplementary_not_method_x_replacement",
    }
    atomic_write_json(out_dir / "pet_guided_kd_train_summary.json", summary)
    atomic_write_text(out_dir / "pet_guided_kd_train_summary.md", f"# PET-Guided KD\n\n- status: {summary['status']}\n- roc_auc: {aggregate.get('roc_auc')}\n")
    write_run_manifest("results/logs/pet_guided_kd_train_manifest.json", stage="pet_guided_kd_train", data_root=None, extra={"aggregate": aggregate})
    print(json.dumps({"status": summary["status"], "roc_auc": aggregate.get("roc_auc"), "blockers": blockers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
