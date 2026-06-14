#!/usr/bin/env python3
"""Y2: MRI-only baseline on Method Y paired subset (same grouped splits as Y1)."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_baseline import average_precision, brier, roc_auc
from scripts.train_pet_method_y import (
    READY_STATUS,
    assert_readiness,
    build_model,
    evaluate,
    predict,
    read_json,
    read_rows,
    read_sample_hashes,
    resolve_training_device,
    run_epoch,
    split_indices,
)
from src.data.volume_augmentation import build_augmentation_policy, maybe_wrap_train_dataset
from src.evaluation.threshold_metrics import evaluate_with_train_selected_threshold
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def cache_index(cache_path: Path) -> dict[str, int]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        return {decode(v): i for i, v in enumerate(h5["sample_hash"][:])}


def load_paired_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["sample_hash"] for row in read_rows(path) if row.get("sample_hash")}


class H5MRIDataset:
    def __init__(self, cache_path: Path, indices: list[int], labels: list[int]) -> None:
        self.cache_path = cache_path
        self.indices = indices
        self.labels = labels
        self.h5 = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open(self):
        if self.h5 is None:
            import h5py

            self.h5 = h5py.File(self.cache_path, "r")
        return self.h5

    def __getitem__(self, item: int):
        import torch

        h5 = self._open()
        idx = self.indices[item]
        image = torch.from_numpy(h5["image"][idx]).float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        label = torch.tensor(float(self.labels[item]), dtype=torch.float32)
        return image, label


def split_indices_paired(
    split_path: Path,
    sample_to_idx: dict[str, int],
    labels_by_sample: dict[str, int],
    paired_hashes: set[str],
    limit: int = 0,
) -> tuple[list[int], list[int]]:
    indices: list[int] = []
    labels: list[int] = []
    for sample_hash in read_sample_hashes(split_path):
        if sample_hash not in paired_hashes:
            continue
        if sample_hash not in sample_to_idx or sample_hash not in labels_by_sample:
            continue
        indices.append(sample_to_idx[sample_hash])
        labels.append(labels_by_sample[sample_hash])
        if limit > 0 and len(indices) >= limit:
            break
    return indices, labels


def catalog_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in read_rows(path):
        sample_hash = row.get("sample_hash", "")
        if sample_hash:
            labels[sample_hash] = int(float(row["ab_label"]))
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description="Train MRI-only Method Y baseline on paired subset (Y2).")
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--paired_csv", default="data/processed/catalogs/method_y_paired_primary_feasible_90.csv")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--readiness", default="results/tables/method_y_training_readiness.json")
    ap.add_argument("--out_dir", default="results/final_results/method_y/mri_paired_training")
    ap.add_argument("--pet_cache", default="", help="Optional PET cache used only to restrict Y2 to the common Method Y hash intersection.")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cpu_debug_limit", type=int, default=64)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--monai_augment", action="store_true", help="Apply train-fold-only MONAI augmentation.")
    ap.add_argument("--augmentation_exposures_per_real", type=int, default=4)
    ap.add_argument("--allow_not_ready_for_tests", action="store_true")
    args = ap.parse_args()

    try:
        import h5py
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("train_mri_method_y_paired requires torch and h5py") from exc

    readiness = assert_readiness(Path(args.readiness), allow_not_ready_for_tests=args.allow_not_ready_for_tests)
    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise SystemExit(f"Missing MRI cache: {cache_path}")
    with h5py.File(cache_path, "r") as h5:
        if "image" not in h5 or "sample_hash" not in h5:
            raise SystemExit("MRI cache must contain image and sample_hash datasets.")

    paired_hashes = load_paired_hashes(Path(args.paired_csv))
    if not paired_hashes:
        raise SystemExit(f"Missing or empty paired hash list: {args.paired_csv}")
    common_hash_intersection_source = "paired_csv"
    if args.pet_cache:
        pet_cache_path = Path(args.pet_cache)
        if not pet_cache_path.exists():
            raise SystemExit(f"Missing PET cache for Y2 common intersection restriction: {pet_cache_path}")
        pet_hashes = set(cache_index(pet_cache_path))
        paired_hashes = paired_hashes & pet_hashes
        common_hash_intersection_source = str(pet_cache_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_training_device(args.device)
    mode = "FULL_METHOD_Y_MRI_PAIRED_TRAINING" if device.type == "cuda" else "CPU_DEBUG_ONLY"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epochs = args.epochs if mode == "FULL_METHOD_Y_MRI_PAIRED_TRAINING" else 1
    folds = args.folds if mode == "FULL_METHOD_Y_MRI_PAIRED_TRAINING" else 1
    limit = 0 if mode == "FULL_METHOD_Y_MRI_PAIRED_TRAINING" else args.cpu_debug_limit
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    sample_to_idx = cache_index(cache_path)
    labels_by_sample = catalog_labels(Path(args.catalog))
    split_dir = Path(args.splits)
    fold_results = []
    all_y: list[int] = []
    all_p: list[float] = []
    threshold_rows: list[dict] = []
    real_train_fold_exposures = 0
    for fold in range(folds):
        train_idx, train_labels = split_indices_paired(
            split_dir / f"fold_{fold}_train.csv", sample_to_idx, labels_by_sample, paired_hashes, limit=limit
        )
        test_limit = max(limit // 4, 16) if limit else 0
        test_idx, test_labels = split_indices_paired(
            split_dir / f"fold_{fold}_test.csv", sample_to_idx, labels_by_sample, paired_hashes, limit=test_limit
        )
        if not train_idx or not test_idx:
            raise RuntimeError(f"Fold {fold} has no MRI cache overlap on paired subset.")
        val_n = max(int(len(train_idx) * args.val_fraction), 1)
        val_n = min(val_n, len(train_idx) - 1)
        combined = list(zip(train_idx, train_labels))
        random.shuffle(combined)
        train_idx, train_labels = zip(*combined)
        val_idx, val_labels = train_idx[:val_n], train_labels[:val_n]
        fit_idx, fit_labels = train_idx[val_n:], train_labels[val_n:]
        real_train_fold_exposures += len(fit_idx)
        train_ds = maybe_wrap_train_dataset(
            H5MRIDataset(cache_path, fit_idx, fit_labels),
            enabled=args.monai_augment,
            image_positions=(0,),
        )
        val_ds = H5MRIDataset(cache_path, val_idx, val_labels)
        test_ds = H5MRIDataset(cache_path, test_idx, test_labels)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        positives = max(float(sum(fit_labels)), 1.0)
        negatives = max(float(len(fit_labels) - sum(fit_labels)), 1.0)
        pos_weight = torch.tensor([min(negatives / positives, 20.0)], dtype=torch.float32, device=device)
        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        losses = []
        for _ in range(epochs):
            losses.append(run_epoch(model, train_loader, optimizer, device, pos_weight, use_amp=amp_enabled, scaler=scaler))
        val_true, val_prob = predict(model, val_loader, device)
        y_true, y_prob = predict(model, test_loader, device)
        metrics = evaluate(y_true, y_prob)
        threshold_eval = evaluate_with_train_selected_threshold(
            validation_y=val_true,
            validation_prob=val_prob,
            test_y=y_true,
            test_prob=y_prob,
        )
        threshold_rows.append({"fold": fold, **threshold_eval})
        metrics.update(
            {
                "fold": fold,
                "train_n": len(fit_idx),
                "val_n": len(val_idx),
                "test_n": len(test_idx),
                "loss_last": losses[-1] if losses else None,
                "threshold_metrics": threshold_eval,
            }
        )
        fold_results.append(metrics)
        all_y.extend(y_true)
        all_p.extend(y_prob)

    aggregate = evaluate(all_y, all_p)
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / (1024**3)) if device.type == "cuda" else None
    aggregate.update(
        {
            "mode": mode,
            "status": "COMPLETED" if mode == "FULL_METHOD_Y_MRI_PAIRED_TRAINING" else "CPU_DEBUG_ONLY",
            "device": device.type,
            "amp_enabled": amp_enabled,
            "epochs": epochs,
            "folds": folds,
            "paired_n": len(paired_hashes),
            "common_hash_intersection_source": common_hash_intersection_source,
            "cache": str(cache_path),
            "uses_pet_as_input": False,
            "uses_mri_as_input": True,
            "method": "Y",
            "scenario": "mri_only_paired_subset",
            "readiness_status": readiness.get("status"),
            "peak_vram_gb": peak_vram_gb,
            "augmentation": build_augmentation_policy(
                enabled=args.monai_augment,
                real_train_n=real_train_fold_exposures,
                exposures_per_real=args.augmentation_exposures_per_real,
            ),
            "threshold_selection": {
                "source": "fold_internal_validation",
                "test_labels_used_for_threshold_selection": False,
                "folds": threshold_rows,
            },
        }
    )
    summary = {"status": aggregate["status"], "aggregate": aggregate, "folds": fold_results}
    atomic_write_json(out_dir / "method_y_mri_paired_metrics.json", summary)
    atomic_write_text(
        out_dir / "method_y_mri_paired_metrics.md",
        f"# Method Y MRI Paired Training (Y2)\n\n- status: `{summary['status']}`\n- paired_n: `{len(paired_hashes)}`\n- roc_auc: `{aggregate.get('roc_auc')}`\n",
    )
    write_run_manifest("results/logs/method_y_mri_paired_training_manifest.json", stage="method_y_mri_paired_train", data_root=None, extra={"aggregate": aggregate})
    print(json.dumps({"status": summary["status"], "roc_auc": aggregate.get("roc_auc")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
