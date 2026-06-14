#!/usr/bin/env python3
"""Y3: Late fusion MRI+PET on Method Y paired subset (separate encoders, concat latents)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_baseline import average_precision, brier, roc_auc
from scripts.train_pet_method_y import (
    H5PETDataset,
    assert_readiness,
    evaluate,
    read_sample_hashes,
    resolve_training_device,
)
from scripts.train_mri_method_y_paired import (
    H5MRIDataset,
    cache_index,
    catalog_labels,
    load_paired_hashes,
)
from src.data.volume_augmentation import build_augmentation_policy, maybe_wrap_train_dataset
from src.evaluation.threshold_metrics import evaluate_with_train_selected_threshold
from src.models.backbones.lightweight_3d import Lightweight3DEncoder
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest


def build_late_fusion_model():
    import torch
    import torch.nn as nn

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mri_encoder = Lightweight3DEncoder(in_channels=1, base_channels=8, out_dim=64)
            self.pet_encoder = Lightweight3DEncoder(in_channels=1, base_channels=8, out_dim=64)
            self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(128, 1))

        def forward(self, mri, pet):
            lat = torch.cat([self.mri_encoder(mri), self.pet_encoder(pet)], dim=-1)
            return self.head(lat).squeeze(-1)

    return _Model()


class H5PairedDataset:
    def __init__(
        self,
        mri_cache: Path,
        pet_cache: Path,
        mri_indices: list[int],
        pet_indices: list[int],
        labels: list[int],
    ) -> None:
        self.mri_ds = H5MRIDataset(mri_cache, mri_indices, labels)
        self.pet_ds = H5PETDataset(pet_cache, pet_indices, labels)

    def __len__(self) -> int:
        return len(self.mri_ds)

    def __getitem__(self, item: int):
        mri, label = self.mri_ds[item]
        pet, _ = self.pet_ds[item]
        return mri, pet, label


def predict_fusion(model, loader, device):
    import torch

    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for mri, pet, label in loader:
            logits = model(mri.to(device), pet.to(device))
            prob = torch.sigmoid(logits).detach().cpu().tolist()
            y_prob.extend(float(v) for v in prob)
            y_true.extend(int(v) for v in label.tolist())
    return y_true, y_prob


def run_fusion_epoch(model, loader, optimizer, device, pos_weight, *, use_amp=False, scaler=None):
    import torch
    import torch.nn.functional as F

    model.train()
    losses: list[float] = []
    amp_enabled = use_amp and device.type == "cuda"
    for mri, pet, label in loader:
        mri = mri.to(device)
        pet = pet.to(device)
        label = label.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(mri, pet)
            loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


def intersect_paired_indices(
    split_path: Path,
    mri_map: dict[str, int],
    pet_map: dict[str, int],
    labels: dict[str, int],
    paired: set[str],
    limit: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    from scripts.train_pet_method_y import read_sample_hashes

    mri_idx, pet_idx, y = [], [], []
    for sample_hash in read_sample_hashes(split_path):
        if sample_hash not in paired or sample_hash not in labels:
            continue
        if sample_hash not in mri_map or sample_hash not in pet_map:
            continue
        mri_idx.append(mri_map[sample_hash])
        pet_idx.append(pet_map[sample_hash])
        y.append(labels[sample_hash])
        if limit > 0 and len(mri_idx) >= limit:
            break
    return mri_idx, pet_idx, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri_cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--pet_cache", default="data/cached_method_y_pet/fbp_pet_96.h5")
    ap.add_argument("--paired_csv", default="data/processed/catalogs/method_y_paired_primary_feasible_90.csv")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--readiness", default="results/tables/method_y_training_readiness.json")
    ap.add_argument("--out_dir", default="results/final_results/method_y/late_fusion_training")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--monai_augment", action="store_true", help="Apply train-fold-only MONAI augmentation.")
    ap.add_argument("--augmentation_exposures_per_real", type=int, default=4)
    ap.add_argument("--allow_not_ready_for_tests", action="store_true")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    readiness = assert_readiness(Path(args.readiness), allow_not_ready_for_tests=args.allow_not_ready_for_tests)
    mri_cache, pet_cache = Path(args.mri_cache), Path(args.pet_cache)
    if not mri_cache.exists() or not pet_cache.exists():
        raise SystemExit("Missing MRI or PET cache for late fusion.")

    paired = load_paired_hashes(Path(args.paired_csv))
    labels = catalog_labels(Path(args.catalog))
    mri_map = cache_index(mri_cache)
    pet_map = cache_index(pet_cache)
    device = resolve_training_device(args.device)
    mode = "FULL_METHOD_Y_LATE_FUSION" if device.type == "cuda" else "CPU_DEBUG_ONLY"
    epochs = args.epochs if mode == "FULL_METHOD_Y_LATE_FUSION" else 1
    folds = args.folds if mode == "FULL_METHOD_Y_LATE_FUSION" else 1
    limit = 0 if mode == "FULL_METHOD_Y_LATE_FUSION" else 32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    all_y, all_p = [], []
    threshold_rows: list[dict] = []
    real_train_fold_exposures = 0
    for fold in range(folds):
        train_m, train_p, train_y = intersect_paired_indices(
            Path(args.splits) / f"fold_{fold}_train.csv", mri_map, pet_map, labels, paired, limit=limit
        )
        test_m, test_p, test_y = intersect_paired_indices(
            Path(args.splits) / f"fold_{fold}_test.csv", mri_map, pet_map, labels, paired, limit=max(limit // 4, 16) if limit else 0
        )
        if not train_m or not test_m:
            raise RuntimeError(f"Fold {fold} has no paired MRI+PET overlap.")
        val_n = max(int(len(train_m) * args.val_fraction), 1)
        val_n = min(val_n, len(train_m) - 1)
        combined = list(zip(train_m, train_p, train_y))
        random.shuffle(combined)
        train_m, train_p, train_y = zip(*combined)
        val_m, val_p, val_y = train_m[:val_n], train_p[:val_n], train_y[:val_n]
        fit_m, fit_p, fit_y = train_m[val_n:], train_p[val_n:], train_y[val_n:]
        real_train_fold_exposures += len(fit_m)
        train_ds = maybe_wrap_train_dataset(
            H5PairedDataset(mri_cache, pet_cache, fit_m, fit_p, fit_y),
            enabled=args.monai_augment,
            image_positions=(0, 1),
        )
        val_ds = H5PairedDataset(mri_cache, pet_cache, val_m, val_p, val_y)
        test_ds = H5PairedDataset(mri_cache, pet_cache, test_m, test_p, test_y)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        pos_weight = torch.tensor([min(max(sum(1 for v in fit_y if v == 0), 1) / max(sum(fit_y), 1), 20.0)], device=device)
        model = build_late_fusion_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
        for _ in range(epochs):
            run_fusion_epoch(model, train_loader, optimizer, device, pos_weight, use_amp=args.amp, scaler=scaler)
        val_true, val_prob = predict_fusion(model, val_loader, device)
        y_true, y_prob = predict_fusion(model, test_loader, device)
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
                "train_n": len(fit_m),
                "val_n": len(val_m),
                "test_n": len(test_m),
                "threshold_metrics": threshold_eval,
            }
        )
        fold_results.append(metrics)
        all_y.extend(y_true)
        all_p.extend(y_prob)

    aggregate = evaluate(all_y, all_p)
    aggregate.update(
        {
            "status": "COMPLETED" if mode == "FULL_METHOD_Y_LATE_FUSION" else "CPU_DEBUG_ONLY",
            "mode": mode,
            "scenario": "late_fusion_mri_pet",
            "uses_pet_as_input": True,
            "uses_mri_as_input": True,
            "readiness_status": readiness.get("status"),
            "pet_label_tautology_caveat": "PET-containing models are evaluated against PET-derived biomarker labels.",
            "method_x_compliant": False,
            "clinical_alzheimer_diagnosis_allowed": False,
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
    atomic_write_json(out_dir / "method_y_late_fusion_metrics.json", summary)
    atomic_write_text(out_dir / "method_y_late_fusion_metrics.md", f"# Y3 Late Fusion\n\n- roc_auc: `{aggregate.get('roc_auc')}`\n")
    write_run_manifest("results/logs/method_y_late_fusion_manifest.json", stage="method_y_late_fusion", data_root=None, extra={"aggregate": aggregate})
    print(json.dumps({"status": summary["status"], "roc_auc": aggregate.get("roc_auc")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
