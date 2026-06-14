#!/usr/bin/env python3
"""Y4: Gated fusion MRI+PET on Method Y paired subset."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_late_fusion_method_y import H5PairedDataset, intersect_paired_indices, predict_fusion, run_fusion_epoch
from scripts.train_mri_method_y_paired import cache_index, catalog_labels, load_paired_hashes
from scripts.train_pet_method_y import assert_readiness, evaluate, resolve_training_device
from src.data.volume_augmentation import build_augmentation_policy, maybe_wrap_train_dataset
from src.evaluation.threshold_metrics import evaluate_with_train_selected_threshold
from src.models.backbones.lightweight_3d import Lightweight3DEncoder
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest


def build_gated_fusion_model():
    import torch
    import torch.nn as nn

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mri_encoder = Lightweight3DEncoder(in_channels=1, base_channels=8, out_dim=64)
            self.pet_encoder = Lightweight3DEncoder(in_channels=1, base_channels=8, out_dim=64)
            self.gate = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.Sigmoid())
            self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(64, 1))

        def forward(self, mri, pet):
            mri_lat = self.mri_encoder(mri)
            pet_lat = self.pet_encoder(pet)
            gate = self.gate(torch.cat([mri_lat, pet_lat], dim=-1))
            fused = gate * mri_lat + (1.0 - gate) * pet_lat
            return self.head(fused).squeeze(-1)

    return _Model()


def main() -> int:
    ap = argparse.ArgumentParser(description="Train gated Method Y MRI+PET fusion classifier.")
    ap.add_argument("--mri_cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--pet_cache", default="data/cached_method_y_pet/fbp_pet_96.h5")
    ap.add_argument("--paired_csv", default="data/processed/catalogs/method_y_paired_primary_feasible_90.csv")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--readiness", default="results/tables/method_y_training_readiness.json")
    ap.add_argument("--out_dir", default="results/final_results/method_y/gated_fusion_training")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--monai_augment", action="store_true")
    ap.add_argument("--augmentation_exposures_per_real", type=int, default=4)
    ap.add_argument("--allow_not_ready_for_tests", action="store_true")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    readiness = assert_readiness(Path(args.readiness), allow_not_ready_for_tests=args.allow_not_ready_for_tests)
    mri_cache, pet_cache = Path(args.mri_cache), Path(args.pet_cache)
    if not mri_cache.exists() or not pet_cache.exists():
        raise SystemExit("Missing MRI or PET cache for gated fusion.")

    paired = load_paired_hashes(Path(args.paired_csv))
    labels = catalog_labels(Path(args.catalog))
    mri_map = cache_index(mri_cache)
    pet_map = cache_index(pet_cache)
    device = resolve_training_device(args.device)
    mode = "FULL_METHOD_Y_GATED_FUSION" if device.type == "cuda" else "CPU_DEBUG_ONLY"
    epochs = args.epochs if mode == "FULL_METHOD_Y_GATED_FUSION" else 1
    folds = args.folds if mode == "FULL_METHOD_Y_GATED_FUSION" else 1
    limit = 0 if mode == "FULL_METHOD_Y_GATED_FUSION" else 32
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    all_y: list[int] = []
    all_p: list[float] = []
    threshold_rows: list[dict] = []
    real_train_fold_exposures = 0
    checkpoint_paths: list[str] = []

    for fold in range(folds):
        train_m, train_p, train_y = intersect_paired_indices(
            Path(args.splits) / f"fold_{fold}_train.csv", mri_map, pet_map, labels, paired, limit=limit
        )
        test_m, test_p, test_y = intersect_paired_indices(
            Path(args.splits) / f"fold_{fold}_test.csv",
            mri_map,
            pet_map,
            labels,
            paired,
            limit=max(limit // 4, 16) if limit else 0,
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
        model = build_gated_fusion_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        losses = []
        for _ in range(epochs):
            losses.append(run_fusion_epoch(model, train_loader, optimizer, device, pos_weight, use_amp=amp_enabled, scaler=scaler))

        val_true, val_prob = predict_fusion(model, val_loader, device)
        y_true, y_prob = predict_fusion(model, test_loader, device)
        threshold_eval = evaluate_with_train_selected_threshold(
            validation_y=val_true,
            validation_prob=val_prob,
            test_y=y_true,
            test_prob=y_prob,
        )
        threshold_rows.append({"fold": fold, **threshold_eval})
        metrics = evaluate(y_true, y_prob)
        metrics.update(
            {
                "fold": fold,
                "train_n": len(fit_m),
                "val_n": len(val_m),
                "test_n": len(test_m),
                "loss_last": losses[-1] if losses else None,
                "threshold_metrics": threshold_eval,
            }
        )
        fold_results.append(metrics)
        all_y.extend(y_true)
        all_p.extend(y_prob)
        ckpt = out_dir / f"method_y_gated_fusion_fold_{fold}.pt"
        torch.save({"model_state_dict": model.state_dict(), "fold": fold, "scenario": "gated_fusion_mri_pet"}, ckpt)
        checkpoint_paths.append(str(ckpt))

    aggregate = evaluate(all_y, all_p)
    aggregate.update(
        {
            "status": "COMPLETED" if mode == "FULL_METHOD_Y_GATED_FUSION" else "CPU_DEBUG_ONLY",
            "mode": mode,
            "scenario": "gated_fusion_mri_pet",
            "uses_pet_as_input": True,
            "uses_mri_as_input": True,
            "method": "Y",
            "method_x_compliant": False,
            "clinical_alzheimer_diagnosis_allowed": False,
            "n": len(all_y),
            "epochs": epochs,
            "folds": folds,
            "readiness_status": readiness.get("status"),
            "pet_label_tautology_caveat": "PET-containing models are evaluated against PET-derived biomarker labels.",
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
            "checkpoints": checkpoint_paths,
        }
    )
    summary = {"status": aggregate["status"], "aggregate": aggregate, "folds": fold_results}
    atomic_write_json(out_dir / "method_y_gated_fusion_metrics.json", summary)
    atomic_write_text(
        out_dir / "method_y_gated_fusion_metrics.md",
        "# Y4 Gated Fusion\n\n"
        f"- status: `{summary['status']}`\n"
        f"- n: `{aggregate.get('n')}`\n"
        f"- roc_auc: `{aggregate.get('roc_auc')}`\n"
        "- Method X compliant: `False`\n"
        "- clinical Alzheimer diagnosis allowed: `False`\n",
    )
    write_run_manifest("results/logs/method_y_gated_fusion_manifest.json", stage="method_y_gated_fusion", data_root=None, extra={"aggregate": aggregate})
    print(json.dumps({"status": summary["status"], "roc_auc": aggregate.get("roc_auc"), "n": aggregate.get("n")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
