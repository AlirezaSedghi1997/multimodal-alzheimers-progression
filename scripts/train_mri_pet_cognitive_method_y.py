#!/usr/bin/env python3
"""Y5: MRI+PET+cognitive Method Y exploratory extension on paired subset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_cognitive_features import COGNITIVE_FEATURES
from scripts.train_late_fusion_method_y import H5PairedDataset, intersect_paired_indices
from scripts.train_mri_method_y_paired import cache_index, catalog_labels, load_paired_hashes
from scripts.train_pet_method_y import assert_readiness, evaluate, resolve_training_device
from src.data.volume_augmentation import build_augmentation_policy, maybe_wrap_train_dataset
from src.evaluation.threshold_metrics import evaluate_with_train_selected_threshold
from src.models.backbones.lightweight_3d import Lightweight3DEncoder
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest


SCENARIO = {
    "name": "mri_pet_cognitive",
    "method": "Y",
    "uses_mri_as_input": True,
    "uses_pet_as_input": True,
    "uses_cognitive_as_input": True,
    "method_x_compliant": False,
    "clinical_alzheimer_diagnosis_allowed": False,
}


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_cognitive_feature_matrix(
    cognitive_csv: Path | str = Path("data/processed/cognitive_features.csv"),
    sample_hashes: set[str] | None = None,
    features: list[str] | None = None,
) -> dict[str, list[float | None]]:
    path = Path(cognitive_csv)
    feature_names = list(features or COGNITIVE_FEATURES)
    out: dict[str, list[float | None]] = {}
    if not path.exists():
        return out
    for row in read_rows(path):
        sample_hash = row.get("sample_hash", "")
        if not sample_hash or (sample_hashes is not None and sample_hash not in sample_hashes):
            continue
        out[sample_hash] = [safe_float(row.get(feature)) for feature in feature_names]
    return out


def fit_cognitive_stats(cog_by_hash: dict[str, list[float | None]], sample_hashes: list[str]):
    import torch

    cols = len(next(iter(cog_by_hash.values()))) if cog_by_hash else len(COGNITIVE_FEATURES)
    rows = []
    for sample_hash in sample_hashes:
        values = cog_by_hash.get(sample_hash, [None] * cols)
        rows.append([float("nan") if value is None else float(value) for value in values])
    tensor = torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, cols), dtype=torch.float32)
    if tensor.numel() == 0:
        return torch.zeros(cols), torch.ones(cols)
    valid = torch.isfinite(tensor)
    counts = valid.sum(dim=0).clamp_min(1)
    filled = torch.where(valid, tensor, torch.zeros_like(tensor))
    mean = filled.sum(dim=0) / counts
    centered = torch.where(valid, tensor - mean, torch.zeros_like(tensor))
    std = torch.sqrt((centered * centered).sum(dim=0) / counts).clamp_min(1.0)
    return mean, std


def normalise_cognitive(values: list[float | None], mean, std):
    import torch

    row = torch.tensor([float("nan") if value is None else float(value) for value in values], dtype=torch.float32)
    row = torch.where(torch.isfinite(row), row, mean)
    return (row - mean) / std


class H5PairedCognitiveDataset:
    def __init__(
        self,
        mri_cache: Path,
        pet_cache: Path,
        mri_indices: list[int],
        pet_indices: list[int],
        sample_hashes: list[str],
        labels: list[int],
        cog_by_hash: dict[str, list[float | None]],
        cog_mean,
        cog_std,
    ) -> None:
        self.image_ds = H5PairedDataset(mri_cache, pet_cache, mri_indices, pet_indices, labels)
        self.sample_hashes = sample_hashes
        self.cog_by_hash = cog_by_hash
        self.cog_mean = cog_mean
        self.cog_std = cog_std
        self.cog_dim = len(cog_mean)

    def __len__(self) -> int:
        return len(self.image_ds)

    def __getitem__(self, item: int):
        mri, pet, label = self.image_ds[item]
        values = self.cog_by_hash.get(self.sample_hashes[item], [None] * self.cog_dim)
        cognitive = normalise_cognitive(values, self.cog_mean, self.cog_std)
        return mri, pet, cognitive, label


def build_model(cognitive_dim: int):
    import torch
    import torch.nn as nn

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mri_encoder = Lightweight3DEncoder(in_channels=1, base_channels=8, out_dim=64)
            self.pet_encoder = Lightweight3DEncoder(in_channels=1, base_channels=8, out_dim=64)
            self.cog_encoder = nn.Sequential(
                nn.Linear(cognitive_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.10),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(160, 1))

        def forward(self, mri, pet, cognitive):
            lat = torch.cat([self.mri_encoder(mri), self.pet_encoder(pet), self.cog_encoder(cognitive)], dim=-1)
            return self.head(lat).squeeze(-1)

    return _Model()


def predict(model, loader, device):
    import torch

    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for mri, pet, cognitive, label in loader:
            logits = model(mri.to(device), pet.to(device), cognitive.to(device))
            prob = torch.sigmoid(logits).detach().cpu().tolist()
            y_prob.extend(float(v) for v in prob)
            y_true.extend(int(v) for v in label.tolist())
    return y_true, y_prob


def run_epoch(model, loader, optimizer, device, pos_weight, *, use_amp=False, scaler=None):
    import torch
    import torch.nn.functional as F

    model.train()
    losses: list[float] = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    for mri, pet, cognitive, label in loader:
        mri = mri.to(device)
        pet = pet.to(device)
        cognitive = cognitive.to(device)
        label = label.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(mri, pet, cognitive)
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


def intersect_paired_indices_with_hashes(
    split_path: Path,
    mri_map: dict[str, int],
    pet_map: dict[str, int],
    labels: dict[str, int],
    paired: set[str],
    limit: int = 0,
) -> tuple[list[int], list[int], list[str], list[int]]:
    from scripts.train_pet_method_y import read_sample_hashes

    mri_idx, pet_idx, hashes, y = [], [], [], []
    for sample_hash in read_sample_hashes(split_path):
        if sample_hash not in paired or sample_hash not in labels:
            continue
        if sample_hash not in mri_map or sample_hash not in pet_map:
            continue
        mri_idx.append(mri_map[sample_hash])
        pet_idx.append(pet_map[sample_hash])
        hashes.append(sample_hash)
        y.append(labels[sample_hash])
        if limit > 0 and len(mri_idx) >= limit:
            break
    return mri_idx, pet_idx, hashes, y


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Method Y MRI+PET+cognitive exploratory extension.")
    ap.add_argument("--mri_cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--pet_cache", default="data/cached_method_y_pet/fbp_pet_96.h5")
    ap.add_argument("--cognitive_csv", default="data/processed/cognitive_features.csv")
    ap.add_argument("--paired_csv", default="data/processed/catalogs/method_y_paired_primary_feasible_90.csv")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--readiness", default="results/tables/method_y_training_readiness.json")
    ap.add_argument("--out_dir", default="results/final_results/method_y/mri_pet_cognitive_training")
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
        raise SystemExit("Missing MRI or PET cache for MRI+PET+cognitive Method Y training.")

    paired = load_paired_hashes(Path(args.paired_csv))
    labels = catalog_labels(Path(args.catalog))
    mri_map = cache_index(mri_cache)
    pet_map = cache_index(pet_cache)
    paired = paired & set(mri_map) & set(pet_map)
    cog_by_hash = build_cognitive_feature_matrix(Path(args.cognitive_csv), sample_hashes=paired)
    if not cog_by_hash:
        raise SystemExit(f"Missing cognitive features for paired hashes: {args.cognitive_csv}")

    device = resolve_training_device(args.device)
    mode = "FULL_METHOD_Y_MRI_PET_COGNITIVE" if device.type == "cuda" else "CPU_DEBUG_ONLY"
    epochs = args.epochs if mode == "FULL_METHOD_Y_MRI_PET_COGNITIVE" else 1
    folds = args.folds if mode == "FULL_METHOD_Y_MRI_PET_COGNITIVE" else 1
    limit = 0 if mode == "FULL_METHOD_Y_MRI_PET_COGNITIVE" else 32
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
        train_m, train_p, train_hashes, train_y = intersect_paired_indices_with_hashes(
            Path(args.splits) / f"fold_{fold}_train.csv", mri_map, pet_map, labels, paired, limit=limit
        )
        test_m, test_p, test_hashes, test_y = intersect_paired_indices_with_hashes(
            Path(args.splits) / f"fold_{fold}_test.csv",
            mri_map,
            pet_map,
            labels,
            paired,
            limit=max(limit // 4, 16) if limit else 0,
        )
        if not train_m or not test_m:
            raise RuntimeError(f"Fold {fold} has no paired MRI+PET+cognitive overlap.")
        val_n = max(int(len(train_m) * args.val_fraction), 1)
        val_n = min(val_n, len(train_m) - 1)
        combined = list(zip(train_m, train_p, train_hashes, train_y))
        random.shuffle(combined)
        train_m, train_p, train_hashes, train_y = zip(*combined)
        val_m, val_p, val_hashes, val_y = train_m[:val_n], train_p[:val_n], train_hashes[:val_n], train_y[:val_n]
        fit_m, fit_p, fit_hashes, fit_y = train_m[val_n:], train_p[val_n:], train_hashes[val_n:], train_y[val_n:]
        real_train_fold_exposures += len(fit_m)
        cog_mean, cog_std = fit_cognitive_stats(cog_by_hash, fit_hashes)
        train_ds = maybe_wrap_train_dataset(
            H5PairedCognitiveDataset(mri_cache, pet_cache, fit_m, fit_p, fit_hashes, fit_y, cog_by_hash, cog_mean, cog_std),
            enabled=args.monai_augment,
            image_positions=(0, 1),
        )
        val_ds = H5PairedCognitiveDataset(mri_cache, pet_cache, val_m, val_p, val_hashes, val_y, cog_by_hash, cog_mean, cog_std)
        test_ds = H5PairedCognitiveDataset(mri_cache, pet_cache, test_m, test_p, test_hashes, test_y, cog_by_hash, cog_mean, cog_std)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        pos_weight = torch.tensor([min(max(sum(1 for v in fit_y if v == 0), 1) / max(sum(fit_y), 1), 20.0)], device=device)
        model = build_model(len(COGNITIVE_FEATURES)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        losses = []
        for _ in range(epochs):
            losses.append(run_epoch(model, train_loader, optimizer, device, pos_weight, use_amp=amp_enabled, scaler=scaler))
        val_true, val_prob = predict(model, val_loader, device)
        y_true, y_prob = predict(model, test_loader, device)
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
        ckpt = out_dir / f"method_y_mri_pet_cognitive_fold_{fold}.pt"
        torch.save({"model_state_dict": model.state_dict(), "fold": fold, "scenario": SCENARIO["name"]}, ckpt)
        checkpoint_paths.append(str(ckpt))

    aggregate = evaluate(all_y, all_p)
    aggregate.update(
        {
            **SCENARIO,
            "status": "COMPLETED" if mode == "FULL_METHOD_Y_MRI_PET_COGNITIVE" else "CPU_DEBUG_ONLY",
            "mode": mode,
            "n": len(all_y),
            "folds": folds,
            "epochs": epochs,
            "paired_intersection_n": len(paired),
            "cognitive_feature_count": len(COGNITIVE_FEATURES),
            "readiness_status": readiness.get("status"),
            "small_n_caveat": "Paired Method Y subset only; report N and imbalance with every result.",
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
    atomic_write_json(out_dir / "method_y_mri_pet_cognitive_metrics.json", summary)
    atomic_write_text(
        out_dir / "method_y_mri_pet_cognitive_metrics.md",
        "# Y5 MRI+PET+Cognitive\n\n"
        f"- status: `{summary['status']}`\n"
        f"- n: `{aggregate.get('n')}`\n"
        f"- roc_auc: `{aggregate.get('roc_auc')}`\n"
        "- Method X compliant: `False`\n"
        "- clinical Alzheimer diagnosis allowed: `False`\n"
        "- caveat: PET-containing Method Y branch on PET-derived biomarker label.\n",
    )
    write_run_manifest("results/logs/method_y_mri_pet_cognitive_manifest.json", stage="method_y_mri_pet_cognitive", data_root=None, extra={"aggregate": aggregate})
    print(json.dumps({"status": summary["status"], "roc_auc": aggregate.get("roc_auc"), "n": aggregate.get("n")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
