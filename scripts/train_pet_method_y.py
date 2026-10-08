from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_baseline import average_precision, brier, roc_auc
from src.data.volume_augmentation import build_augmentation_policy, maybe_wrap_train_dataset
from src.evaluation.threshold_metrics import evaluate_with_train_selected_threshold
from src.models.backbones.lightweight_3d import Lightweight3DEncoder
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.reproducibility import seed_everything, seeded_torch_generator, seed_worker
from src.utils.run_manifest import write_run_manifest


READY_STATUS = "READY_FOR_EXPLORATORY_METHOD_Y_TRAINING"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_sample_hashes(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [row["sample_hash"] for row in read_rows(path)]


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def catalog_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in read_rows(path):
        sample_hash = row.get("sample_hash", "")
        if not sample_hash:
            continue
        labels[sample_hash] = int(float(row["ab_label"]))
    return labels


def cache_index(cache_path: Path) -> dict[str, int]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        return {decode(v): i for i, v in enumerate(h5["sample_hash"][:])}


class H5PETDataset:
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
        image = torch.from_numpy(h5["pet_volume"][idx]).float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        label = torch.tensor(float(self.labels[item]), dtype=torch.float32)
        return image, label


def build_model():
    import torch.nn as nn

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Lightweight3DEncoder(in_channels=1, base_channels=8, out_dim=64)
            self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(64, 1))

        def forward(self, image):
            return self.head(self.encoder(image)).squeeze(-1)

    return _Model()


def split_indices(
    split_path: Path,
    sample_to_idx: dict[str, int],
    labels_by_sample: dict[str, int],
    limit: int = 0,
) -> tuple[list[int], list[int]]:
    indices: list[int] = []
    labels: list[int] = []
    for sample_hash in read_sample_hashes(split_path):
        if sample_hash not in sample_to_idx or sample_hash not in labels_by_sample:
            continue
        indices.append(sample_to_idx[sample_hash])
        labels.append(labels_by_sample[sample_hash])
        if limit > 0 and len(indices) >= limit:
            break
    return indices, labels


def resolve_training_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Method Y PET training, but torch.cuda.is_available() is false")
    return torch.device(requested)


def run_epoch(model, loader, optimizer, device, pos_weight, *, use_amp: bool = False, scaler=None) -> float:
    import torch
    import torch.nn.functional as F

    model.train()
    losses: list[float] = []
    amp_enabled = use_amp and device.type == "cuda"
    for image, label in loader:
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(image)
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


def predict(model, loader, device) -> tuple[list[int], list[float]]:
    import torch

    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for image, label in loader:
            logits = model(image.to(device))
            prob = torch.sigmoid(logits).detach().cpu().tolist()
            y_prob.extend(float(v) for v in prob)
            y_true.extend(int(v) for v in label.tolist())
    return y_true, y_prob


def evaluate(y_true: list[int], y_prob: list[float]) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {"n": len(y_true), "brier": brier(y_true, y_prob)}
    try:
        metrics["roc_auc"] = roc_auc(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = None
    try:
        metrics["pr_auc"] = average_precision(y_true, y_prob)
    except ValueError:
        metrics["pr_auc"] = None
    return metrics


def assert_readiness(path: Path, *, allow_not_ready_for_tests: bool) -> dict:
    readiness = read_json(path)
    status = readiness.get("status")
    if status != READY_STATUS and not allow_not_ready_for_tests:
        raise SystemExit(f"Method Y PET training blocked by readiness status: {status or 'MISSING_READINESS'}")
    return readiness


def main() -> int:
    ap = argparse.ArgumentParser(description="Train gated PET-only Method Y exploratory classifier.")
    ap.add_argument("--tracer", default="fbp")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--readiness", default="results/tables/method_y_training_readiness.json")
    ap.add_argument("--out_dir", default="results/final_results/method_y/pet_training")
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
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--allow_not_ready_for_tests", action="store_true")
    args = ap.parse_args()
    seed_everything(args.seed)

    try:
        import h5py
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("train_pet_method_y requires torch and h5py") from exc

    readiness = assert_readiness(Path(args.readiness), allow_not_ready_for_tests=args.allow_not_ready_for_tests)
    cache_path = Path(args.cache or f"data/cached_method_y_pet/{args.tracer}_pet_96.h5")
    if not cache_path.exists():
        raise SystemExit(f"Missing Method Y PET cache: {cache_path}")
    with h5py.File(cache_path, "r") as h5:
        if "pet_volume" not in h5 or "sample_hash" not in h5:
            raise SystemExit("Method Y PET cache must contain pet_volume and sample_hash datasets.")
        if bool(h5.attrs.get("not_method_x_cache")) is not True:
            raise SystemExit("Refusing PET training cache without not_method_x_cache=true attribute.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_training_device(args.device)
    mode = "FULL_METHOD_Y_PET_TRAINING" if device.type == "cuda" else "CPU_DEBUG_ONLY"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epochs = args.epochs if mode == "FULL_METHOD_Y_PET_TRAINING" else 1
    folds = args.folds if mode == "FULL_METHOD_Y_PET_TRAINING" else 1
    limit = 0 if mode == "FULL_METHOD_Y_PET_TRAINING" else args.cpu_debug_limit
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
    checkpoint_paths: list[str] = []
    for fold in range(folds):
        train_idx, train_labels = split_indices(split_dir / f"fold_{fold}_train.csv", sample_to_idx, labels_by_sample, limit=limit)
        test_limit = max(limit // 4, 16) if limit else 0
        test_idx, test_labels = split_indices(split_dir / f"fold_{fold}_test.csv", sample_to_idx, labels_by_sample, limit=test_limit)
        if not train_idx or not test_idx:
            raise RuntimeError(f"Fold {fold} has no PET cache-overlapping samples.")
        val_n = max(int(len(train_idx) * args.val_fraction), 1)
        val_n = min(val_n, len(train_idx) - 1)
        combined = list(zip(train_idx, train_labels))
        random.Random(args.seed + fold).shuffle(combined)
        train_idx, train_labels = zip(*combined)
        val_idx, val_labels = train_idx[:val_n], train_labels[:val_n]
        fit_idx, fit_labels = train_idx[val_n:], train_labels[val_n:]
        real_train_fold_exposures += len(fit_idx)
        train_ds = maybe_wrap_train_dataset(
            H5PETDataset(cache_path, fit_idx, fit_labels),
            enabled=args.monai_augment,
            image_positions=(0,),
        )
        val_ds = H5PETDataset(cache_path, val_idx, val_labels)
        test_ds = H5PETDataset(cache_path, test_idx, test_labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=seeded_torch_generator(args.seed + fold),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
        )
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
        ckpt = out_dir / f"method_y_pet_{args.tracer}_fold_{fold}.pt"
        torch.save({"model_state_dict": model.state_dict(), "fold": fold, "mode": mode, "tracer": args.tracer}, ckpt)
        checkpoint_paths.append(str(ckpt))

    aggregate = evaluate(all_y, all_p)
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / (1024**3)) if device.type == "cuda" else None
    
    from src.evaluation.conformal_prediction import evaluate_conformal_prediction, conformal_risk_control
    conformal_eval = evaluate_conformal_prediction(all_y, all_p, alpha=0.1)
    conformal_risk = conformal_risk_control(all_y, all_p, target_error_rate=0.1)
    
    aggregate.update(
        {
            "seed": int(args.seed),
            "mode": mode,
            "status": "COMPLETED" if mode == "FULL_METHOD_Y_PET_TRAINING" else "CPU_DEBUG_ONLY",
            "tracer": args.tracer,
            "device": device.type,
            "device_requested": args.device,
            "amp_enabled": amp_enabled,
            "num_workers": args.num_workers,
            "val_fraction": args.val_fraction,
            "epochs": epochs,
            "folds": folds,
            "batch_size": args.batch_size,
            "cache": str(cache_path),
            "model": "MethodYPETOnlyLightweight3D",
            "uses_pet_as_input": True,
            "uses_mri_as_input": False,
            "method": "Y",
            "method_x_claim_unchanged": True,
            "clinical_alzheimer_diagnosis_allowed": False,
            "readiness_status": readiness.get("status"),
            "peak_vram_gb": peak_vram_gb,
            "checkpoints": checkpoint_paths,
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
            "conformal_prediction": conformal_eval,
            "conformal_risk_control": conformal_risk,
        }
    )
    summary = {"status": aggregate["status"], "aggregate": aggregate, "folds": fold_results}
    atomic_write_json(out_dir / "method_y_pet_metrics.json", summary)
    lines = [
        "# Method Y PET Training Metrics\n\n",
        f"- status: `{summary['status']}`\n",
        f"- mode: `{aggregate['mode']}`\n",
        f"- tracer: `{args.tracer}`\n",
        f"- n: `{aggregate['n']}`\n",
        f"- roc_auc: `{aggregate['roc_auc']}`\n",
        f"- pr_auc: `{aggregate['pr_auc']}`\n",
        f"- brier: `{aggregate['brier']}`\n",
        f"- device: `{aggregate['device']}`\n",
        f"- amp_enabled: `{aggregate['amp_enabled']}`\n",
        "- method_x_claim_unchanged: `True`\n",
        "- clinical_alzheimer_diagnosis_allowed: `False`\n",
        f"- conformal_coverage: `{conformal_eval['empirical_coverage']}`\n",
        f"- conformal_avg_set_size: `{conformal_eval['average_prediction_set_size']}`\n",
        f"- conformal_risk_error_rate: `{conformal_risk['empirical_error_rate']}`\n",
    ]
    atomic_write_text(out_dir / "method_y_pet_metrics.md", "".join(lines))
    write_run_manifest("results/logs/method_y_pet_training_manifest.json", stage="method_y_pet_train", data_root=None, extra={"aggregate": aggregate})
    print(json.dumps({"status": summary["status"], "device": device.type, "roc_auc": aggregate["roc_auc"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
