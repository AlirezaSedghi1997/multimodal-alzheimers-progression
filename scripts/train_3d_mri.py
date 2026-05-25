from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_baseline import average_precision, brier, roc_auc
from src.models.backbones.lightweight_3d import MethodX3DClassifier
from src.models.losses.latent_alignment import latent_alignment_loss
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_sample_hashes(path: Path) -> list[str]:
    return [row["sample_hash"] for row in read_rows(path)]


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def cache_index(cache_path: Path) -> dict[str, int]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        return {decode(v): i for i, v in enumerate(h5["sample_hash"][:])}


def cache_resolution(cache_path: Path) -> int | None:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        value = h5.attrs.get("resolution")
        return int(value) if value is not None else None


class H5MRIDataset:
    def __init__(self, cache_path: Path, indices: list[int], tab_mean=None, tab_std=None) -> None:
        self.cache_path = cache_path
        self.indices = indices
        self.h5 = None
        self.tab_mean = tab_mean
        self.tab_std = tab_std

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
        tabular = torch.from_numpy(h5["tabular"][idx]).float()
        if self.tab_mean is not None and self.tab_std is not None:
            tabular = (tabular - self.tab_mean) / self.tab_std
        label = torch.tensor(float(h5["label"][idx]), dtype=torch.float32)
        return image, tabular, label


def tabular_stats(cache_path: Path, indices: list[int]):
    import h5py
    import numpy as np
    import torch

    with h5py.File(cache_path, "r") as h5:
        vals_np = np.stack([h5["tabular"][idx] for idx in indices], axis=0)
        vals = torch.tensor(vals_np, dtype=torch.float32)
    mean = vals.mean(dim=0)
    std = vals.std(dim=0).clamp_min(1.0)
    return mean, std


def split_indices(split_path: Path, sample_to_idx: dict[str, int], limit: int = 0) -> list[int]:
    out = [sample_to_idx[s] for s in read_sample_hashes(split_path) if s in sample_to_idx]
    if limit > 0:
        return out[:limit]
    return out


def unique_indices(values: list[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def run_epoch(model, loader, optimizer, device, pos_weight, latent_alignment_lambda: float = 0.0):
    import torch
    import torch.nn.functional as F

    model.train()
    losses = []
    for image, tabular, label in loader:
        image = image.to(device)
        tabular = tabular.to(device)
        label = label.to(device)
        optimizer.zero_grad(set_to_none=True)
        if latent_alignment_lambda > 0:
            logits, mri_latent, tab_latent = model.forward_with_latents(image, tabular)
        else:
            logits = model(image, tabular)
        loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)
        if latent_alignment_lambda > 0:
            loss = loss + latent_alignment_lambda * latent_alignment_loss(mri_latent, tab_latent)
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
        for image, tabular, label in loader:
            logits = model(image.to(device), tabular.to(device))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_64.h5")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out_dir", default="results/final_results/three_d")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cpu_debug_limit", type=int, default=64)
    ap.add_argument("--fusion", choices=["concat", "cross_attention"], default="cross_attention")
    ap.add_argument("--latent_alignment_lambda", type=float, default=0.0)
    args = ap.parse_args()

    try:
        import h5py
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("train_3d_mri requires torch and h5py") from exc

    cache_path = Path(args.cache)
    split_dir = Path(args.splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "FULL_3D_TRAINING" if device.type == "cuda" else "CPU_DEBUG_ONLY"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epochs = args.epochs if mode == "FULL_3D_TRAINING" else 1
    folds = args.folds if mode == "FULL_3D_TRAINING" else 1
    limit = 0 if mode == "FULL_3D_TRAINING" else args.cpu_debug_limit

    sample_to_idx = cache_index(cache_path)
    fold_results = []
    all_y: list[int] = []
    all_p: list[float] = []
    checkpoint_paths = []
    for fold in range(folds):
        train_idx = split_indices(split_dir / f"fold_{fold}_train.csv", sample_to_idx, limit=limit)
        test_idx = split_indices(split_dir / f"fold_{fold}_test.csv", sample_to_idx, limit=max(limit // 4, 16) if limit else 0)
        if not train_idx or not test_idx:
            raise RuntimeError(f"Fold {fold} has no cache-overlapping samples.")
        tab_mean, tab_std = tabular_stats(cache_path, train_idx)
        train_ds = H5MRIDataset(cache_path, train_idx, tab_mean, tab_std)
        test_ds = H5MRIDataset(cache_path, test_idx, tab_mean, tab_std)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        with h5py.File(cache_path, "r") as h5:
            y_train = [int(h5["label"][idx]) for idx in train_idx]
        positives = max(float(sum(y_train)), 1.0)
        negatives = max(float(len(y_train) - sum(y_train)), 1.0)
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        model = MethodX3DClassifier(fusion=args.fusion).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        losses = []
        for _ in range(epochs):
            losses.append(run_epoch(model, train_loader, optimizer, device, pos_weight, args.latent_alignment_lambda))
        y_true, y_prob = predict(model, test_loader, device)
        metrics = evaluate(y_true, y_prob)
        metrics["fold"] = fold
        metrics["train_n"] = len(train_idx)
        metrics["test_n"] = len(test_idx)
        metrics["loss_last"] = losses[-1] if losses else None
        fold_results.append(metrics)
        all_y.extend(y_true)
        all_p.extend(y_prob)
        ckpt = out_dir / f"method_x_3d_fold_{fold}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "fold": fold,
                "mode": mode,
                "fusion": args.fusion,
                "latent_alignment_lambda": args.latent_alignment_lambda,
                "tab_mean": tab_mean.detach().cpu(),
                "tab_std": tab_std.detach().cpu(),
            },
            ckpt,
        )
        checkpoint_paths.append(str(ckpt))

    external_like = None
    external_path = split_dir / "external_holdout.csv"
    if external_path.exists():
        dev_idx = unique_indices(
            split_indices(split_dir / "fold_0_train.csv", sample_to_idx, limit=limit)
            + split_indices(split_dir / "fold_0_test.csv", sample_to_idx, limit=limit)
        )
        external_idx = split_indices(external_path, sample_to_idx, limit=max(limit // 4, 16) if limit else 0)
        if dev_idx and external_idx:
            tab_mean, tab_std = tabular_stats(cache_path, dev_idx)
            dev_ds = H5MRIDataset(cache_path, dev_idx, tab_mean, tab_std)
            external_ds = H5MRIDataset(cache_path, external_idx, tab_mean, tab_std)
            dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            external_loader = DataLoader(external_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            with h5py.File(cache_path, "r") as h5:
                y_dev = [int(h5["label"][idx]) for idx in dev_idx]
            positives = max(float(sum(y_dev)), 1.0)
            negatives = max(float(len(y_dev) - sum(y_dev)), 1.0)
            pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
            model = MethodX3DClassifier(fusion=args.fusion).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            for _ in range(epochs):
                run_epoch(model, dev_loader, optimizer, device, pos_weight, args.latent_alignment_lambda)
            y_ext, p_ext = predict(model, external_loader, device)
            external_like = evaluate(y_ext, p_ext)
            external_like["n"] = len(y_ext)
            external_like["fit_scope"] = "dev_only"
            external_like["external_used_for_fit"] = False
            ckpt = out_dir / "method_x_3d_dev_external.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": mode,
                    "fit_scope": "dev_only",
                    "fusion": args.fusion,
                    "latent_alignment_lambda": args.latent_alignment_lambda,
                    "tab_mean": tab_mean.detach().cpu(),
                    "tab_std": tab_std.detach().cpu(),
                },
                ckpt,
            )
            checkpoint_paths.append(str(ckpt))

    aggregate = evaluate(all_y, all_p)
    peak_vram_gb = None
    if device.type == "cuda":
        peak_vram_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
    aggregate.update(
        {
            "mode": mode,
            "device": device.type,
            "epochs": epochs,
            "folds": folds,
            "batch_size": args.batch_size,
            "cache": str(cache_path),
            "cache_resolution": cache_resolution(cache_path),
            "model": "MethodX3DClassifier",
            "fusion": args.fusion,
            "latent_alignment_lambda": args.latent_alignment_lambda,
            "ablation_variant": (
                f"{args.fusion}+latent_alignment_{args.latent_alignment_lambda:g}"
                if args.latent_alignment_lambda > 0
                else args.fusion
            ),
            "uses_pet_as_input": False,
            "uses_raw_identifiers": False,
            "checkpoints": checkpoint_paths,
            "external_like": external_like,
            "peak_vram_gb": peak_vram_gb,
            "final_claim_allowed": mode == "FULL_3D_TRAINING",
        }
    )
    summary = {"aggregate": aggregate, "folds": fold_results}
    atomic_write_json(out_dir / "three_d_metrics.json", summary)
    lines = [
        "# 3D MRI Training Metrics\n\n",
        f"- mode: `{mode}`\n",
        f"- device: `{device.type}`\n",
        f"- model: `MethodX3DClassifier`\n",
        f"- fusion: `{args.fusion}`\n",
        f"- latent_alignment_lambda: `{args.latent_alignment_lambda}`\n",
        f"- ablation_variant: `{aggregate['ablation_variant']}`\n",
        f"- n: `{aggregate['n']}`\n",
        f"- roc_auc: `{aggregate['roc_auc']}`\n",
        f"- pr_auc: `{aggregate['pr_auc']}`\n",
        f"- brier: `{aggregate['brier']}`\n",
        f"- peak_vram_gb: `{aggregate['peak_vram_gb']}`\n",
        f"- external_like_roc_auc: `{external_like.get('roc_auc') if external_like else None}`\n",
        f"- final_claim_allowed: `{aggregate['final_claim_allowed']}`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
    ]
    atomic_write_text(out_dir / "three_d_metrics.md", "".join(lines))
    write_run_manifest("results/logs/three_d_manifest.json", stage="train_3d", data_root=None, extra={"aggregate": aggregate})
    StateManager().append_event("train_3d", "COMPLETED" if aggregate["final_claim_allowed"] else "DEBUG_ONLY", "3D MRI training stage completed.")
    print(json.dumps({"mode": mode, "device": device.type, "roc_auc": aggregate["roc_auc"], "final_claim_allowed": aggregate["final_claim_allowed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
