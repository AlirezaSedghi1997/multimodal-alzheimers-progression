#!/usr/bin/env python3
"""Train Method X 3D MRI + tabular fusion with optional SynIB and latent alignment."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_baseline import average_precision, brier, roc_auc
from scripts.train_pet_method_y import resolve_training_device
from src.data.volume_augmentation import build_augmentation_policy, maybe_wrap_train_dataset
from src.evaluation.threshold_metrics import evaluate_with_train_selected_threshold
from src.evaluation.tta import TTAPredictor
from src.models.backbones.lightweight_3d import MethodX3DClassifier
from src.models.losses.latent_alignment import latent_alignment_loss
from src.models.losses.contrastive_alignment import ContrastiveAlignmentLoss
from src.models.losses.synib_synergy_loss import SynIBConfig, synib_multimodal_loss
from src.utils.final_claim_policy import evaluate_three_d_final_claim
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_sample_hashes(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [row["sample_hash"] for row in read_rows(path) if row.get("sample_hash")]


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def cache_index(cache_path: Path) -> dict[str, int]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        return {decode(v): i for i, v in enumerate(h5["sample_hash"][:])}


def ok_indices(cache_path: Path) -> set[int]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        return {i for i, v in enumerate(h5["status"][:]) if decode(v) == "OK"}


class H5MethodXDataset:
    def __init__(self, cache_path: Path, indices: list[int], labels: list[int]) -> None:
        self.cache_path = cache_path
        self.indices = indices
        self.labels = labels
        self._h5 = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open(self):
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self.cache_path, "r")
        return self._h5

    def __getitem__(self, item: int):
        import torch

        h5 = self._open()
        idx = self.indices[item]
        image = torch.from_numpy(h5["image"][idx].astype("float32"))
        tabular = torch.from_numpy(h5["tabular"][idx].astype("float32"))
        label = torch.tensor(float(self.labels[item]), dtype=torch.float32)
        return image, tabular, label


def split_indices(
    split_path: Path,
    sample_to_idx: dict[str, int],
    ok: set[int],
    limit: int = 0,
) -> tuple[list[int], list[int]]:
    indices: list[int] = []
    labels: list[int] = []
    for row in read_rows(split_path):
        sample_hash = row.get("sample_hash", "")
        if sample_hash not in sample_to_idx:
            continue
        idx = sample_to_idx[sample_hash]
        if idx not in ok:
            continue
        indices.append(idx)
        labels.append(int(float(row["ab_label"])))
        if limit > 0 and len(indices) >= limit:
            break
    return indices, labels


def tabular_stats(cache_path: Path, indices: list[int]):
    import torch

    if not indices:
        return None, None
    dataset = H5MethodXDataset(cache_path, indices, [0] * len(indices))
    rows = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)
    return rows.mean(dim=0), rows.std(dim=0).clamp_min(1.0)


def normalize_tabular(tabular, tab_mean, tab_std):
    if tab_mean is None or tab_std is None:
        return tabular
    tab_mean = tab_mean.to(tabular.device)
    tab_std = tab_std.to(tabular.device)
    return (tabular - tab_mean) / tab_std


def compute_loss(
    model,
    image,
    tabular,
    label,
    *,
    pos_weight,
    latent_alignment_lambda: float,
    synib_cfg: SynIBConfig | None,
    use_synib: bool,
    contrastive_alignment: bool = False,
    contrastive_temperature: float = 0.07,
    contrastive_loss_fn=None,
):
    import torch
    import torch.nn.functional as F

    mri_latent, tab_raw = model.encode(image, tabular)
    logits = model.logits_from_latents(mri_latent, tab_raw)
    aligned_tab = model.alignment_projection(tab_raw)
    task = F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)
    loss = task
    parts: dict[str, float] = {"task_loss": float(task.detach())}

    align_term = None
    if contrastive_alignment and contrastive_loss_fn is not None:
        align = contrastive_loss_fn(mri_latent, aligned_tab)
        align_term = latent_alignment_lambda * align if latent_alignment_lambda > 0.0 else align
        loss = loss + align_term
        parts["contrastive_alignment"] = float(align.detach())
    elif latent_alignment_lambda > 0.0:
        align = latent_alignment_loss(mri_latent, aligned_tab)
        align_term = latent_alignment_lambda * align
        loss = loss + align_term
        parts["latent_alignment"] = float(align.detach())

    if use_synib and synib_cfg is not None and synib_cfg.lambda_syn > 0.0:
        zeros_mri = torch.zeros_like(mri_latent)
        zeros_tab = torch.zeros_like(tab_raw)
        logits_tab = model.logits_from_latents(zeros_mri, tab_raw)
        logits_mri = model.logits_from_latents(mri_latent, zeros_tab)
        syn_total, syn_parts = synib_multimodal_loss(logits, logits_tab, logits_mri, label, config=synib_cfg, pos_weight=pos_weight)
        loss = syn_total
        if align_term is not None:
            loss = loss + align_term
        parts.update({f"synib_{k}": v for k, v in syn_parts.items()})

    parts["total"] = float(loss.detach())
    return loss, parts


def eval_loss(
    model,
    loader,
    device,
    pos_weight,
    *,
    tab_mean,
    tab_std,
) -> float:
    import torch
    import torch.nn.functional as F

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for image, tabular, label in loader:
            image = image.to(device)
            tabular = normalize_tabular(tabular.to(device), tab_mean, tab_std)
            label = label.to(device)
            logits = model(image, tabular)
            losses.append(float(F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)))
    return float(sum(losses) / max(len(losses), 1))


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    pos_weight,
    *,
    latent_alignment_lambda: float,
    synib_cfg: SynIBConfig | None,
    use_synib: bool,
    tab_mean,
    tab_std,
    use_amp: bool = False,
    scaler=None,
    max_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
    label_smoothing: float = 0.0,
    contrastive_alignment: bool = False,
    contrastive_temperature: float = 0.07,
    contrastive_loss_fn=None,
) -> float:
    import torch

    model.train()
    losses: list[float] = []
    amp_enabled = use_amp and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    for step, (image, tabular, label) in enumerate(loader):
        image = image.to(device)
        tabular = normalize_tabular(tabular.to(device), tab_mean, tab_std)
        label = label.to(device)
        if label_smoothing > 0.0:
            label = label * (1.0 - label_smoothing) + 0.5 * label_smoothing
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss, _ = compute_loss(
                model,
                image,
                tabular,
                label,
                pos_weight=pos_weight,
                latent_alignment_lambda=latent_alignment_lambda,
                synib_cfg=synib_cfg,
                use_synib=use_synib,
                contrastive_alignment=contrastive_alignment,
                contrastive_temperature=contrastive_temperature,
                contrastive_loss_fn=contrastive_loss_fn,
            )
            loss = loss / grad_accum_steps
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None and amp_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()) * grad_accum_steps)
    return float(sum(losses) / max(len(losses), 1))


def predict(model, loader, device, tab_mean, tab_std, use_tta: bool = False, tta_augmentations: int = 5) -> tuple[list[int], list[float]]:
    import torch

    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    tta = TTAPredictor(model, n_augmentations=tta_augmentations) if use_tta else None
    with torch.no_grad():
        for image, tabular, label in loader:
            image = image.to(device)
            tabular = normalize_tabular(tabular.to(device), tab_mean, tab_std)
            if tta is not None:
                probs = tta.predict(image, tabular)
            else:
                logits = model(image, tabular)
                probs = torch.sigmoid(logits)
            y_true.extend(int(v) for v in label.tolist())
            y_prob.extend(float(v) for v in probs.detach().cpu().tolist())
    return y_true, y_prob


def resolve_mode(device, requested_phase: str, limit: int) -> tuple[str, int, int, int]:
    if device.type != "cuda":
        return "CPU_DEBUG_ONLY", 1, 1, max(limit, 32)
    if requested_phase != "final":
        return "FULL_3D_TRAINING", 0, 0, 0
    return "FULL_3D_TRAINING", 0, 0, 0


def build_aggregate(
    *,
    mode: str,
    args,
    device,
    epochs: int,
    folds: int,
    use_synib: bool,
    use_amp: bool,
    cache_path: Path,
    cache_resolution: int | None,
    fold_metrics: list[dict],
    all_y: list[int],
    all_p: list[float],
    checkpoints: list[str],
    augmentation_policy: dict | None = None,
    threshold_rows: list[dict] | None = None,
) -> dict:
    claim_policy = evaluate_three_d_final_claim(
        mode=mode,
        phase=args.phase,
        resolution=args.resolution,
        folds=folds,
        cache_resolution=int(cache_resolution) if cache_resolution is not None else None,
    )
    return {
        "model": "MethodX3DClassifier",
        "mode": mode,
        "phase": args.phase,
        "fusion": args.fusion,
        "backbone": args.backbone,
        "input_mode": args.input_mode,
        "latent_alignment_lambda": args.latent_alignment_lambda,
        "synib_lambda": args.synib_lambda,
        "modality_dropout": args.modality_dropout,
        "literature_anchor_synib": "LIT019 arXiv:2606.09853" if use_synib else None,
        "epochs": epochs,
        "folds": folds,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "early_stopping_patience": args.early_stopping_patience,
        "requested_resolution": args.resolution,
        "cache_resolution": cache_resolution,
        "cache": str(Path(args.cache)),
        "device_requested": args.device,
        "device": device.type,
        "amp_enabled": use_amp,
        "num_workers": args.num_workers,
        "augmentation": augmentation_policy
        or build_augmentation_policy(
            enabled=args.monai_augment,
            real_train_n=sum(int(m.get("train_n") or 0) for m in fold_metrics),
            exposures_per_real=args.augmentation_exposures_per_real,
        ),
        "threshold_selection": {
            "source": "fold_internal_validation",
            "test_labels_used_for_threshold_selection": False,
            "folds": threshold_rows or [],
        },
        "uses_pet_as_input": False,
        "uses_synthetic_pet_as_input": False,
        "uses_raw_identifiers": False,
        "n": len(all_y),
        "roc_auc": roc_auc(all_y, all_p) if all_y else None,
        "pr_auc": average_precision(all_y, all_p) if all_y else None,
        "brier": brier(all_y, all_p) if all_y else None,
        "checkpoints": checkpoints,
        "final_claim_allowed": claim_policy["final_claim_allowed"],
        "claim_policy": claim_policy,
        "final_contract": {
            "final_requires_96": True,
            "final_requires_5fold": True,
            "pet_input_allowed": False,
            "synthetic_pet_input_allowed": False,
        },
    }


def write_metrics_payload(
    out_dir: Path,
    aggregate: dict,
    fold_metrics: list[dict],
    *,
    status: str = "IN_PROGRESS",
) -> None:
    payload = {"aggregate": {**aggregate, "status": status}, "folds": fold_metrics}
    atomic_write_json(out_dir / "three_d_metrics.json", payload)


def evaluate_saved_fold(
    ckpt_path: Path,
    *,
    fold: int,
    cache_path: Path,
    split_dir: Path,
    sample_to_idx: dict[str, int],
    ok: set[int],
    per_split_limit: int,
    device,
    args,
    epochs: int,
) -> tuple[dict, list[int], list[float]]:
    import torch
    from torch.utils.data import DataLoader

    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    train_idx, train_y = split_indices(split_dir / f"fold_{fold}_train.csv", sample_to_idx, ok, limit=per_split_limit)
    test_idx, test_y = split_indices(
        split_dir / f"fold_{fold}_test.csv",
        sample_to_idx,
        ok,
        limit=max(per_split_limit // 4, 8) if per_split_limit else 0,
    )
    val_n = max(int(len(train_idx) * args.val_fraction), 1)
    val_n = min(val_n, len(train_idx) - 1)
    fit_idx = train_idx[val_n:]
    tab_mean = payload.get("tab_mean")
    tab_std = payload.get("tab_std")
    model = MethodX3DClassifier(fusion=args.fusion, backbone=args.backbone).to(device)
    model.load_state_dict(payload["model_state_dict"])
    test_loader = DataLoader(
        H5MethodXDataset(cache_path, test_idx, test_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    y_true, y_prob = predict(model, test_loader, device, tab_mean, tab_std, use_tta=args.tta, tta_augmentations=args.tta_augmentations)
    metrics = {
        "fold": fold,
        "n": len(y_true),
        "test_n": len(y_true),
        "train_n": len(fit_idx),
        "val_n": val_n,
        "roc_auc": roc_auc(y_true, y_prob),
        "pr_auc": average_precision(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "best_val_loss": None,
        "val_loss_last": None,
        "loss_last": None,
        "epochs_run": payload.get("epochs_run"),
        "early_stopped": None,
        "train_loss_curve": [],
        "val_loss_curve": [],
        "synib_lambda": args.synib_lambda,
        "modality_dropout": args.modality_dropout,
        "resumed_from_checkpoint": True,
    }
    if metrics["epochs_run"] is None:
        metrics["epochs_run"] = epochs
    return metrics, y_true, y_prob


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_64.h5")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out_dir", default="results/final_results/three_d")
    ap.add_argument("--phase", choices=["dev", "final"], default="dev")
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument("--fusion", default="cross_attention", choices=["concat", "cross_attention", "moe"])
    ap.add_argument("--backbone", default="resnet18_3d", choices=["tiny", "resnet10_3d", "resnet18_3d", "densenet121_3d", "efficientnet3d_b0"])
    ap.add_argument("--latent_alignment_lambda", type=float, default=0.0)
    ap.add_argument("--contrastive_alignment", action="store_true", help="Use InfoNCE contrastive alignment instead of MSE latent alignment.")
    ap.add_argument("--contrastive_temperature", type=float, default=0.07, help="Temperature for contrastive alignment loss.")
    ap.add_argument("--tta", action="store_true", help="Apply test-time augmentation during evaluation.")
    ap.add_argument("--tta_augmentations", type=int, default=5, help="Number of TTA augmentations.")
    ap.add_argument("--synib_lambda", type=float, default=0.0)
    ap.add_argument("--modality_dropout", type=float, default=0.5)
    ap.add_argument("--input_mode", default="mri_plus_tabular")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="Optional per-split cap for CPU smoke.")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps for larger effective batch size.")
    ap.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup epochs before cosine annealing.")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor (0.0 = no smoothing).")
    ap.add_argument("--monai_augment", action="store_true", help="Apply train-fold-only MONAI augmentation to T1 MRI.")
    ap.add_argument("--augmentation_exposures_per_real", type=int, default=4)
    ap.add_argument(
        "--no_resume",
        action="store_true",
        help="Retrain all folds even when method_x_3d_fold_*.pt checkpoints already exist.",
    )
    args = ap.parse_args()

    cache_path = PROJECT_ROOT / args.cache
    if not cache_path.exists():
        report = {"status": "BLOCKED", "blocker": "MRI_CACHE_MISSING", "cache": str(cache_path)}
        out_dir = PROJECT_ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "three_d_metrics.json", {"aggregate": report})
        print(json.dumps(report, indent=2))
        return 2

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception:
        report = {"status": "BLOCKED", "blocker": "PYTORCH_MISSING"}
        out_dir = PROJECT_ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "three_d_metrics.json", {"aggregate": report})
        print(json.dumps(report, indent=2))
        return 1

    device = resolve_training_device(args.device)
    mode, epoch_cap, fold_cap, default_limit = resolve_mode(device, args.phase, args.limit)
    epochs = epoch_cap or args.epochs
    folds = fold_cap or args.folds
    per_split_limit = args.limit or default_limit

    sample_to_idx = cache_index(cache_path)
    ok = ok_indices(cache_path)
    split_dir = PROJECT_ROOT / args.splits
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    synib_cfg = SynIBConfig(lambda_syn=args.synib_lambda, mask_prob=args.modality_dropout, tabular_dim=5, mri_dim=0)
    use_synib = args.synib_lambda > 0.0
    use_amp = args.amp and device.type == "cuda"
    contrastive_loss_fn = ContrastiveAlignmentLoss(temperature=args.contrastive_temperature) if args.contrastive_alignment else None

    fold_metrics: list[dict] = []
    all_y: list[int] = []
    all_p: list[float] = []
    checkpoints: list[str] = []
    threshold_rows: list[dict] = []

    cache_resolution = args.resolution
    summary_path = PROJECT_ROOT / "results" / "tables" / "mri_volume_cache_summary.json"
    if summary_path.exists():
        cache_resolution = json.loads(summary_path.read_text(encoding="utf-8")).get("resolution", args.resolution)
    try:
        import h5py

        with h5py.File(cache_path, "r") as h5:
            if "resolution" in h5.attrs:
                cache_resolution = int(h5.attrs["resolution"])
    except Exception:
        pass

    for fold in range(folds):
        ckpt_path = out_dir / f"method_x_3d_fold_{fold}.pt"
        if ckpt_path.exists() and not args.no_resume:
            metrics, y_true, y_prob = evaluate_saved_fold(
                ckpt_path,
                fold=fold,
                cache_path=cache_path,
                split_dir=split_dir,
                sample_to_idx=sample_to_idx,
                ok=ok,
                per_split_limit=per_split_limit,
                device=device,
                args=args,
                epochs=epochs,
            )
            fold_metrics.append(metrics)
            all_y.extend(y_true)
            all_p.extend(y_prob)
            checkpoints.append(str(ckpt_path.relative_to(PROJECT_ROOT)))
            partial_aggregate = build_aggregate(
                mode=mode,
                args=args,
                device=device,
                epochs=epochs,
                folds=folds,
                use_synib=use_synib,
                use_amp=use_amp,
                cache_path=cache_path,
                cache_resolution=cache_resolution,
                fold_metrics=fold_metrics,
                all_y=all_y,
                all_p=all_p,
                checkpoints=checkpoints,
                threshold_rows=threshold_rows,
            )
            write_metrics_payload(out_dir, partial_aggregate, fold_metrics, status="IN_PROGRESS")
            continue

        train_idx, train_y = split_indices(split_dir / f"fold_{fold}_train.csv", sample_to_idx, ok, limit=per_split_limit)
        test_idx, test_y = split_indices(split_dir / f"fold_{fold}_test.csv", sample_to_idx, ok, limit=max(per_split_limit // 4, 8) if per_split_limit else 0)
        if len(train_idx) < 8 or len(test_idx) < 4:
            continue

        val_n = max(int(len(train_idx) * args.val_fraction), 1)
        val_n = min(val_n, len(train_idx) - 1)
        combined = list(zip(train_idx, train_y))
        random.shuffle(combined)
        train_idx, train_y = zip(*combined)
        val_idx = train_idx[:val_n]
        val_y = train_y[:val_n]
        fit_idx = train_idx[val_n:]
        fit_y = train_y[val_n:]

        tab_mean, tab_std = tabular_stats(cache_path, fit_idx)
        train_loader = DataLoader(
            maybe_wrap_train_dataset(
                H5MethodXDataset(cache_path, fit_idx, fit_y),
                enabled=args.monai_augment,
                image_positions=(0,),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            H5MethodXDataset(cache_path, val_idx, val_y),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            H5MethodXDataset(cache_path, test_idx, test_y),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = MethodX3DClassifier(fusion=args.fusion, backbone=args.backbone).to(device)
        pos_weight = torch.tensor(
            [min(max(sum(1 for v in fit_y if v == 0), 1) / max(sum(fit_y), 1), 20.0)],
            device=device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - args.warmup_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val = float("inf")
        patience = 0
        train_curve: list[float] = []
        val_curve: list[float] = []
        for epoch in range(epochs):
            scheduler.step()
            train_loss = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                pos_weight,
                latent_alignment_lambda=args.latent_alignment_lambda,
                synib_cfg=synib_cfg,
                use_synib=use_synib,
                tab_mean=tab_mean,
                tab_std=tab_std,
                use_amp=use_amp,
                scaler=scaler,
                grad_accum_steps=args.grad_accum,
                label_smoothing=args.label_smoothing,
                contrastive_alignment=args.contrastive_alignment,
                contrastive_temperature=args.contrastive_temperature,
                contrastive_loss_fn=contrastive_loss_fn,
            )
            val_loss = eval_loss(
                model,
                val_loader,
                device,
                pos_weight,
                tab_mean=tab_mean,
                tab_std=tab_std,
            )

            train_curve.append(train_loss)
            val_curve.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stopping_patience:
                    break

        val_true, val_prob = predict(model, val_loader, device, tab_mean, tab_std, use_tta=args.tta, tta_augmentations=args.tta_augmentations)
        y_true, y_prob = predict(model, test_loader, device, tab_mean, tab_std, use_tta=args.tta, tta_augmentations=args.tta_augmentations)
        threshold_eval = evaluate_with_train_selected_threshold(
            validation_y=val_true,
            validation_prob=val_prob,
            test_y=y_true,
            test_prob=y_prob,
        )
        threshold_rows.append({"fold": fold, **threshold_eval})
        metrics = {
            "fold": fold,
            "n": len(y_true),
            "test_n": len(y_true),
            "train_n": len(fit_idx),
            "val_n": len(val_idx),
            "roc_auc": roc_auc(y_true, y_prob),
            "pr_auc": average_precision(y_true, y_prob),
            "brier": brier(y_true, y_prob),
            "best_val_loss": best_val,
            "val_loss_last": val_curve[-1] if val_curve else None,
            "loss_last": train_curve[-1] if train_curve else None,
            "epochs_run": len(train_curve),
            "early_stopped": len(train_curve) < epochs,
            "train_loss_curve": train_curve,
            "val_loss_curve": val_curve,
            "synib_lambda": args.synib_lambda,
            "modality_dropout": args.modality_dropout,
            "threshold_metrics": threshold_eval,
        }
        fold_metrics.append(metrics)
        all_y.extend(y_true)
        all_p.extend(y_prob)

        ckpt_path = out_dir / f"method_x_3d_fold_{fold}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "fold": fold,
                "mode": mode,
                "fusion": args.fusion,
                "latent_alignment_lambda": args.latent_alignment_lambda,
                "synib_lambda": args.synib_lambda,
                "modality_dropout": args.modality_dropout,
                "tab_mean": tab_mean,
            "tab_std": tab_std,
            "epochs_run": len(train_curve),
            "augmentation_enabled": args.monai_augment,
        },
            ckpt_path,
        )
        checkpoints.append(str(ckpt_path.relative_to(PROJECT_ROOT)))
        partial_aggregate = build_aggregate(
            mode=mode,
            args=args,
            device=device,
            epochs=epochs,
            folds=folds,
            use_synib=use_synib,
            use_amp=use_amp,
            cache_path=cache_path,
            cache_resolution=cache_resolution,
            fold_metrics=fold_metrics,
            all_y=all_y,
            all_p=all_p,
            checkpoints=checkpoints,
            threshold_rows=threshold_rows,
        )
        write_metrics_payload(
            out_dir,
            partial_aggregate,
            fold_metrics,
            status="IN_PROGRESS" if len(checkpoints) < folds else "COMPLETED",
        )

    aggregate = build_aggregate(
        mode=mode,
        args=args,
        device=device,
        epochs=epochs,
        folds=folds,
        use_synib=use_synib,
        use_amp=use_amp,
        cache_path=cache_path,
        cache_resolution=cache_resolution,
        fold_metrics=fold_metrics,
        all_y=all_y,
        all_p=all_p,
        checkpoints=checkpoints,
        threshold_rows=threshold_rows,
    )
    aggregate["status"] = "COMPLETED"

    payload = {"aggregate": aggregate, "folds": fold_metrics}
    atomic_write_json(out_dir / "three_d_metrics.json", payload)
    md = [
        "# Method X 3D metrics\n\n",
        f"- model: `{aggregate['model']}`\n",
        f"- fusion: `{aggregate['fusion']}`\n",
        f"- backbone: `{aggregate['backbone']}`\n",
        f"- mode: `{aggregate['mode']}`\n",
        f"- synib_lambda: `{aggregate['synib_lambda']}`\n",
        f"- roc_auc: `{aggregate['roc_auc']}`\n",
        f"- final_claim_allowed: `{aggregate['final_claim_allowed']}`\n",
    ]
    atomic_write_text(out_dir / "three_d_metrics.md", "".join(md))
    write_run_manifest(
        "results/logs/train_3d_mri_manifest.json",
        stage="train-3d",
        data_root=None,
        extra={"aggregate": aggregate},
    )
    print(json.dumps({"status": "COMPLETED", "roc_auc": aggregate["roc_auc"], "mode": mode}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
