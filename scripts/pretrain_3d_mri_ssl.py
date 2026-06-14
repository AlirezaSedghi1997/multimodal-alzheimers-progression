#!/usr/bin/env python3
"""Self-supervised 3D MAE pretraining on MRI volumes (no labels needed)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_3d_mri import H5MethodXDataset, cache_index, decode, ok_indices, read_rows
from src.utils.io_helpers import atomic_write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--out_dir", default="results/pretrained/mae_3d")
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--encoder_depth", type=int, default=6)
    ap.add_argument("--encoder_heads", type=int, default=8)
    ap.add_argument("--decoder_embed_dim", type=int, default=128)
    ap.add_argument("--decoder_depth", type=int, default=4)
    ap.add_argument("--decoder_heads", type=int, default=4)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    cache_path = PROJECT_ROOT / args.cache
    if not cache_path.exists():
        print(json.dumps({"status": "BLOCKED", "blocker": "CACHE_MISSING"}))
        return 1

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception:
        print(json.dumps({"status": "BLOCKED", "blocker": "PYTORCH_MISSING"}))
        return 1

    from scripts.train_3d_mri import resolve_training_device

    device = resolve_training_device("auto")
    use_amp = args.amp and device.type == "cuda"

    sample_to_idx = cache_index(cache_path)
    ok = ok_indices(cache_path)
    all_indices = []
    all_labels = []
    for sample_hash, idx in sample_to_idx.items():
        if idx in ok:
            all_indices.append(idx)
            all_labels.append(0)
            if args.limit > 0 and len(all_indices) >= args.limit:
                break

    dataset = H5MethodXDataset(cache_path, all_indices, all_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    from src.models.ssl.mae_3d import MAE3D

    model = MAE3D(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        mask_ratio=args.mask_ratio,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    losses = []
    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_factor = (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warmup_factor
        elif epoch == args.warmup_epochs:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
            )
            scheduler.step()
        else:
            scheduler.step()

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for image, _, _ in loader:
            image = image.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, _, _ = model(image)
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(json.dumps({"epoch": epoch, "loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]}))

        if (epoch + 1) % 10 == 0:
            ckpt_path = out_dir / f"mae_3d_epoch_{epoch + 1}.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "loss": avg_loss}, ckpt_path)

    final_ckpt = out_dir / "mae_3d_final.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs, "loss": losses[-1]}, final_ckpt)

    report = {
        "status": "COMPLETED",
        "epochs": args.epochs,
        "final_loss": losses[-1],
        "checkpoint": str(final_ckpt.relative_to(PROJECT_ROOT)),
    }
    atomic_write_json(out_dir / "mae_pretraining_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
