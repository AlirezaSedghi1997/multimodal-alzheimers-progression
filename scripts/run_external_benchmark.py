#!/usr/bin/env python3
"""External dataset benchmark: evaluate model on public AD datasets."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.external_dataset_adapter import get_adapter
from src.utils.io_helpers import atomic_write_json, atomic_write_text


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def evaluate_on_external_cache(cache_path: Path, checkpoint_path: Path, device) -> dict:
    import h5py
    import torch
    import torch.nn.functional as F

    from scripts.run_xai_summary import load_model, normalize_tabular

    model, checkpoint, tab_mean, tab_std = load_model(checkpoint_path, device)

    y_true = []
    y_prob = []

    with h5py.File(cache_path, "r") as h5:
        ok = {i for i, v in enumerate(h5["status"][:]) if decode(v) == "OK"}
        n_total = len(h5["label"][:])

        if tab_mean is None or tab_std is None:
            all_indices = [i for i in range(n_total) if i in ok]
            if all_indices:
                vals = torch.stack([torch.tensor(h5["tabular"][i], dtype=torch.float32) for i in all_indices[:200]])
                tab_mean = vals.mean(dim=0).to(device)
                tab_std = vals.std(dim=0).clamp_min(1.0).to(device)

        for idx in range(n_total):
            if idx not in ok:
                continue
            image = torch.tensor(h5["image"][idx], dtype=torch.float32, device=device).unsqueeze(0)
            raw_tabular = torch.tensor(h5["tabular"][idx], dtype=torch.float32, device=device).unsqueeze(0)
            tabular = normalize_tabular(raw_tabular, tab_mean, tab_std)
            label = int(h5["label"][idx])

            with torch.no_grad():
                logit = model(image, tabular)[0]
                prob = float(torch.sigmoid(logit).detach().cpu())

            y_true.append(label)
            y_prob.append(prob)

    return compute_metrics(y_true, y_prob)


def compute_metrics(y_true: list[int], y_prob: list[float]) -> dict:
    from scripts.train_baseline import average_precision, brier, roc_auc

    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos

    metrics = {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": round(n_pos / max(n, 1), 4),
    }

    if n_pos > 0 and n_neg > 0:
        metrics["roc_auc"] = round(roc_auc(y_true, y_prob), 4)
        metrics["pr_auc"] = round(average_precision(y_true, y_prob), 4)
        metrics["brier"] = round(brier(y_true, y_prob), 4)

        threshold = 0.5
        y_pred = [1 if p >= threshold else 0 for p in y_prob]
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        metrics["accuracy"] = round((tp + tn) / max(n, 1), 4)
        metrics["sensitivity"] = round(tp / max(tp + fn, 1), 4)
        metrics["specificity"] = round(tn / max(tn + fp, 1), 4)
        metrics["f1"] = round(2 * tp / max(2 * tp + fp + fn, 1), 4)
    else:
        metrics["roc_auc"] = None
        metrics["note"] = "Cannot compute AUC: single class in data"

    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="External Dataset Benchmark")
    ap.add_argument("--dataset", required=True, choices=["adni", "oasis"], help="Dataset name")
    ap.add_argument("--metadata_csv", required=True, help="Path to dataset metadata CSV")
    ap.add_argument("--checkpoint", default="results/final_results/three_d/method_x_3d_fold_0.pt")
    ap.add_argument("--out_dir", default="results/external_benchmark")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of subjects (0=all)")
    args = ap.parse_args()

    metadata_path = Path(args.metadata_csv)
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    out_dir = PROJECT_ROOT / args.out_dir

    if not metadata_path.exists():
        print(json.dumps({"status": "BLOCKED", "blocker": f"METADATA_MISSING: {metadata_path}"}))
        return 1
    if not checkpoint_path.exists():
        print(json.dumps({"status": "BLOCKED", "blocker": f"CHECKPOINT_MISSING: {checkpoint_path}"}))
        return 1

    try:
        import torch
    except Exception:
        print(json.dumps({"status": "BLOCKED", "blocker": "PYTORCH_MISSING"}))
        return 1

    adapter = get_adapter(args.dataset)
    import pandas as pd

    metadata = pd.read_csv(metadata_path)
    if args.limit > 0:
        metadata = metadata.head(args.limit)

    labels = adapter.get_labels(metadata)
    tabular = adapter.get_tabular_features(metadata)

    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": args.dataset,
        "n_subjects": len(labels),
        "n_positive": sum(1 for v in labels.values() if v == 1),
        "n_negative": sum(1 for v in labels.values() if v == 0),
        "status": "METADATA_ONLY",
        "note": "Full evaluation requires preprocessed MRI volumes in HDF5 cache format.",
        "available_dapters": ["adni", "oasis"],
        "cache_format": {
            "keys": ["image", "label", "sample_hash", "tabular", "status"],
            "image_shape": "(N, 1, 96, 96, 96)",
            "tabular_columns": ["age", "education_years", "apoe4_carrier", "mmse_base", "sex_binary"],
        },
    }

    cache_path = out_dir / f"{args.dataset}_cache.h5"
    if cache_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = evaluate_on_external_cache(cache_path, checkpoint_path, device)
        summary["evaluation_metrics"] = metrics
        summary["status"] = "EVALUATED"
        print(json.dumps(metrics, indent=2))

    atomic_write_json(out_dir / f"{args.dataset}_benchmark.json", summary)
    atomic_write_text(out_dir / f"{args.dataset}_benchmark.md", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
