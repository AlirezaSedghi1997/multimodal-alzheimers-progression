from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


TABULAR_CACHE_FEATURES = ["age", "education_years", "apoe4_carrier", "mmse_base", "sex"]
COARSE_ROIS = {
    "left": lambda z, y, x: x < 0.5,
    "right": lambda z, y, x: x >= 0.5,
    "posterior": lambda z, y, x: y < 0.5,
    "anterior": lambda z, y, x: y >= 0.5,
    "inferior": lambda z, y, x: z < 0.5,
    "superior": lambda z, y, x: z >= 0.5,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def write_blocked(out_dir: Path, reason: str) -> None:
    summary = {
        "schema_version": 1,
        "status": "BLOCKED",
        "reason": reason,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "contains_images": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "xai_summary.json", summary)
    atomic_write_text(out_dir / "xai_summary.md", f"# XAI Summary\n\n- status: `BLOCKED`\n- reason: `{reason}`\n")


def choose_indices(h5, split_path: Path, limit: int) -> list[int]:
    sample_to_idx = {decode(v): i for i, v in enumerate(h5["sample_hash"][:])}
    ok = {i for i, v in enumerate(h5["status"][:]) if decode(v) == "OK"}
    if split_path.exists():
        ordered = [sample_to_idx[row["sample_hash"]] for row in read_rows(split_path) if row["sample_hash"] in sample_to_idx]
    else:
        ordered = list(range(len(sample_to_idx)))
    selected = [idx for idx in ordered if idx in ok]
    return selected[:limit] if limit > 0 else selected


def split_indices(h5, split_path: Path) -> list[int]:
    sample_to_idx = {decode(v): i for i, v in enumerate(h5["sample_hash"][:])}
    if not split_path.exists():
        return []
    return [sample_to_idx[row["sample_hash"]] for row in read_rows(split_path) if row["sample_hash"] in sample_to_idx]


def read_tabular_rows(h5, indices: list[int], device):
    import torch

    if not indices:
        return torch.empty((0, len(TABULAR_CACHE_FEATURES)), dtype=torch.float32, device=device)
    rows = [torch.tensor(h5["tabular"][idx], dtype=torch.float32, device=device) for idx in indices]
    return torch.stack(rows, dim=0)


def tabular_stats_from_split(h5, split_path: Path, device):
    indices = split_indices(h5, split_path)
    if not indices:
        return None, None
    vals = read_tabular_rows(h5, indices, device)
    return vals.mean(dim=0), vals.std(dim=0).clamp_min(1.0)


def h5_attr(h5, key: str, default=None):
    value = h5.attrs.get(key, default)
    if hasattr(value, "item"):
        return value.item()
    return value


def load_model(checkpoint_path: Path, device):
    import torch
    from src.models.backbones.lightweight_3d import MethodX3DClassifier

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    fusion = checkpoint.get("fusion", "cross_attention")
    model = MethodX3DClassifier(fusion=fusion).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    tab_mean = checkpoint.get("tab_mean")
    tab_std = checkpoint.get("tab_std")
    if tab_mean is not None and tab_std is not None:
        tab_mean = tab_mean.to(device)
        tab_std = tab_std.to(device).clamp_min(1.0)
    return model, checkpoint, tab_mean, tab_std


def find_last_conv3d(model):
    import torch.nn as nn

    last = None
    for module in model.encoder.modules():
        if isinstance(module, nn.Conv3d):
            last = module
    if last is None:
        raise RuntimeError("No Conv3d layer found for Grad-CAM.")
    return last


def normalize_tabular(raw_tabular, tab_mean, tab_std):
    if tab_mean is None or tab_std is None:
        return raw_tabular
    return (raw_tabular - tab_mean) / tab_std


def roi_scores(cam) -> dict[str, float]:
    import torch

    d, h, w = cam.shape
    z = torch.linspace(0, 1, d, device=cam.device).view(d, 1, 1)
    y = torch.linspace(0, 1, h, device=cam.device).view(1, h, 1)
    x = torch.linspace(0, 1, w, device=cam.device).view(1, 1, w)
    out = {}
    for name, fn in COARSE_ROIS.items():
        mask = fn(z, y, x).expand_as(cam)
        out[name] = float(cam[mask].mean().detach().cpu())
    return out


def grad_cam_for_sample(model, target_layer, image, tabular):
    import torch
    import torch.nn.functional as F

    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(image, tabular)[0]
        logit.backward()
        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1))[0]
        denom = cam.max().clamp_min(1e-8)
        return cam / denom
    finally:
        forward_handle.remove()
        backward_handle.remove()


def predict_logit(model, image, raw_tabular, tab_mean, tab_std) -> float:
    import torch

    with torch.no_grad():
        tabular = normalize_tabular(raw_tabular, tab_mean, tab_std)
        return float(model(image, tabular)[0].detach().cpu())


def exact_tabular_shapley(model, image, raw_tabular, baseline, tab_mean, tab_std) -> list[float]:
    m = raw_tabular.shape[-1]
    values: dict[tuple[int, ...], float] = {}
    feature_ids = tuple(range(m))
    for size in range(m + 1):
        for subset in itertools.combinations(feature_ids, size):
            candidate = baseline.clone()
            if subset:
                ids = list(subset)
                candidate[:, ids] = raw_tabular[:, ids]
            values[subset] = predict_logit(model, image, candidate, tab_mean, tab_std)
    phis = []
    for feature in feature_ids:
        phi = 0.0
        others = [idx for idx in feature_ids if idx != feature]
        for size in range(m):
            for subset in itertools.combinations(others, size):
                with_feature = tuple(sorted((*subset, feature)))
                weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
                phi += weight * (values[with_feature] - values[tuple(subset)])
        phis.append(phi)
    return phis


def mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--checkpoint", default="results/final_results/three_d_ablation/cross_attention/method_x_3d_fold_0.pt")
    ap.add_argument("--fallback_checkpoint", default="results/final_results/three_d/method_x_3d_fold_0.pt")
    ap.add_argument("--split", default="splits/fold_0_test.csv")
    ap.add_argument("--train_split", default="splits/fold_0_train.csv")
    ap.add_argument("--out_dir", default="results/final_results/xai")
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = Path(args.fallback_checkpoint)
    if not cache_path.exists():
        write_blocked(out_dir, f"missing_cache:{cache_path}")
        return 2
    if not checkpoint_path.exists():
        write_blocked(out_dir, f"missing_checkpoint:{checkpoint_path}")
        return 2

    try:
        import h5py
        import torch
    except Exception as exc:
        raise RuntimeError("run_xai_summary requires h5py and torch") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, tab_mean, tab_std = load_model(checkpoint_path, device)
    target_layer = find_last_conv3d(model)

    gradcam_rows: list[dict[str, float]] = []
    shap_rows: list[list[float]] = []
    with h5py.File(cache_path, "r") as h5:
        cache_resolution = h5_attr(h5, "resolution")
        cache_source_mode = h5_attr(h5, "source_mode", "unknown")
        if tab_mean is None or tab_std is None:
            tab_mean, tab_std = tabular_stats_from_split(h5, Path(args.train_split), device)
        indices = choose_indices(h5, Path(args.split), args.limit)
        if not indices:
            write_blocked(out_dir, "no_ok_samples_in_cache_for_xai")
            return 2
        tabular_all = read_tabular_rows(h5, indices, device)
        baseline = tabular_all.mean(dim=0, keepdim=True)
        for idx in indices:
            image = torch.tensor(h5["image"][idx], dtype=torch.float32, device=device).unsqueeze(0)
            raw_tabular = torch.tensor(h5["tabular"][idx], dtype=torch.float32, device=device).unsqueeze(0)
            tabular = normalize_tabular(raw_tabular, tab_mean, tab_std)
            cam = grad_cam_for_sample(model, target_layer, image, tabular)
            gradcam_rows.append(roi_scores(cam))
            shap_rows.append(exact_tabular_shapley(model, image, raw_tabular, baseline, tab_mean, tab_std))

    shap_mean_abs = {
        name: float(sum(abs(row[i]) for row in shap_rows) / len(shap_rows))
        for i, name in enumerate(TABULAR_CACHE_FEATURES)
    }
    shap_mean_signed = {
        name: float(sum(row[i] for row in shap_rows) / len(shap_rows))
        for i, name in enumerate(TABULAR_CACHE_FEATURES)
    }
    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "n_samples": len(shap_rows),
        "device": device.type,
        "cache": str(cache_path),
        "cache_resolution": cache_resolution,
        "cache_source_mode": cache_source_mode,
        "interpretation_scope": (
            "final_96_validation_model" if cache_resolution == 96 else "current_3d_model_pending_final_96_refresh"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_fusion": checkpoint.get("fusion", "cross_attention"),
        "grad_cam": {
            "modality": "MRI",
            "target": "last_encoder_conv3d",
            "public_granularity": "aggregate_coarse_roi_level",
            "mean_roi_importance": mean_dicts(gradcam_rows),
        },
        "shap": {
            "modality": "tabular",
            "method": "exact_shapley_5_features",
            "feature_order": TABULAR_CACHE_FEATURES,
            "mean_abs_importance": shap_mean_abs,
            "mean_signed_contribution": shap_mean_signed,
        },
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "contains_images": False,
        "contains_per_subject_outputs": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "xai_summary.json", summary)
    md = [
        "# XAI Summary\n\n",
        f"- status: `{summary['status']}`\n",
        f"- n_samples: `{summary['n_samples']}`\n",
        f"- cache_resolution: `{summary['cache_resolution']}`\n",
        f"- interpretation_scope: `{summary['interpretation_scope']}`\n",
        "- Grad-CAM modality: `MRI`\n",
        "- Grad-CAM public granularity: `aggregate_coarse_roi_level`\n",
        f"- Grad-CAM mean ROI importance: `{summary['grad_cam']['mean_roi_importance']}`\n",
        "- SHAP modality: `tabular`\n",
        "- SHAP method: `exact_shapley_5_features`\n",
        f"- SHAP mean abs importance: `{summary['shap']['mean_abs_importance']}`\n",
        "- contains_pet_input: `False`\n",
        "- contains_raw_identifiers: `False`\n",
        "- contains_images: `False`\n",
        "- contains_per_subject_outputs: `False`\n",
    ]
    atomic_write_text(out_dir / "xai_summary.md", "".join(md))
    write_run_manifest("results/logs/xai_manifest.json", stage="xai", data_root=None, extra={"summary": summary})
    StateManager().append_event("xai", "COMPLETED", "Aggregate MRI Grad-CAM and tabular SHAP XAI summary generated.")
    print(json.dumps({"xai": "COMPLETED", "n_samples": len(shap_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
