#!/usr/bin/env python3
"""End-to-end AD region diagnosis: MRI volume -> model -> Grad-CAM -> AAL regions -> report."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ad_region_report import generate_ad_region_report, format_region_report_text
from src.evaluation.xai_regions import AAL_REGION_NAMES, compute_region_attribution
from src.utils.io_helpers import atomic_write_json, atomic_write_text


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def normalize_tabular(raw_tabular, tab_mean, tab_std):
    if tab_mean is None or tab_std is None:
        return raw_tabular
    return (raw_tabular - tab_mean) / tab_std


def find_last_conv3d(model):
    import torch.nn as nn

    last = None
    for module in model.encoder.modules():
        if isinstance(module, nn.Conv3d):
            last = module
    if last is None:
        raise RuntimeError("No Conv3d layer found for Grad-CAM.")
    return last


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


def upscale_cam_to_input(cam, target_shape):
    import torch
    import torch.nn.functional as F

    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=target_shape, mode="trilinear", align_corners=False)
    return cam[0, 0]


def create_synthetic_aal_atlas(shape):
    import numpy as np

    atlas = np.zeros(shape, dtype=np.int32)
    d, h, w = shape
    for region_id in range(1, 117):
        z = int((region_id - 1) / 4) % d
        y = int((region_id - 1) / 2) % h
        x = (region_id - 1) % w
        r = max(1, min(d, h, w) // 20)
        z_min, z_max = max(0, z - r), min(d, z + r)
        y_min, y_max = max(0, y - r), min(h, y + r)
        x_min, x_max = max(0, x - r), min(w, x + r)
        atlas[z_min:z_max, y_min:y_max, x_min:x_max] = region_id
    return atlas


def main() -> int:
    ap = argparse.ArgumentParser(description="AD Region Diagnosis Report")
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--checkpoint", default="results/final_results/three_d/method_x_3d_fold_0.pt")
    ap.add_argument("--split", default="splits/fold_0_test.csv")
    ap.add_argument("--out_dir", default="results/region_diagnosis")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--atlas", default=None, help="Path to AAL atlas .nii.gz (optional)")
    args = ap.parse_args()

    cache_path = PROJECT_ROOT / args.cache
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    split_path = PROJECT_ROOT / args.split
    out_dir = PROJECT_ROOT / args.out_dir

    if not cache_path.exists():
        print(json.dumps({"status": "BLOCKED", "blocker": "CACHE_MISSING"}))
        return 1
    if not checkpoint_path.exists():
        print(json.dumps({"status": "BLOCKED", "blocker": "CHECKPOINT_MISSING"}))
        return 1

    try:
        import h5py
        import torch
    except Exception:
        print(json.dumps({"status": "BLOCKED", "blocker": "PYTORCH_MISSING"}))
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from scripts.run_xai_summary import load_model, normalize_tabular as norm_tab

    model, checkpoint, tab_mean, tab_std = load_model(checkpoint_path, device)
    target_layer = find_last_conv3d(model)

    if args.atlas and Path(args.atlas).exists():
        import nibabel as nib

        atlas_img = nib.load(args.atlas)
        atlas_data = atlas_img.get_fdata().astype(np.int32)
    else:
        import numpy as np

        atlas_data = create_synthetic_aal_atlas((96, 96, 96))

    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    with h5py.File(cache_path, "r") as h5:
        sample_to_idx = {decode(v): i for i, v in enumerate(h5["sample_hash"][:])}
        ok = {i for i, v in enumerate(h5["status"][:]) if decode(v) == "OK"}

        if split_path.exists():
            rows = read_rows(split_path)
            indices = [sample_to_idx[r["sample_hash"]] for r in rows if r.get("sample_hash", "") in sample_to_idx]
        else:
            indices = list(range(len(sample_to_idx)))

        indices = [i for i in indices if i in ok][: args.limit]

        if tab_mean is None or tab_std is None:
            train_path = PROJECT_ROOT / "splits" / "fold_0_train.csv"
            if train_path.exists():
                train_rows = read_rows(train_path)
                train_indices = [sample_to_idx[r["sample_hash"]] for r in train_rows if r.get("sample_hash", "") in sample_to_idx]
                train_indices = [i for i in train_indices if i in ok]
                if train_indices:
                    vals = torch.stack([torch.tensor(h5["tabular"][i], dtype=torch.float32) for i in train_indices])
                    tab_mean = vals.mean(dim=0).to(device)
                    tab_std = vals.std(dim=0).clamp_min(1.0).to(device)

        for idx in indices:
            image = torch.tensor(h5["image"][idx], dtype=torch.float32, device=device).unsqueeze(0)
            raw_tabular = torch.tensor(h5["tabular"][idx], dtype=torch.float32, device=device).unsqueeze(0)
            tabular = norm_tab(raw_tabular, tab_mean, tab_std)

            cam = grad_cam_for_sample(model, target_layer, image, tabular)

            cam_96 = upscale_cam_to_input(cam, (96, 96, 96))
            cam_np = cam_96.detach().cpu().numpy()

            with torch.no_grad():
                prediction = float(torch.sigmoid(model(image, tabular)[0]).detach().cpu())

            sample_hash = decode(h5["sample_hash"][idx])
            report = generate_ad_region_report(
                grad_cam=cam_np,
                atlas_data=atlas_data,
                prediction=prediction,
                patient_id=sample_hash,
            )

            report_dict = {
                "patient_id": sample_hash,
                "prediction": report.prediction,
                "amyloid_status": report.amyloid_status,
                "ad_pattern_score": report.ad_pattern_score,
                "spatial_pattern": report.spatial_pattern,
                "primary_evidence_regions": report.primary_evidence_regions[:5],
                "core_region_involvement": report.core_region_involvement,
                "clinical_interpretation": report.clinical_interpretation,
            }
            reports.append(report_dict)

            text_report = format_region_report_text(report)
            atomic_write_text(out_dir / f"region_diagnosis_{sample_hash}.txt", text_report)

    summary = {
        "status": "COMPLETED",
        "n_samples": len(reports),
        "mean_ad_pattern_score": round(sum(r["ad_pattern_score"] for r in reports) / max(len(reports), 1), 4),
        "ab_positive_count": sum(1 for r in reports if r["amyloid_status"] == "AB+"),
        "ab_negative_count": sum(1 for r in reports if r["amyloid_status"] == "AB-"),
        "spatial_patterns": {},
    }
    for r in reports:
        p = r["spatial_pattern"]
        summary["spatial_patterns"][p] = summary["spatial_patterns"].get(p, 0) + 1

    atomic_write_json(out_dir / "region_diagnosis_summary.json", {"aggregate": summary, "reports": reports})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
