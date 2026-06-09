#!/usr/bin/env python3
"""Y9: Missing-PET inference simulation protocol and optional fusion-model dropout evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json, atomic_write_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_metrics", default="results/final_results/method_y/late_fusion_training/method_y_late_fusion_metrics.json")
    ap.add_argument("--mri_metrics", default="results/final_results/method_y/mri_paired_training/method_y_mri_paired_metrics.json")
    ap.add_argument("--out_json", default="results/tables/method_y_missing_pet_simulation.json")
    ap.add_argument("--out_md", default="results/tables/method_y_missing_pet_simulation.md")
    args = ap.parse_args()

    fusion_path = PROJECT_ROOT / args.fusion_metrics
    mri_path = PROJECT_ROOT / args.mri_metrics
    fusion = json.loads(fusion_path.read_text(encoding="utf-8")) if fusion_path.exists() else None
    mri = json.loads(mri_path.read_text(encoding="utf-8")) if mri_path.exists() else None

    status = "PENDING_FUSION_TRAINING"
    if fusion and (fusion.get("aggregate") or {}).get("status") in {"COMPLETED", "CPU_DEBUG_ONLY"}:
        status = "PROTOCOL_READY_AWAITING_DROPOUT_EVAL"

    report = {
        "stage": "method_y_missing_pet_simulation",
        "status": status,
        "protocol": {
            "train_with": "MRI+PET late/gated fusion on paired subset",
            "inference_modes": ["full_mri_pet", "pet_encoder_zeroed", "pet_encoder_dropped"],
            "primary_question": "Does fusion degrade gracefully to MRI-only when PET unavailable at inference?",
        },
        "reference_metrics": {
            "mri_paired_y2": (mri or {}).get("aggregate"),
            "late_fusion_y3": (fusion or {}).get("aggregate"),
        },
        "reporting": "Supplementary sensitivity; not a non-invasive Method X claim.",
    }
    atomic_write_json(PROJECT_ROOT / args.out_json, report)
    md = (
        "# Method Y Missing-PET Simulation (Y9)\n\n"
        f"- Status: `{status}`\n"
        "- Inference: evaluate fusion model with PET encoder dropped/zero at test time.\n"
        "- Compare to Y2 MRI-only on identical paired hashes.\n"
    )
    atomic_write_text(PROJECT_ROOT / args.out_md, md)
    print(json.dumps({"status": status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
