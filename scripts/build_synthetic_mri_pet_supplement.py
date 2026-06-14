#!/usr/bin/env python3
"""Write supplementary MRI-to-PET synthetic augmentation policy artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def build_synthetic_policy(*, folds: int = 5, synthetic_per_real: int = 1) -> dict:
    return {
        "branch": "supplementary_mri_to_pet_synthetic_augmentation",
        "status": "DESIGN_READY_SUPPLEMENTARY_ONLY",
        "folds": int(folds),
        "synthetic_per_real_train_sample": int(synthetic_per_real),
        "fit_generator_scope": "train_fold_only",
        "add_synthetic_to": "same_train_fold_only",
        "validation_real_only": True,
        "test_real_only": True,
        "holdout_real_only": True,
        "method_x_input_allowed": False,
        "method_y_only": True,
        "synthetic_counted_as_real_subjects": False,
        "public_output_granularity": "aggregate_policy_only",
        "raw_or_synthetic_images_public": False,
        "generator_storage_public": False,
        "leakage_guards": [
            "Fit MRI-to-PET generator independently inside each train fold.",
            "Do not train generator on validation, test, or holdout hashes.",
            "Do not add synthetic PET to validation, test, holdout, or Method X.",
            "Report real N separately from synthetic train exposures.",
        ],
        "claim_boundary": (
            "Supplementary augmentation sensitivity only; not evidence of more real patients, "
            "not a non-invasive Method X input, and not a clinical Alzheimer diagnosis model."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--synthetic_per_real", type=int, default=1)
    ap.add_argument("--out_json", default="results/tables/synthetic_mri_pet_supplement.json")
    ap.add_argument("--out_md", default="results/tables/synthetic_mri_pet_supplement.md")
    args = ap.parse_args()

    policy = build_synthetic_policy(folds=args.folds, synthetic_per_real=args.synthetic_per_real)
    atomic_write_json(args.out_json, policy)
    lines = [
        "# Synthetic MRI-to-PET Supplementary Branch\n\n",
        f"- status: `{policy['status']}`\n",
        f"- generator fit scope: `{policy['fit_generator_scope']}`\n",
        f"- synthetic added to: `{policy['add_synthetic_to']}`\n",
        f"- validation real only: `{policy['validation_real_only']}`\n",
        f"- test real only: `{policy['test_real_only']}`\n",
        f"- Method X input allowed: `{policy['method_x_input_allowed']}`\n",
        f"- synthetic counted as real subjects: `{policy['synthetic_counted_as_real_subjects']}`\n\n",
        "## Leakage Guards\n\n",
    ]
    lines.extend(f"- {guard}\n" for guard in policy["leakage_guards"])
    lines.extend(["\n## Claim Boundary\n\n", policy["claim_boundary"], "\n"])
    atomic_write_text(args.out_md, "".join(lines))
    write_run_manifest("results/logs/synthetic_mri_pet_supplement_manifest.json", stage="synthetic_mri_pet_supplement", data_root=None, extra=policy)
    StateManager().append_event("synthetic_mri_pet_supplement", policy["status"], "Fold-local synthetic MRI-to-PET policy artifact generated.")
    print(json.dumps({"status": policy["status"], "method_x_input_allowed": policy["method_x_input_allowed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
