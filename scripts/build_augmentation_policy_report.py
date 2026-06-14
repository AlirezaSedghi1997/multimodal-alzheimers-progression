#!/usr/bin/env python3
"""Write aggregate train-only augmentation policy report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.volume_augmentation import build_augmentation_policy
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_train_n", type=int, default=1500)
    ap.add_argument("--exposures_per_real", type=int, default=4)
    ap.add_argument("--out_json", default="results/tables/augmentation_policy.json")
    ap.add_argument("--out_md", default="results/tables/augmentation_policy.md")
    args = ap.parse_args()

    report = build_augmentation_policy(
        enabled=True,
        real_train_n=args.real_train_n,
        exposures_per_real=args.exposures_per_real,
    )
    report.update(
        {
            "stage": "train_only_monai_augmentation_policy",
            "status": "COMPLETED",
            "modalities": ["T1 MRI", "Method Y PET"],
            "method_x_pet_input_allowed": False,
            "validation_test_holdout_augmented": False,
        }
    )
    atomic_write_json(args.out_json, report)
    atomic_write_text(
        args.out_md,
        "# Train-only MONAI Augmentation Policy\n\n"
        f"- real_train_n: `{report['real_train_n']}`\n"
        f"- exposures_per_real: `{report['exposures_per_real']}`\n"
        f"- augmented_training_exposures: `{report['augmented_training_exposures']}`\n"
        "- validation/test/holdout augmented: `False`\n"
        "- synthetic or augmented counted as real subjects: `False`\n",
    )
    write_run_manifest("results/logs/augmentation_policy_manifest.json", stage="augmentation_policy", data_root=None, extra=report)
    StateManager().append_event("augmentation_policy", "COMPLETED", "Train-only MONAI augmentation policy report generated.")
    print(json.dumps({"status": report["status"], "augmented_training_exposures": report["augmented_training_exposures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
