from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.method_y_revival.pet_augmentation_qc_v4 import (  # noqa: E402
    build_locked_foreground_mask,
    calibrate_geometric_envelope,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build outcome-blind Method Y PET augmentation QC contract."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sample-count-per-branch", type=int, default=30)
    parser.add_argument("--draws-per-subject", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output_root = root / "results" / "method_y_revival_v4_image_factorial_v1"
    mask_path = (
        root
        / "data"
        / "templates"
        / "method_y_mni152nlin2009casym_v1"
        / "pet_augmentation_foreground_mask_v4.nii.gz"
    )
    mask_record = build_locked_foreground_mask(
        reference_template=(
            root
            / "data"
            / "templates"
            / "method_y_mni152nlin2009casym_v1"
            / "reference_128_1p5mm.nii.gz"
        ),
        output_mask=mask_path,
    )
    report = calibrate_geometric_envelope(
        branch_caches={
            "amyloid": root
            / "caches"
            / "method_y"
            / "revival_v3_image"
            / "amyloid_pattern_128_v1.h5",
            "tau": root
            / "caches"
            / "method_y"
            / "revival_v3_image"
            / "tau_pattern_128_v1.h5",
        },
        foreground_mask=mask_path,
        output_json=output_root / "provenance" / "pet_augmentation_qc_contract.json",
        sample_count_per_branch=int(args.sample_count_per_branch),
        draws_per_subject=int(args.draws_per_subject),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "mask": mask_record,
                "selected_envelope": report["selected_envelope"],
                "contract_sha256": report["contract_sha256"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
