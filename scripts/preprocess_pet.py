#!/usr/bin/env python3
"""Backward-compatible wrapper for the Method Y PET spatial/QC pipeline."""
from __future__ import annotations

import re
from pathlib import Path

from src.utils.pet_manifest import tier_for_gap as _tier_for_gap
from src.utils.pet_spatial_qc import assert_method_y_output_root


METHOD_Y_COMPATIBILITY_CONTRACT = {
    "method_x_pet_input_allowed": False,
    "raw_identifiers_exported": False,
    "raw_paths_exported_public": False,
    "pet_t1_gap_used_as_model_feature": False,
}


def assert_output_root_is_method_y_only(out_root: str | Path) -> Path:
    return assert_method_y_output_root(out_root)


def tier_for_gap(gap: float | None, policy: dict) -> str:
    return _tier_for_gap(gap, policy)


def parse_pet_image_filename(path: str | Path) -> tuple[str, str] | None:
    name = Path(path).name
    patterns = [
        r"(?P<substudy>A4|LEARN|SF).*?(?P<bid>B?\d{4,})",
        r"(?P<bid>B?\d{4,}).*?(?P<substudy>A4|LEARN|SF)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if not match:
            continue
        substudy = match.group("substudy").upper()
        bid = match.group("bid").upper()
        if not bid.startswith("B"):
            bid = "B" + bid
        return substudy, bid
    return None


def main() -> int:
    from scripts.preprocess_pet_spatial import main as spatial_main

    return spatial_main()


if __name__ == "__main__":
    raise SystemExit(main())

