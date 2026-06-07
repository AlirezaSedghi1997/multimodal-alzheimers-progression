#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.pet_spatial_qc import atomic_write_json, load_yaml, sha256_file


def compact(path: Path) -> dict:
    obj = load_yaml(path)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "policy_name": obj.get("policy_name"),
        "version": obj.get("version"),
        "method": obj.get("method"),
        "key_decisions": {
            "method_x_pet_input_allowed": obj.get("method_x_boundary", {}).get("method_x_pet_image_input_allowed"),
            "spatial_required": obj.get("spatial_preprocessing", {}).get("required_for_training_ready_outputs"),
            "default_normalization": obj.get("normalization", {}).get("default_mode"),
            "scalar_suvr_as_method_x_input": obj.get("global_rules", {}).get("scalar_suvr_as_method_x_input"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tables/method_y_compact_policy_index.json")
    ap.add_argument("policies", nargs="*", default=[
        "configs/model/method_y_spatial_preprocess_policy.yaml",
        "configs/model/method_y_suvr_reference_policy.yaml",
    ])
    args = ap.parse_args()
    out = [compact(Path(p)) for p in args.policies if Path(p).exists()]
    atomic_write_json(args.out, {"policies": out})
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
