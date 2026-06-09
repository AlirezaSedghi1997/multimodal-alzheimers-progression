#!/usr/bin/env python3
"""Record human visual QC sign-off on a PET preprocess QC JSON artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visual_qc import apply_visual_qc_signoff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_hash", required=True)
    ap.add_argument("--decision", required=True, choices=["pass", "fail", "hold"])
    ap.add_argument("--tracer", default="fbp")
    ap.add_argument("--pet_qc_dir", default="data/preprocessed_pet/fbp/qc")
    ap.add_argument("--reviewer", default="human_reviewer")
    args = ap.parse_args()

    qc_path = PROJECT_ROOT / args.pet_qc_dir / f"{args.sample_hash}.json"
    payload = apply_visual_qc_signoff(qc_path, decision=args.decision, reviewer=args.reviewer)
    print(json.dumps({"status": "OK", "sample_hash": args.sample_hash, "visual_qc_signoff": payload.get("visual_qc_signoff")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
