"""Read-only Method Y synthesis from locked artifacts; no training or inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.method_y_revival.posttraining_science_status_v4_2 import write_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_root or root / "results" / "method_y_revival_v4_posttraining_correction_v4_2"
    report = write_status(root, output)
    print(json.dumps({
        "status": report["status"],
        "scientific_validation_status": report["scientific_validation_status"],
        "claim_status": report["claim_status"],
        "report_sha256": report["report_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
