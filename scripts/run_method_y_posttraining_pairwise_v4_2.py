"""Recompute descriptive Method Y v4 pairs from committed artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.method_y_revival.posttraining_pairwise_descriptive_v4_2 import write_pairwise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project_root.resolve()
    root = project / "results" / "method_y_revival_v4_image_factorial_v1"
    output = project / "results" / "method_y_revival_v4_posttraining_correction_v4_2"
    report = write_pairwise(
        root, output / "locked_artifact_correction.json",
        output / "branchwise_matched_pair_points.json",
    )
    print(json.dumps({"status": report["status"],
                      "planned_pair_count_per_branch": report["planned_pair_count_per_branch"],
                      "report_sha256": report["report_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
