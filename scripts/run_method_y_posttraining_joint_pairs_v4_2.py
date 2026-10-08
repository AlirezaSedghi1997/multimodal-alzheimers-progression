"""Statistical reanalysis of locked Method Y v4 contrasts; no model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.method_y_revival.posttraining_joint_pairs_v4_2 import write_joint_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    project = args.project_root.resolve()
    root = project / "results" / "method_y_revival_v4_image_factorial_v1"
    base = project / "results" / "method_y_revival_v4_posttraining_correction_v4_2"
    report = write_joint_report(
        root,
        base / "locked_artifact_correction.json",
        base / "branchwise_matched_pair_points.json",
        base / "branchwise_joint_fold_bootstrap.json",
        iterations=args.iterations, seed=args.seed,
    )
    print(json.dumps({
        "status": report["status"],
        "iterations_per_branch": report["iterations_per_branch"],
        "report_sha256": report["report_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
