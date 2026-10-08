"""Reconcile locked Method Y v4 results without running models or outer tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.method_y_revival.posttraining_correction_v4_2 import (  # noqa: E402
    CorrectionV42Error,
    write_correction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT / "results" / "method_y_revival_v4_image_factorial_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT / "results" / "method_y_revival_v4_posttraining_correction_v4_2"
                 / "locked_artifact_correction.json"),
    )
    args = parser.parse_args()
    if not args.source_root.resolve().is_relative_to(PROJECT):
        raise SystemExit("SOURCE_MUST_BE_WITHIN_PROJECT")
    if not args.output.resolve().is_relative_to(PROJECT):
        raise SystemExit("OUTPUT_MUST_BE_WITHIN_PROJECT")
    try:
        report = write_correction(args.source_root, args.output)
    except CorrectionV42Error as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), flush=True)
        return 2
    print(json.dumps({
        "status": report["status"],
        "science_gate_status": report["science_gate_status"],
        "claim_status": report["claim_status"],
        "fold_selected_arms": {
            branch: [row["selected_arm"] for row in body["folds"]]
            for branch, body in report["branches"].items()
        },
        "output": str(args.output),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
