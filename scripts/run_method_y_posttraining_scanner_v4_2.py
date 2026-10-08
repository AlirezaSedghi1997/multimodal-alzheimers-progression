"""Run fold-local scanner probes from committed Method Y v4 embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.method_y_revival.posttraining_scanner_v4_2 import (  # noqa: E402
    CorrectionV42Error,
    write_scanner_correction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=49)
    args = parser.parse_args()
    root = PROJECT / "results" / "method_y_revival_v4_image_factorial_v1"
    out_root = PROJECT / "results" / "method_y_revival_v4_posttraining_correction_v4_2"
    try:
        report = write_scanner_correction(
            root, out_root / "locked_artifact_correction.json",
            out_root / "fold_local_scanner_probes.json",
            permutations=args.permutations,
        )
    except CorrectionV42Error as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), flush=True)
        return 2
    print(json.dumps({
        "status": report["status"],
        "science_gate_status": report["science_gate_status"],
        "completed_probes": {
            branch: data["completed_probes"]
            for branch, data in report["results"].items()
        },
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
