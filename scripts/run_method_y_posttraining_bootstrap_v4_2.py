"""Bootstrap the locked Method Y v4 fold-local policy without model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.method_y_revival.posttraining_bootstrap_v4_2 import (  # noqa: E402
    CorrectionV42Error,
    write_bootstrap,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    root = PROJECT / "results" / "method_y_revival_v4_image_factorial_v1"
    out_root = PROJECT / "results" / "method_y_revival_v4_posttraining_correction_v4_2"
    try:
        report = write_bootstrap(
            root, out_root / "locked_artifact_correction.json",
            out_root / "fold_stratified_paired_bootstrap.json",
            iterations=args.iterations, seed=args.seed,
        )
    except CorrectionV42Error as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), flush=True)
        return 2
    print(json.dumps({
        "status": report["status"],
        "claim_status": report["claim_status"],
        "branches": {b: {"delta_ibs": v["point_delta_ibs"],
                          "delta_uno": v["point_delta_uno"],
                          "valid_resamples": v["valid_resamples"]}
                     for b, v in report["branches"].items()},
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
