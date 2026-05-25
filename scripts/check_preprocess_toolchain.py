from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preprocess_mri_full import check_toolchain, write_toolchain_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=os.environ.get("METHOD_X_MNI_TEMPLATE"))
    ap.add_argument(
        "--diagnostic_only",
        action="store_true",
        help="Return exit code 0 even when blocked; useful for interactive environment checks.",
    )
    args = ap.parse_args()

    template = Path(args.template) if args.template else None
    toolchain = check_toolchain(template)
    report = {
        "status": toolchain["status"],
        "toolchain": toolchain,
        "missing": toolchain["missing"],
        "execute_requested": False,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "output_resolution": 96,
    }
    write_toolchain_report(report)
    print(json.dumps(report, indent=2))
    if args.diagnostic_only:
        return 0
    return 0 if toolchain["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
