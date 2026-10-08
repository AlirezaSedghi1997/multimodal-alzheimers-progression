"""Compare Method Y image participants with the locked A4 endpoint cohort."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.method_y_revival.posttraining_correction_v4_2 import CorrectionV42Error  # noqa: E402
from src.method_y_revival.posttraining_selection_flow_v4_2 import write_selection_flow  # noqa: E402


def main() -> int:
    out = PROJECT / "results" / "method_y_revival_v4_posttraining_correction_v4_2"
    try:
        report = write_selection_flow(
            PROJECT, out / "locked_artifact_correction.json",
            out / "image_cohort_selection_flow.json"
        )
    except CorrectionV42Error as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), flush=True)
        return 2
    print(json.dumps({
        "status": report["status"],
        "branches": {
            branch: {
                "parent_n": data["endpoint_parent_n"],
                "included_n": data["image_included_n"],
                "excluded_n": data["image_excluded_n"],
                "included_events": data["included_events"],
                "excluded_events": data["excluded_events"],
            }
            for branch, data in report["branches"].items()
        },
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
