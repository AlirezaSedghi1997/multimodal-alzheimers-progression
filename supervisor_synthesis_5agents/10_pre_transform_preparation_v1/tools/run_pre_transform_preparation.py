from __future__ import annotations

import argparse
import json
from pathlib import Path

from preflight import run_preparation, _safe_json
from post_transform_gate import evaluate_post_transform_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=r"H:\New project")
    parser.add_argument(
        "--output-root",
        default=r"H:\New project\supervisor_synthesis_5agents\10_pre_transform_preparation_v1",
    )
    args = parser.parse_args()
    result = run_preparation(Path(args.project_root), Path(args.output_root))
    transform = result["transform_tfinal"]
    live_status, _ = _safe_json(
        Path(args.project_root) / "results" / "method_y_revival_v2" / "reports" / "LIVE_STATUS_V2.json"
    )
    gate = evaluate_post_transform_gate(transform, live_status)
    (Path(args.output_root) / "03_audits").mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "03_audits" / "post_transform_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "state": "PRE_TRANSFORM_PREPARATION_VERIFIED",
                "active_writer": result["transform_t0"]["active_writer"],
                "row_count_t0": result["transform_t0"]["row_count"],
                "status_counts_t0": result["transform_t0"]["status_counts"],
                "row_count_tfinal": result["transform_tfinal"]["row_count"],
                "status_counts_tfinal": result["transform_tfinal"]["status_counts"],
                "training_allowed_now": result["completion_policy"]["training_allowed_now"],
                "outer_test_allowed_now": result["completion_policy"]["outer_test_allowed_now"],
                "claim_allowed_now": result["completion_policy"]["claim_allowed_now"],
                "post_transform_gate": gate["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
