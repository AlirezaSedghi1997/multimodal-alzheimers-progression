"""Build aggregate, fail-closed Method Y v2 branch readiness artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.method_y_revival.branch_readiness import (
    BRANCH_NAMES,
    build_branch_readiness,
    collect_project_evidence,
    write_branch_readiness,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--output-root",
        default=None,
        help="Default: <project-root>/results/method_y_revival_v2/readiness",
    )
    parser.add_argument("--branch", choices=BRANCH_NAMES, action="append")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    selected = tuple(args.branch or BRANCH_NAMES)
    evidence = collect_project_evidence(project_root, branches=selected)
    report = build_branch_readiness(
        evidence, source_role="read_only_project_aggregate_contracts"
    )
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else project_root / "results" / "method_y_revival_v2" / "readiness"
    )
    json_path, markdown_path = write_branch_readiness(report, output_root)
    print(
        json.dumps(
            {
                "status": "COMPLETED_FAIL_CLOSED",
                "readiness_json": json_path.name,
                "readiness_markdown": markdown_path.name,
                "selected_evidence_count": len(evidence),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
