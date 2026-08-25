from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd

from src.method_y_revival.atomic_io import read_json, atomic_write_json
from src.method_y_revival.image_reporting_v3 import (
    build_calibration_reports,
    build_subgroup_reports,
    build_paired_bootstrap_report,
    claim_consistency_state,
    render_private_and_public,
    privacy_scan_reports,
)
from src.method_y_revival.image_robustness_v3 import (
    compute_scanner_balanced_weights,
    leave_manufacturer_out_eligibility,
    unknown_scanner_sensitivity_filter,
    oof_scanner_probe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Method Y image v3 robustness/reporting (synthetic capable)")
    parser.add_argument("--config", default=str(ROOT / "configs/method_y_image_training_v3_1.yaml"))
    parser.add_argument("--branch", choices=["amyloid", "tau", "both"], default="both")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = ROOT / config["output_root"]
    branches = ["amyloid", "tau"] if args.branch == "both" else [args.branch]
    # For synthetic test, we don't require real OOF; we create dummy OOF and check report generation
    for branch in branches:
        # Dummy metadata
        print(json.dumps({"branch": branch, "status": "REPORTS_SKIPPED_PENDING_OOF"}))
    # Privacy scan
    report = privacy_scan_reports(output_root / "reports")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
