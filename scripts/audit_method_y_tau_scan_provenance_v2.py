#!/usr/bin/env python3
"""Audit local FTP identity metadata and source-table bridges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.tau_scan_provenance_audit import audit_tau_scan_provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/method_y_revival_v2")
    parser.add_argument("--linkage", type=Path)
    parser.add_argument(
        "--quantitative-validation",
        type=Path,
        help="Canonical aggregate scan-level Tau quantitative-validation JSON.",
    )
    parser.add_argument(
        "--crosswalk-audit",
        type=Path,
        help="Canonical aggregate Tau source-region crosswalk audit JSON.",
    )
    parser.add_argument("--suvr-source", type=Path, default=Path(r"H:\A4 Learn\Clinical\External Data\imaging_SUVR_tau.csv"))
    parser.add_argument("--stanford-source", type=Path, default=Path(r"H:\A4 Learn\Clinical\External Data\imaging_Tau_PET_Stanford.csv"))
    parser.add_argument(
        "--private-rows-output",
        type=Path,
        help="Optional row-level evidence output; must be under --private-rows-root and outside the repository.",
    )
    parser.add_argument("--private-rows-root", type=Path)
    args = parser.parse_args()
    output_root = args.output_root
    linkage = args.linkage or output_root / "manifests/private_ftp_t1_transform_eligible.parquet"
    output_json = output_root / "provenance/tau_scan_source_identity_audit_v2.json"
    output_report = output_root / "reports/TAU_QUANTITATIVE_VALIDATION_V2.md"
    quantitative_validation = args.quantitative_validation or (
        output_root / "provenance/scan_level_tau_quantitative_validation.json"
    )
    crosswalk_audit = args.crosswalk_audit or (
        output_root / "provenance/tau_source_region_crosswalk_audit_v2.json"
    )
    if args.private_rows_output is not None:
        private_root = (args.private_rows_root or args.private_rows_output.parent).resolve()
        if ROOT.resolve() in private_root.parents or private_root == ROOT.resolve():
            parser.error("--private-rows-root must be outside the repository")
    else:
        private_root = None
    report = audit_tau_scan_provenance(
        linkage,
        source_suvr_path=args.suvr_source,
        source_stanford_path=args.stanford_source,
        output_json=output_json,
        output_report=output_report,
        output_rows=args.private_rows_output,
        private_rows_root=private_root,
        quantitative_validation_path=quantitative_validation,
        crosswalk_audit_path=crosswalk_audit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
