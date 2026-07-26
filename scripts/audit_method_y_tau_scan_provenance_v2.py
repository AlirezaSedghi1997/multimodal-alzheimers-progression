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
    parser.add_argument("--suvr-source", type=Path, default=Path(r"H:\A4 Learn\Clinical\External Data\imaging_SUVR_tau.csv"))
    parser.add_argument("--stanford-source", type=Path, default=Path(r"H:\A4 Learn\Clinical\External Data\imaging_Tau_PET_Stanford.csv"))
    args = parser.parse_args()
    output_root = args.output_root
    linkage = args.linkage or output_root / "manifests/private_ftp_t1_transform_eligible.parquet"
    output_json = output_root / "provenance/tau_scan_source_identity_audit_v2.json"
    output_rows = output_root / "provenance/tau_scan_source_identity_audit_v2.parquet"
    report = audit_tau_scan_provenance(
        linkage,
        source_suvr_path=args.suvr_source,
        source_stanford_path=args.stanford_source,
        output_json=output_json,
        output_rows=output_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
