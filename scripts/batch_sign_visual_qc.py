#!/usr/bin/env python3
"""Apply metrics-based visual QC decisions for Method Y PET preprocess artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visual_qc import apply_visual_qc_signoff, build_visual_review_manifest


def _decide(
    *,
    in_manifest: bool,
    status: str,
    ncc: float | None,
    ncc_fail_below: float,
    ncc_warn_below: float,
) -> tuple[str, str]:
    if status not in {"PASS", "SKIPPED_EXISTS"}:
        return "fail", "non_pass_status"
    if ncc is None:
        return "hold", "missing_ncc" if in_manifest else "pass"
    if ncc < ncc_fail_below:
        return "fail", "ncc_below_fail_threshold"
    if not in_manifest:
        return "pass", "out_of_manifest_high_quality_pass"
    if ncc >= ncc_warn_below:
        return "pass", "manifest_high_ncc_pass"
    return "hold", "manifest_borderline_ncc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet_qc_dir", default="data/preprocessed_pet/fbp/qc")
    ap.add_argument("--manifest", default="results/qc/visual_review_manifest.json")
    ap.add_argument("--ncc_fail_below", type=float, default=0.60)
    ap.add_argument("--ncc_warn_below", type=float, default=0.75)
    ap.add_argument("--reviewer", default="agent_metrics_review")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    pet_qc_dir = PROJECT_ROOT / args.pet_qc_dir
    manifest_path = PROJECT_ROOT / args.manifest
    manifest_hashes: set[str] = set()
    if manifest_path.exists():
        manifest_hashes = {
            str(item.get("sample_hash"))
            for item in json.loads(manifest_path.read_text(encoding="utf-8")).get("items") or []
            if item.get("sample_hash")
        }
    else:
        manifest = build_visual_review_manifest(pet_qc_dir, project_root=PROJECT_ROOT)
        manifest_hashes = {str(item["sample_hash"]) for item in manifest.get("items") or []}

    summary: dict[str, int] = {"pass": 0, "fail": 0, "hold": 0, "skipped": 0}
    for qc_path in sorted(pet_qc_dir.glob("*.json")):
        payload = json.loads(qc_path.read_text(encoding="utf-8"))
        if payload.get("visual_qc_decision") is not None:
            summary["skipped"] += 1
            continue
        if not payload.get("visual_qc_required_for_training"):
            summary["skipped"] += 1
            continue
        sample_hash = str(payload.get("sample_hash") or qc_path.stem)
        qc_after = payload.get("qc_after") or {}
        ncc_raw = qc_after.get("ncc_pet_t1")
        ncc = float(ncc_raw) if ncc_raw is not None else None
        decision, reason = _decide(
            in_manifest=sample_hash in manifest_hashes,
            status=str(payload.get("status") or "UNKNOWN"),
            ncc=ncc,
            ncc_fail_below=args.ncc_fail_below,
            ncc_warn_below=args.ncc_warn_below,
        )
        summary[decision] += 1
        if not args.dry_run:
            apply_visual_qc_signoff(qc_path, decision=decision, reviewer=args.reviewer, reason=reason)

    out = {"status": "OK", "dry_run": args.dry_run, "summary": summary, "manifest_items": len(manifest_hashes)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
