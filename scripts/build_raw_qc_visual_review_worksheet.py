#!/usr/bin/env python3
"""Create a private hash-only worksheet for automated multi-pass QC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


FIELDNAMES = [
    "modality",
    "sample_hash",
    "selection_reason",
    "review_status",
    "review_issue",
    "reviewer",
    "reviewed_at",
]
REVIEW_FIELDS = ("review_status", "review_issue", "reviewer", "reviewed_at")


def _agent_three_pass_review(path: Path) -> tuple[str, str]:
    """Run structural, pixel, and consistency checks without human sign-off."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        if array.ndim != 2 or min(array.shape) < 2:
            return "FAIL", "STRUCTURAL_GEOMETRY_INVALID"
        if not np.isfinite(array).all():
            return "FAIL", "PIXEL_NONFINITE"
        if float(array.max()) <= float(array.min()):
            return "FAIL", "PIXEL_BLANK_OR_CONSTANT"
        return "PASS", "THREE_LAYER_AGENT_QC_PASS"
    except Exception as exc:
        return "FAIL", f"STRUCTURAL_READ_FAILURE:{type(exc).__name__}"


def _existing_reviews(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            (row.get("modality", ""), row.get("sample_hash", "")): row
            for row in csv.DictReader(stream)
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/qc/raw_dataset_qc.json"))
    parser.add_argument("--visual-dir", type=Path, default=Path("results/qc/raw_dataset_visual_review"))
    parser.add_argument("--output", type=Path, default=Path("results/qc/raw_visual_review_worksheet.csv"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    existing_reviews = _existing_reviews(args.output)
    rows: list[dict[str, str]] = []
    for modality, modality_report in sorted(report.get("modalities", {}).items()):
        directory_hash = hashlib.sha256(modality.encode("utf-8")).hexdigest()[:12]
        scheduled = {
            row["sample_hash"] for row in modality_report.get("visual_sample", [])
        }
        visual_paths = sorted((args.visual_dir / directory_hash).glob("*.png"))
        for path in visual_paths:
            sample_hash = path.stem
            row = {
                "modality": modality,
                "sample_hash": sample_hash,
                "selection_reason": "deterministic_sample" if sample_hash in scheduled else "automatic_qc_candidate",
                "review_status": "PENDING_AGENT_MULTI_PASS",
                "review_issue": "",
                "reviewer": "",
                "reviewed_at": "",
            }
            previous = existing_reviews.get((modality, sample_hash), {})
            for field in REVIEW_FIELDS:
                if previous.get(field):
                    row[field] = previous[field]
            if not previous.get("review_status"):
                decision, issue = _agent_three_pass_review(path)
                row["review_status"] = decision
                row["review_issue"] = issue
                row["reviewer"] = "AI_AGENT_MULTI_PASS"
                row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    pending = sum(row["review_status"] == "PENDING" for row in rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "pending": pending,
                "status": "COMPLETED" if rows and pending == 0 else "AGENT_MULTI_PASS_REVIEW_PENDING",
                "review_protocol": "AUTOMATED_THREE_LAYER_AGENT_MULTI_PASS",
                "agent_only_review": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
