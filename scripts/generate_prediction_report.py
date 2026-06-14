#!/usr/bin/env python3
"""Generate human-interpretable prediction reports for amyloid status."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_prediction_report(
    prediction: float,
    confidence: float,
    key_regions: list[dict],
    tabular_factors: list[dict],
    patient_id: str = "anonymous",
    ad_region_report: dict | None = None,
) -> dict:
    """Generate a structured prediction report.

    Args:
        prediction: Model output probability (0-1)
        confidence: Confidence score (0-1)
        key_regions: Top contributing brain regions
        tabular_factors: Feature importance for tabular inputs
        patient_id: Patient identifier (hashed)
        ad_region_report: Optional AD region diagnosis report dict

    Returns:
        Structured report dictionary.
    """
    ab_status = "AB+" if prediction > 0.5 else "AB-"
    risk_level = "HIGH" if prediction > 0.7 else ("MODERATE" if prediction > 0.3 else "LOW")

    report = {
        "patient_id": patient_id,
        "prediction": {
            "amyloid_status": ab_status,
            "probability": round(prediction, 4),
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
        },
        "key_brain_regions": [
            {"region": r["name"], "attribution": round(r["attribution"], 4)}
            for r in key_regions[:5]
        ],
        "tabular_factors": [
            {"feature": f["name"], "importance": round(f["importance"], 4)}
            for f in tabular_factors
        ],
    }

    if ad_region_report is not None:
        report["ad_region_analysis"] = {
            "ad_pattern_score": ad_region_report.get("ad_pattern_score", 0.0),
            "spatial_pattern": ad_region_report.get("spatial_pattern", "unknown"),
            "primary_evidence_regions": ad_region_report.get("primary_evidence_regions", [])[:5],
            "core_region_involvement": ad_region_report.get("core_region_involvement", {}),
            "clinical_interpretation": ad_region_report.get("clinical_interpretation", ""),
        }
    else:
        report["ad_region_analysis"] = None

    report["disclaimer"] = (
        "This is a research prediction tool. Results should not be used "
        "for clinical diagnosis without consultation with a specialist."
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", type=float, required=True)
    ap.add_argument("--confidence", type=float, default=0.0)
    ap.add_argument("--patient_id", type=str, default="anonymous")
    ap.add_argument("--key_regions", type=str, default="[]", help="JSON list of region dicts")
    ap.add_argument("--tabular_factors", type=str, default="[]", help="JSON list of factor dicts")
    ap.add_argument("--ad_region_report", type=str, default=None, help="JSON AD region report path or inline JSON")
    ap.add_argument("--out_dir", default="results/reports")
    args = ap.parse_args()

    key_regions = json.loads(args.key_regions)
    tabular_factors = json.loads(args.tabular_factors)

    ad_region_report = None
    if args.ad_region_report:
        if Path(args.ad_region_report).exists():
            ad_region_report = json.loads(Path(args.ad_region_report).read_text(encoding="utf-8"))
        else:
            ad_region_report = json.loads(args.ad_region_report)

    report = generate_prediction_report(
        prediction=args.prediction,
        confidence=args.confidence,
        key_regions=key_regions,
        tabular_factors=tabular_factors,
        patient_id=args.patient_id,
        ad_region_report=ad_region_report,
    )

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_{args.patient_id}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
