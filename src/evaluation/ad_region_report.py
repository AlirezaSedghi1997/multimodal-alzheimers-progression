from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from src.evaluation.ad_region_kb import (
    ADRegionInfo,
    compute_ad_pattern_score,
    get_ad_relevance,
    get_core_region_involvement,
    get_spatial_pattern,
    rank_by_ad_importance,
)
from src.evaluation.xai_regions import AAL_REGION_NAMES, RegionAttribution, compute_region_attribution


@dataclass
class ADRegionReport:
    """Structured AD region diagnosis report."""

    prediction: float
    amyloid_status: str
    ad_pattern_score: float
    spatial_pattern: str
    primary_evidence_regions: list[dict]
    core_region_involvement: dict
    clinical_interpretation: str
    regions: list[dict]


def generate_ad_region_report(
    grad_cam: Any,
    atlas_data: Any,
    prediction: float,
    patient_id: str = "anonymous",
) -> ADRegionReport:
    """Generate end-to-end AD region diagnosis report.

    Args:
        grad_cam: 3D numpy array of Grad-CAM values
        atlas_data: 3D numpy array of AAL atlas labels
        prediction: Model output probability (0-1)
        patient_id: Patient identifier

    Returns:
        ADRegionReport with full region analysis.
    """
    if np is None:
        raise RuntimeError("NumPy required for AD region report")

    attributions = compute_region_attribution(grad_cam, atlas_data, AAL_REGION_NAMES)
    ad_pattern_score = compute_ad_pattern_score(attributions)
    spatial_pattern = get_spatial_pattern(attributions)
    core_involved = get_core_region_involvement(attributions)
    ranked = rank_by_ad_importance(attributions)

    amyloid_status = "AB+" if prediction > 0.5 else "AB-"
    primary_evidence = []
    for attr, info in ranked[:10]:
        primary_evidence.append({
            "region": attr.region_name,
            "aal_id": attr.region_id,
            "relevance": info.ad_relevance,
            "attribution": round(float(np.abs(attr.mean_attribution)), 4),
            "clinical_note": info.clinical_notes,
            "typical_change": info.typical_ad_change,
        })

    clinical_interpretation = _build_clinical_interpretation(
        prediction, amyloid_status, ad_pattern_score, spatial_pattern, core_involved, primary_evidence
    )

    all_regions = []
    for attr in attributions:
        info = get_ad_relevance(attr.region_id)
        all_regions.append({
            "region": attr.region_name,
            "aal_id": attr.region_id,
            "attribution": round(float(np.abs(attr.mean_attribution)), 4),
            "ad_relevance": info.ad_relevance if info else "minimal",
        })

    return ADRegionReport(
        prediction=round(prediction, 4),
        amyloid_status=amyloid_status,
        ad_pattern_score=round(ad_pattern_score, 4),
        spatial_pattern=spatial_pattern,
        primary_evidence_regions=primary_evidence,
        core_region_involvement=core_involved,
        clinical_interpretation=clinical_interpretation,
        regions=all_regions,
    )


def _build_clinical_interpretation(
    prediction: float,
    amyloid_status: str,
    ad_pattern_score: float,
    spatial_pattern: str,
    core_involved: dict,
    primary_evidence: list[dict],
) -> str:
    lines = []
    if amyloid_status == "AB+":
        lines.append(f"Model predicts AB+ (amyloid positive) with probability {prediction:.1%}.")
    else:
        lines.append(f"Model predicts AB- (amyloid negative) with probability {1 - prediction:.1%}.")

    if ad_pattern_score > 0.7:
        lines.append("The brain region attribution pattern is STRONGLY consistent with Alzheimer's disease pathology.")
    elif ad_pattern_score > 0.4:
        lines.append("The brain region attribution pattern is MODERATELY consistent with Alzheimer's disease pathology.")
    elif ad_pattern_score > 0.2:
        lines.append("The brain region attribution pattern shows SOME features associated with Alzheimer's disease.")
    else:
        lines.append("The brain region attribution pattern does NOT strongly match typical Alzheimer's disease patterns.")

    if core_involved:
        top_regions = list(core_involved.keys())[:3]
        lines.append(f"Primary involvement in: {', '.join(top_regions)}.")

    pattern_descriptions = {
        "bilateral_hippocampal": "Bilateral hippocampal involvement suggests early-to-moderate stage AD.",
        "unilateral_hippocampal": "Unilateral hippocampal involvement may indicate very early AD or asymmetric presentation.",
        "bilateral_temporal": "Bilateral temporal involvement suggests language and memory network disruption.",
        "unilateral_temporal": "Unilateral temporal involvement may indicate early lateralized pathology.",
        "diffuse_cortical": "Diffuse cortical involvement suggests more advanced disease progression.",
        "no_ad_pattern": "No clear AD-typical spatial pattern detected.",
    }
    desc = pattern_descriptions.get(spatial_pattern, "Spatial pattern requires further clinical correlation.")
    lines.append(desc)

    return " ".join(lines)


def format_region_report_text(report: ADRegionReport) -> str:
    """Format AD region report as human-readable text."""
    lines = [
        "=" * 60,
        "ALZHEIMER'S DISEASE REGION DIAGNOSIS REPORT",
        "=" * 60,
        "",
        f"Prediction: {report.amyloid_status} (probability: {report.prediction:.1%})",
        f"AD Pattern Score: {report.ad_pattern_score:.2f} / 1.00",
        f"Spatial Pattern: {report.spatial_pattern}",
        "",
        "-" * 60,
        "PRIMARY EVIDENCE REGIONS:",
        "-" * 60,
    ]
    for i, region in enumerate(report.primary_evidence_regions, 1):
        lines.append(f"  {i}. {region['region']} ({region['relevance'].upper()})")
        lines.append(f"     Attribution: {region['attribution']:.4f}")
        lines.append(f"     Clinical: {region['clinical_note']}")
        lines.append("")

    lines.append("-" * 60)
    lines.append("CORE REGION INVOLVEMENT:")
    lines.append("-" * 60)
    for region, score in report.core_region_involvement.items():
        bar = "#" * int(score * 50)
        lines.append(f"  {region}: {score:.4f} {bar}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("CLINICAL INTERPRETATION:")
    lines.append("-" * 60)
    lines.append(f"  {report.clinical_interpretation}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("DISCLAIMER: This is a research prediction tool. Results should not be")
    lines.append("used for clinical diagnosis without consultation with a specialist.")
    lines.append("=" * 60)
    return "\n".join(lines)
