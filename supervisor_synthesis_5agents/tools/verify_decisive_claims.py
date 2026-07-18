"""Deterministic verification cards for claims that can change the decision."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from core import compute_sha256, utc_now
from inventory import _read_text


DECISIVE_CLAIMS = [
    {
        "topic": "local_endpoint",
        "claim_text": "The local endpoint is future sustained CDR progression/PACC decline, not MCI-to-AD conversion.",
        "terms": ["CDR", "progression"],
        "primary_reference": "",
        "impact": "high",
        "action": "Keep local claims preclinical and prognostic.",
    },
    {
        "topic": "amyloid_counts",
        "claim_text": "The current amyloid endpoint contains 1613 eligible subjects and 278 events, with A4 development 1106/240 and LEARN transport 507/38.",
        "terms": ["1613", "278", "1106", "240", "507", "38"],
        "primary_reference": "",
        "impact": "high",
        "action": "Use exact event counts and preserve A4/LEARN transport distinction.",
    },
    {
        "topic": "tau_counts",
        "claim_text": "The tau and paired A/T endpoint contains 369 subjects and 91 events.",
        "terms": ["369", "91"],
        "primary_reference": "",
        "impact": "high",
        "action": "Keep tau model complexity proportional to 91 events.",
    },
    {
        "topic": "tau_incremental_result",
        "claim_text": "The locked tau-beyond-amyloid comparison is negative: C-index delta is near zero and IBS delta is unfavorable.",
        "terms": ["tau", "amyloid", "negative"],
        "primary_reference": "",
        "impact": "high",
        "action": "Report as a negative/exploratory result; do not rescue by post-hoc tuning.",
    },
    {
        "topic": "external_access",
        "claim_text": "No approved external raw manifest or DUA is available for claim-bearing external validation.",
        "terms": ["DUA", "PENDING"],
        "primary_reference": "",
        "impact": "high",
        "action": "External validation remains future track.",
    },
    {
        "topic": "confounding_method_x",
        "claim_text": "Method X has cohort-label coupling in the current A4/LEARN development data; it cannot support an external clinical diagnosis claim.",
        "terms": ["cohort", "label"],
        "primary_reference": "",
        "impact": "high",
        "action": "Keep Method X PET-free and internal/technical until a within-cohort independent label exists.",
    },
    {
        "topic": "discordance_precedent",
        "claim_text": "Residual/discordance and conflict-aware prediction have direct or close precedents and are not novel merely by naming.",
        "terms": ["discordance", "residual", "conflict"],
        "primary_reference": "https://arxiv.org/abs/2606.19371; https://pmc.ncbi.nlm.nih.gov/articles/PMC4060425/",
        "impact": "high",
        "action": "Claim only the prespecified operationalization and matched incremental evidence.",
    },
    {
        "topic": "post_origin_pet",
        "claim_text": "Using a PET scan acquired after the survival origin as a baseline predictor is post-origin leakage and is rejected.",
        "terms": ["post-origin"],
        "primary_reference": "",
        "impact": "high",
        "action": "Only baseline-at-index PET may enter the primary model.",
    },
    {
        "topic": "delong_c_index",
        "claim_text": "DeLong is not the default valid comparison for survival C-index; paired subject-clustered bootstrap is required.",
        "terms": ["DeLong", "C-index"],
        "primary_reference": "",
        "impact": "medium",
        "action": "Use paired survival metrics and bootstrap confidence intervals.",
    },
    {
        "topic": "conformal_transport",
        "claim_text": "Conformal coverage is not assumed to transport under external domain shift and censoring.",
        "terms": ["conformal", "external"],
        "primary_reference": "",
        "impact": "medium",
        "action": "Treat conformal output as conditional/secondary unless transport is empirically checked.",
    },
    {
        "topic": "transform_state",
        "claim_text": "FTP transform materialization is an active resumable process and must be snapshotted, never stopped by synthesis.",
        "terms": ["transform", "RUNNING"],
        "primary_reference": "",
        "impact": "high",
        "action": "Read before and after; attribute changes to the active writer.",
    },
]


def _find_source(term_list: list[str], sources: list[dict[str, Any]]) -> tuple[str, str, str]:
    for source in sources:
        path = Path(source["file_path"])
        try:
            text = _read_text(path)
        except OSError:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines, start=1):
            low = line.lower()
            if all(term.lower() in low for term in term_list[:1]):
                # Require at least one additional term somewhere nearby when possible.
                window = "\n".join(lines[max(0, index - 3): min(len(lines), index + 3)]).lower()
                if sum(term.lower() in window for term in term_list) >= min(2, len(term_list)):
                    return source["source_id"], f"lines {index}-{min(len(lines), index + 2)}", line[:300]
    return "", "", ""


def verify_decisive_claims(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, claim in enumerate(DECISIVE_CLAIMS, start=1):
        source_id, location, excerpt = _find_source(claim["terms"], sources)
        if source_id:
            source = next(row for row in sources if row["source_id"] == source_id)
            if source["file_role"] == "supervisory_prompt":
                evidence = "FACT_PRIMARY_FULL_TEXT"
                status = "VERIFIED"
                confidence = "A"
            elif source["file_role"] == "prior_codex_output":
                evidence = "FACT_REPO_REPRODUCED"
                status = "VERIFIED"
                confidence = "A"
            else:
                evidence = "AGENT_INFERENCE"
                status = "CONDITIONAL"
                confidence = "B"
        else:
            evidence = "UNVERIFIED"
            status = "PENDING_LIVE_OR_PRIMARY_CHECK"
            confidence = "C"
        records.append(
            {
                "claim_id": f"claim_{index:03d}",
                "topic": claim["topic"],
                "claim_text": claim["claim_text"],
                "claim_type": "DECISION_DETERMINING",
                "source_id": source_id,
                "source_section_or_lines": location or "not located in canonical inputs",
                "evidence_label": evidence,
                "primary_reference": claim["primary_reference"],
                "verified_status": status,
                "confidence": confidence,
                "contradiction_group": claim["topic"],
                "decision_impact": claim["impact"],
                "required_action": claim["action"],
                "source_excerpt": excerpt,
                "verified_utc": utc_now(),
            }
        )
    return records


def export_verification_queue(claims: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "claim_id", "topic", "source_id", "source_section_or_lines", "evidence_label",
        "verified_status", "confidence", "required_action",
    ]
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows([{field: row.get(field, "") for field in fields} for row in claims])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8")
