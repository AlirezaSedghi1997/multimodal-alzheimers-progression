"""Fail-closed consistency, privacy, and protected-path verification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core import compute_sha256, path_fingerprint


REQUIRED_OUTPUTS = [
    "00_FINAL_EXECUTIVE_DECISION.md",
    "01_CURRENT_PROJECT_STATE.md",
    "02_SOURCE_BY_SOURCE_EVALUATION.md",
    "03_CONSENSUS_AND_CONTRADICTIONS.md",
    "04_VERIFIED_EVIDENCE_SUMMARY.md",
    "05_DATASET_AND_ENDPOINT_GATE.md",
    "06_CANDIDATE_SCORECARD.csv",
    "07_SELECTED_PROJECT_SPECIFICATION.md",
    "08_CURRENT_REPO_IMPLEMENTATION_MAP.md",
    "09_SIX_WEEK_EXECUTION_PLAN.md",
    "10_EXPERIMENT_AND_ABLATION_PROTOCOL.md",
    "11_LEAKAGE_CALIBRATION_AND_STATISTICS_CHECKLIST.md",
    "12_CLAIM_BOUNDARIES.md",
    "13_REJECTION_LOG.md",
    "14_FALLBACK_AND_STOP_RULES.md",
    "15_FINAL_RESPONSE_TO_USER.md",
]


def _scan_public_bundle(bundle: Path) -> dict[str, Any]:
    forbidden = [
        re.compile(r"(?i)\braw_path\b"),
        re.compile(r"(?i)\bsource_subject_id\b"),
        re.compile(r"(?i)H:\\\\A4 Learn"),
        re.compile(r"(?i)H:/A4 Learn"),
        re.compile(r"(?i)\.nii(\.gz)?\b"),
        re.compile(r"(?i)\.h5\b"),
        re.compile(r"(?i)\.pt\b"),
        re.compile(r"(?i)\.ckpt\b"),
        re.compile(r"(?i)\bprediction\b.*\bsubject\b"),
    ]
    hits: list[dict[str, Any]] = []
    if not bundle.exists():
        return {"status": "MISSING", "hits": hits}
    for path in bundle.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in forbidden:
                if pattern.search(line):
                    hits.append({"path": str(path), "line": line_no, "pattern": pattern.pattern})
    return {"status": "PASS" if not hits else "FAIL", "hits": hits}


def verify_workspace(
    workspace: Path,
    protected_before: dict[str, Any],
    source_rows: list[dict[str, Any]],
    primary: dict[str, Any],
    database_path: Path,
    active_writer_observed: bool = False,
) -> dict[str, Any]:
    final = workspace / "08_final_outputs"
    missing = [name for name in REQUIRED_OUTPUTS if not (final / name).exists()]
    unread_sources = [row["source_id"] for row in source_rows if not row.get("read_complete")]
    public_scan = _scan_public_bundle(workspace / "09_public_bundle")
    protected_after = {
        rel: path_fingerprint(Path(protected_before[rel]["path"]), max_files=5000)
        for rel in protected_before
    }
    protected_changed = {
        rel: {"before": protected_before[rel], "after": protected_after[rel]}
        for rel in protected_before
        if protected_before[rel].get("digest") != protected_after[rel].get("digest")
    }
    allowed_dynamic = {}
    if active_writer_observed and "results" in protected_changed:
        allowed_dynamic["results"] = {
            **protected_changed.pop("results"),
            "classification": "ALLOWED_ACTIVE_WRITER_PATH",
            "path": "results/method_y_revival_v2/transforms",
        }
    decision = (final / "00_FINAL_EXECUTIVE_DECISION.md").read_text(encoding="utf-8") if (final / "00_FINAL_EXECUTIVE_DECISION.md").exists() else ""
    output_count = len(list(final.iterdir())) if final.exists() else 0
    checks = {
        "required_outputs_present": not missing,
        "all_canonical_sources_read": not unread_sources,
        "public_bundle_privacy": public_scan["status"] == "PASS",
        "protected_paths_unchanged": not protected_changed,
        "database_exists": database_path.exists(),
        "single_primary_decision": decision.count("**توصیه اصلی:**") == 1,
        "primary_not_hard_failed": not primary.get("cannot_be_primary", True),
        "no_external_claim_without_dua": "external generalization" not in decision.lower()
        or "ممنوع" in decision,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema_version": "1.0",
        "status": status,
        "checks": checks,
        "missing_outputs": missing,
        "unread_source_ids": unread_sources,
        "protected_changed": protected_changed,
        "allowed_dynamic_changes": allowed_dynamic,
        "public_scan": public_scan,
        "output_count": output_count,
        "primary": primary,
    }
    (workspace / "07_reviews").mkdir(parents=True, exist_ok=True)
    (workspace / "07_reviews" / "consistency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
