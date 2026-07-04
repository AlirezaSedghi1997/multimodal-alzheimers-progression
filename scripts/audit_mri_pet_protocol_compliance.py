#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spm_completion_guard import spm_completion_status


PROTOCOL_FILES = [
    "ALL_IN_ONE_AI_READABLE_PROTOCOL.md",
    "03_pipeline_rules.yaml",
    "05_branch_pipelines.yaml",
    "04_manifest_schema.yaml",
    "06_qc_audit_checklist.yaml",
    "07_augmentation_imbalance.yaml",
    "01_scientific_knowledge_base.md",
    "08_evidence_table.md",
    "02_technical_implementation_blueprint.md",
    "09_open_questions.md",
    "10_agent_execution_prompt.md",
]
METHOD_X_FORBIDDEN_TERMS = (
    "pet",
    "suvr",
    "centiloid",
    "amyloid",
    "tau",
    "tracer",
)
LEGACY_PET_CACHE = "caches/method_y/pet_mni_128.h5"
LEGACY_PET_DEFAULT_PATTERNS = (
    'get("pet_h5", "caches/method_y/pet_mni_128.h5")',
    "get('pet_h5', 'caches/method_y/pet_mni_128.h5')",
    'inputs.get("pet_h5", "caches/method_y/pet_mni_128.h5")',
    "inputs.get('pet_h5', 'caches/method_y/pet_mni_128.h5')",
    'DEFAULT_PET_H5 = "caches/method_y/pet_mni_128.h5"',
    "DEFAULT_PET_H5 = 'caches/method_y/pet_mni_128.h5'",
)
LEGACY_REFERENCE_ALLOWLIST = {
    "scripts/audit_mri_pet_protocol_compliance.py",
    "scripts/method_y/_stage_common.py",
    "tests/test_preprocess_protocol_compliance.py",
}
SAFE_PET_CACHE_REPORTS = {
    "suvr_preserved_v1": Path("results/method_y_execution/suvr_preserved_v1/cache_variant_report.json"),
    "broad_s1_fbp_only_hq_v1": Path("results/method_y_execution/broad_s1_fbp_only_hq_v1/cache_variant_report.json"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def protocol_hashes(protocol_dir: Path) -> dict[str, Any]:
    files = {}
    missing = []
    for name in PROTOCOL_FILES:
        digest = sha256_file(protocol_dir / name)
        if digest is None:
            missing.append(name)
        else:
            files[name] = digest
    return {
        "protocol_dir": str(protocol_dir),
        "files_hashed": files,
        "missing": missing,
        "all_required_present": not missing,
    }


def resume_counts(path: Path) -> dict[str, Any]:
    payload = read_json(path) or {}
    items = payload.get("items") if isinstance(payload, dict) else {}
    if not isinstance(items, dict):
        items = {}
    return {
        "path": rel(path),
        "exists": path.exists(),
        "stage": payload.get("stage"),
        "item_count": len(items),
        "status_counts": dict(Counter(str((item or {}).get("status", "UNKNOWN")) for item in items.values())),
    }


def method_x_catalog_audit(path: Path) -> dict[str, Any]:
    rows = csv_rows(path)
    columns = list(rows[0].keys()) if rows else []
    forbidden_columns = [
        col for col in columns if any(term in col.lower() for term in METHOD_X_FORBIDDEN_TERMS)
    ]
    return {
        "path": rel(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "columns": columns,
        "forbidden_pet_related_columns": forbidden_columns,
        "status": "PASS" if rows and not forbidden_columns else "FAIL",
    }


def method_y_pet_input_audit(path: Path) -> dict[str, Any]:
    rows = csv_rows(path)
    columns = list(rows[0].keys()) if rows else []
    tracer_counts = Counter(str(row.get("tracer", "")).upper() for row in rows)
    non_fbp = sorted(k for k in tracer_counts if k and k != "FBP")
    raw_path_columns = [col for col in columns if "path" in col.lower()]
    return {
        "path": rel(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "columns": columns,
        "tracer_counts": dict(tracer_counts),
        "non_fbp_tracers": non_fbp,
        "raw_path_columns_private_only": raw_path_columns,
        "status": "PASS" if rows and not non_fbp else "FAIL",
    }


def public_privacy_scan(root: Path) -> dict[str, Any]:
    patterns = [
        re.compile(r"[A-Z]:\\"),
        re.compile(r'(?<![A-Za-z0-9_])raw_subject_id(?![A-Za-z0-9_])\s*[:=]', re.I),
        re.compile(r'(?<![A-Za-z0-9_])raw_path(?![A-Za-z0-9_])\s*[:=]', re.I),
    ]
    matches: list[dict[str, Any]] = []
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".csv", ".yaml", ".yml", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hit_terms = [p.pattern for p in patterns if p.search(text)]
            if hit_terms:
                matches.append({"path": rel(path), "patterns": hit_terms})
                if len(matches) >= 50:
                    break
    return {
        "root": rel(root),
        "matches_first_50": matches,
        "status": "PASS" if not matches else "FAIL",
    }


def safe_pet_cache_audit() -> dict[str, Any]:
    out: dict[str, Any] = {}
    legacy = PROJECT_ROOT / LEGACY_PET_CACHE
    out["legacy_zscore_cache"] = {
        "path": LEGACY_PET_CACHE,
        "exists": legacy.exists(),
        "final_claim_status": "prohibited",
        "reason": "Legacy Method Y PET cache is documented as per-scan z-score; PET SUVR/Centiloid/tau-burden final claims require SUVR-preserving inputs.",
        "replacement": "caches/method_y/pet_mni_128_broad_s1_fbp_only_hq_v1.h5",
    }
    for name, report_rel in SAFE_PET_CACHE_REPORTS.items():
        report_path = PROJECT_ROOT / report_rel
        report = read_json(report_path) or {}
        contract = report.get("preprocessing_contract") or {}
        per_scan_zscore = contract.get("per_scan_zscore")
        tracer_counts = (report.get("source_summary") or {}).get("per_tracer_subject_count") or {}
        status = "defensible_with_audit"
        blockers = []
        if report.get("status") != "DONE":
            status = "blocked"
            blockers.append("CACHE_VARIANT_NOT_DONE")
        if per_scan_zscore is not False:
            status = "blocked"
            blockers.append("PER_SCAN_ZSCORE_NOT_FALSE")
        if sorted(tracer_counts) != ["FBP"]:
            status = "blocked"
            blockers.append("TRACER_SET_NOT_FBP_ONLY")
        out[name] = {
            "report": rel(report_path),
            "exists": report_path.exists(),
            "cache": rel(PROJECT_ROOT / str(report.get("out_mni", ""))) if report.get("out_mni") else None,
            "subjects_ok": report.get("subjects_ok"),
            "intensity_mode": report.get("intensity_mode"),
            "preprocessing_contract": contract,
            "tracer_counts": tracer_counts,
            "status": status,
            "blockers": blockers,
            "claim_boundary": report.get("claim_boundary"),
        }
    return out


def config_legacy_references() -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for path in sorted((PROJECT_ROOT / "configs").glob("method_y_stage*.yaml")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if LEGACY_PET_CACHE in text.replace("\\", "/"):
            refs.append({"path": rel(path), "reference": LEGACY_PET_CACHE})
    return refs


def code_legacy_default_references() -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    roots = [PROJECT_ROOT / "scripts" / "method_y", PROJECT_ROOT / "src" / "training"]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            rel_path = rel(path)
            if rel_path in LEGACY_REFERENCE_ALLOWLIST:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").replace("\\", "/")
            for pattern in LEGACY_PET_DEFAULT_PATTERNS:
                if pattern in text:
                    refs.append({"path": rel_path, "pattern": pattern, "reference": LEGACY_PET_CACHE})
    return refs


def stage_gate_summary() -> dict[str, Any]:
    stage_reports = {}
    for path in sorted((PROJECT_ROOT / "results/method_y_execution").glob("stage_s*_report.json")):
        payload = read_json(path) or {}
        stage_reports[path.stem] = {
            "path": rel(path),
            "status": payload.get("status"),
            "acceptance_gate_passed": payload.get("acceptance_gate_passed"),
            "blocker": payload.get("blocker"),
            "next_allowed_stage": payload.get("next_allowed_stage"),
        }
    s1_audit = read_json(PROJECT_ROOT / "results/method_y_execution/s1_gate_failure_audit.json") or {}
    return {
        "stage_reports": stage_reports,
        "s1_gate_failure_audit": {
            "status": s1_audit.get("status"),
            "tracer_probe_status": s1_audit.get("tracer_probe_status"),
            "pet_input_list_summary": s1_audit.get("pet_input_list_summary"),
        },
    }


def build_report(protocol_dir: Path) -> dict[str, Any]:
    spm_status = spm_completion_status()
    mri_resume = resume_counts(PROJECT_ROOT / "results/logs/spm_mri_preprocess_v2_128_resume.json")
    pet_resume = resume_counts(PROJECT_ROOT / "results/logs/method_y_pet_preprocess_resume_fbp_spm_pet_t1_mni.json")
    method_x = method_x_catalog_audit(PROJECT_ROOT / "data/processed/catalogs/method_x_strict_catalog.csv")
    method_y = method_y_pet_input_audit(PROJECT_ROOT / "data/method_y/pet_input_list_broad_s1_fbp_only.csv")
    if not method_y["exists"]:
        method_y = method_y_pet_input_audit(PROJECT_ROOT / "data/method_y/pet_input_list.csv")
    cache_audit = safe_pet_cache_audit()
    legacy_refs = config_legacy_references()
    code_legacy_refs = code_legacy_default_references()
    blockers = []
    if not spm_status.get("ready"):
        blockers.append("MRI_SPM_INCOMPLETE")
    if method_x.get("status") != "PASS":
        blockers.append("METHOD_X_PET_FREE_CATALOG_FAIL")
    if method_y.get("status") != "PASS":
        blockers.append("METHOD_Y_FBP_TRACER_VALIDATION_FAIL")
    if legacy_refs:
        blockers.append("LEGACY_PET_ZSCORE_CACHE_REFERENCED_BY_STAGE_CONFIG")
    if code_legacy_refs:
        blockers.append("LEGACY_PET_ZSCORE_CACHE_REFERENCED_BY_CODE_DEFAULT")

    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "script": "scripts/audit_mri_pet_protocol_compliance.py",
        "protocol": protocol_hashes(protocol_dir),
        "task_interpretation": {
            "affected_branches": ["MethodX", "MethodY_FBP_Amyloid", "Tau_FTP_Progression", "Multimodal_MRI_PET"],
            "purpose": "Audit current MRI/PET preprocessing outputs against the July 2026 protocol before destructive replacement.",
            "destructive_replacement_authorized_by_this_report": False,
        },
        "mri": {
            "spm_completion_guard": spm_status,
            "resume_manifest": mri_resume,
            "decision": "rerun_or_resume_required" if not spm_status.get("ready") else "usable_with_audit",
        },
        "pet": {
            "resume_manifest": pet_resume,
            "method_y_input_manifest": method_y,
            "cache_audit": cache_audit,
            "decision": "legacy_cache_supersede_only_use_suvr_preserving_cache",
        },
        "method_x": {
            "strict_catalog": method_x,
            "decision": "pet_free_guard_passed" if method_x.get("status") == "PASS" else "blocked",
        },
        "privacy": {
            "public_bundle": public_privacy_scan(PROJECT_ROOT / "public_bundle"),
        },
        "stage_gates": stage_gate_summary(),
        "config_legacy_references": legacy_refs,
        "code_legacy_default_references": code_legacy_refs,
        "blockers": blockers,
        "final_claim_status": {
            "MethodX_MRI": "blocked_until_spm_complete" if not spm_status.get("ready") else "defensible_with_audit",
            "MethodY_FBP_PET": "defensible_with_audit_for_preprocessing_cache_but_science_gate_still_required",
            "Tau_FTP": "exploratory_only_unless_progression_specific_manifest_and_off_target_audit_pass",
            "Multimodal": "blocked_until_common_geometry_and_scanner_cohort_probe_pass",
            "legacy_method_y_pet_zscore_cache": "prohibited_for_final_biomarker_claim",
        },
        "recommended_actions": [
            "Do not delete previous outputs blindly; mark legacy PET z-score cache and dependent reports as superseded/not-final.",
            "Resume official MRI SPM preprocessing through scripts/run_mri_spm_128_resume.ps1 after resetting stale RUNNING entries if no writer is active.",
            "Use SUVR-preserving FBP-only Method Y caches for any future Method Y runs.",
            "Keep Method X loaders on data/processed/catalogs/method_x_strict_catalog.csv, not master_catalog.csv.",
            "Run scanner/site/cohort probes and subgroup metrics before any public claim.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MRI/PET Protocol Compliance Audit",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Protocol directory: `{report['protocol']['protocol_dir']}`",
        f"- Protocol files present: `{report['protocol']['all_required_present']}`",
        "",
        "## Current Decision",
        "",
        f"- MRI: `{report['mri']['decision']}`",
        f"- PET: `{report['pet']['decision']}`",
        f"- Method X: `{report['method_x']['decision']}`",
        f"- Blockers: `{report['blockers']}`",
        "",
        "## MRI",
        "",
        f"- SPM status: `{report['mri']['spm_completion_guard']['status']}`",
        f"- Done count: `{report['mri']['spm_completion_guard']['done_count']}` / `{report['mri']['spm_completion_guard']['expected_total']}`",
        f"- QC missing: `{report['mri']['spm_completion_guard']['preprocessed_qc_missing_count']}`",
        f"- Resume status counts: `{report['mri']['resume_manifest']['status_counts']}`",
        "",
        "## PET",
        "",
        f"- PET resume status counts: `{report['pet']['resume_manifest']['status_counts']}`",
        f"- Method Y tracer counts: `{report['pet']['method_y_input_manifest']['tracer_counts']}`",
        f"- Legacy cache final-claim status: `{report['pet']['cache_audit']['legacy_zscore_cache']['final_claim_status']}`",
        f"- Broad HQ cache status: `{report['pet']['cache_audit'].get('broad_s1_fbp_only_hq_v1', {}).get('status')}`",
        f"- Unsafe legacy Python defaults: `{len(report.get('code_legacy_default_references', []))}`",
        "",
        "## Privacy",
        "",
        f"- Public bundle raw-path scan: `{report['privacy']['public_bundle']['status']}`",
        "",
        "## Required Actions",
        "",
    ]
    lines.extend(f"- {item}" for item in report["recommended_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit MRI/PET preprocessing against the external July 2026 protocol.")
    parser.add_argument("--protocol_dir", default=r"C:\Users\USER\Desktop\edit preprocess")
    parser.add_argument("--out_json", default="results/tables/preprocess_protocol_compliance_audit.json")
    parser.add_argument("--out_md", default="results/tables/preprocess_protocol_compliance_audit.md")
    args = parser.parse_args()

    report = build_report(Path(args.protocol_dir))
    out_json = PROJECT_ROOT / args.out_json
    out_md = PROJECT_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "OK", "out_json": rel(out_json), "blockers": report["blockers"]}, indent=2))
    return 1 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
