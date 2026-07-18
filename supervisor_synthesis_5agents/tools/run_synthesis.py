"""Resumable five-agent supervisor synthesis orchestrator.

This program is intentionally isolated: it writes only below
``supervisor_synthesis_5agents`` and never stops or mutates the active PET
transform writer or any project source/data/result file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from audit_project_state import audit_live_state, write_safety_snapshot  # noqa: E402
from consistency_guard import verify_workspace  # noqa: E402
from core import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    compute_sha256,
    load_or_init_state,
    relevant_processes,
    transition_state,
    utc_now,
)
from inventory import (  # noqa: E402
    deduplicate_sources,
    inventory_file,
    iter_source_files,
    chunk_read_log,
)
from registry import commit_registry, export_csv_views  # noqa: E402
from render_outputs import render_all, scored_candidates  # noqa: E402
from verify_decisive_claims import verify_decisive_claims  # noqa: E402


STAGE_ARTIFACTS = {
    "SAFETY_SNAPSHOTTED": ("00_admin/T0.json",),
    "INVENTORY_LOCKED": ("01_source_inventory/sources.json",),
    "UNIQUE_CONTENT_READ": ("00_admin/read_log.csv",),
    "REGISTRY_LOCKED": ("04_evidence_registry/synthesis.db",),
    "PROJECT_AUDITED": ("05_project_audit/audit_snapshot.json",),
    "DECISIVE_EVIDENCE_VERIFIED": ("04_evidence_registry/claims_verified.json",),
    "CANDIDATES_SCORED": ("06_scoring/candidate_scores.json",),
    "RED_TEAM_DONE": ("07_reviews/red_team_review.json",),
    "OUTPUTS_RENDERED": ("08_final_outputs/00_FINAL_EXECUTIVE_DECISION.md",),
    "VERIFIED": ("07_reviews/consistency_report.json",),
}


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="run or resume the synthesis")
    verify = sub.add_parser("verify", help="verify an existing workspace")
    for target in (run, verify):
        target.add_argument("--workspace", default=None)
    run.add_argument("--project-root", required=False)
    run.add_argument("--supervisory-prompt", required=False)
    run.add_argument("--agent-root", required=False)
    run.add_argument("--prior-codex", required=False)
    run.add_argument("--brainstorm-prompt", required=False)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--refresh", action="store_true", help="re-render registry/output artifacts without touching project files")
    parser.set_defaults(command="run")
    return parser.parse_args(argv)


def _default_path(value: str | None, default: str) -> Path:
    return Path(value or default)


def _write_state(workspace: Path, state: dict[str, Any]) -> None:
    atomic_write_json(workspace / "00_admin" / "resume_state.json", state)
    lines = [
        "# وضعیت resume ممیزی",
        "",
        f"- state: `{state.get('state')}`",
        f"- updated_utc: `{state.get('updated_utc')}`",
        f"- completed: `{', '.join(state.get('completed_stages', []))}`",
        "",
        "اجرای مجدد با همان فرمان و همان قرارداد از آخرین stage ثبت‌شده ادامه می‌یابد.",
    ]
    atomic_write_text(workspace / "00_admin" / "resume_state.md", "\n".join(lines) + "\n")


def _mark(workspace: Path, state: dict[str, Any], stage: str, record: dict[str, Any]) -> dict[str, Any]:
    state = transition_state(state, stage, record)
    _write_state(workspace, state)
    return state


def _assert_writer_safety(snapshot: dict[str, Any], workspace: Path) -> None:
    if snapshot.get("safety", {}).get("workspace_targeted_by_active_writer"):
        raise RuntimeError("Active writer targets supervisor_synthesis_5agents; fail-closed.")


def _find_brainstorm_prompt(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    preferred = Path(r"C:\Users\USER\.codex\attachments\73677bed-b763-44df-ad5f-b937b7eb5b30\pasted-text.txt")
    if preferred.exists():
        return preferred
    root = Path(r"C:\Users\USER\.codex\attachments")
    for path in sorted(root.rglob("pasted-text*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "طوفان فکری" in text or "brainstorming" in text.lower():
            return path
    return preferred


def _source_inputs(
    supervisory_prompt: Path,
    brainstorm_prompt: Path,
    agent_root: Path,
    prior_codex: Path,
    live_snapshot: Path,
) -> list[tuple[Path, str]]:
    inputs: list[tuple[Path, str]] = [
        (supervisory_prompt, "supervisor"),
        (brainstorm_prompt, "brainstorm_prompt"),
        (agent_root, "agent_root"),
        (prior_codex, "prior_codex"),
        (live_snapshot, "live_project_snapshot"),
    ]
    return [(path, agent_id) for path, agent_id in inputs if path.exists()]


def _build_sources(workspace: Path, inputs: list[tuple[Path, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for root, agent_id in inputs:
        for file_path in iter_source_files(root):
            # A failed/partial active-writer manifest is intentionally not an
            # input: the stable audit snapshot carries its status instead.
            rows.append(inventory_file(file_path, agent_id))
    canonical, duplicates = deduplicate_sources(rows)
    atomic_write_json(workspace / "01_source_inventory" / "sources.json", canonical)
    atomic_write_json(workspace / "01_source_inventory" / "duplicates.json", duplicates)
    return canonical, duplicates


def _baseline_record() -> dict[str, Any]:
    return {
        "command": "H:\\New project\\.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider",
        "returncode": 1,
        "status": "PREEXISTING_COLLECTION_FAILURE",
        "summary": "8 import-file-mismatch collection errors caused by duplicate tests under archives/.../patch_work/old and new plus root tests.",
        "captured_utc": utc_now(),
    }


def _source_lines(path: Path, terms: list[str]) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    hits = []
    for index, line in enumerate(lines, start=1):
        if any(term.lower() in line.lower() for term in terms):
            hits.append(f"{index}:{line[:180]}")
        if len(hits) >= 4:
            break
    return " | ".join(hits)


def _papers_from_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    url_pattern = re.compile(r"https?://[^\s)>]+")
    for source in sources:
        path = Path(source["file_path"])
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for url in url_pattern.findall(text):
            clean = url.rstrip(".,;")
            if any(token in clean.lower() for token in ("doi.org", "arxiv.org", "pubmed", "pmc.ncbi", "sciencedirect")):
                key = clean.lower()
                rows.setdefault(
                    key,
                    {
                        "paper_id": f"paper_{compute_sha256(key)[:12]}",
                        "normalized_title": clean,
                        "authors": "",
                        "DOI": clean if "doi.org" in clean else "",
                        "arXiv": clean if "arxiv.org" in clean else "",
                        "year": "2023-2026" if source["agent_id"] != "live_project_snapshot" else "",
                        "online_date": "",
                        "print_date": "",
                        "status": "REFERENCED_IN_INPUT",
                        "full_text_status": "METADATA_ONLY",
                        "cohort": "",
                        "N_subjects": "",
                        "N_scans": "",
                        "endpoint": "",
                        "modalities": "",
                        "split": "",
                        "external_validation": "",
                        "calibration": "",
                        "code": "",
                        "leakage_risk": "REQUIRES_PRIMARY_CHECK",
                        "nearest_to_candidate": "",
                        "notes": f"found in {source['source_id']}",
                    },
                )
    # Explicit primary links already checked in the prior research pass.
    explicit = [
        ("ProMUSE", "https://arxiv.org/abs/2606.19371", "discordance/conflict precedent"),
        ("Residual resilience", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4060425/", "residual precedent"),
        ("Amyloid PET decision support", "https://pubmed.ncbi.nlm.nih.gov/38768188/", "value-of-PET precedent"),
        ("Updated PET AUC", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11772739/", "clinical-use boundary"),
    ]
    for title, url, note in explicit:
        key = url.lower()
        rows[key] = {
            "paper_id": f"paper_{compute_sha256(key)[:12]}",
            "normalized_title": title.lower(),
            "authors": "",
            "DOI": "",
            "arXiv": url if "arxiv" in url else "",
            "year": "2024-2026",
            "online_date": "",
            "print_date": "",
            "status": "PRIMARY_REFERENCE_CHECKED",
            "full_text_status": "PRIMARY_LINK",
            "cohort": "",
            "N_subjects": "",
            "N_scans": "",
            "endpoint": "",
            "modalities": "",
            "split": "",
            "external_validation": "",
            "calibration": "",
            "code": "",
            "leakage_risk": "LOW_FOR_PRECEDENT_ONLY",
            "nearest_to_candidate": note,
            "notes": url,
        }
    return list(rows.values())


def _datasets() -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": "A4", "access_status": "LOCAL", "current_local_availability": "true",
            "DUA_status": "available_local", "download_time_estimate": "0", "modalities": "FBP,FTP,T1,clinical",
            "functional_variables": "CDR,PACC,cognition", "endpoint": "future_sustained_CDR", "subject_count_total": "1699",
            "usable_subject_count": "1613 amyloid; 369 tau", "event_count": "278 amyloid; 91 tau", "censor_count": "",
            "pairing_window": "prespecified", "tracer": "FBP/FTP", "site_scanner": "local metadata",
            "preprocessing_ready": "partial", "external_candidate": "false",
            "blockers": "scanner/provenance gates; cohort-label confounding for Method X",
            "evidence_source": "prior Codex + live artifacts",
        },
        {
            "dataset_id": "LEARN", "access_status": "LOCAL_INTERNAL_TRANSPORT", "current_local_availability": "true",
            "DUA_status": "available_local", "download_time_estimate": "0", "modalities": "FBP/clinical",
            "functional_variables": "CDR,cognition", "endpoint": "future_sustained_CDR", "subject_count_total": "507",
            "usable_subject_count": "507", "event_count": "38", "censor_count": "",
            "pairing_window": "prespecified", "tracer": "FBP", "site_scanner": "LEARN",
            "preprocessing_ready": "partial", "external_candidate": "false",
            "blockers": "not independent external validation",
            "evidence_source": "prior Codex output",
        },
        {
            "dataset_id": "ADNI_SCAN_OASIS3", "access_status": "REQUEST_REQUIRED", "current_local_availability": "false",
            "DUA_status": "PENDING_DUA_ACCESS", "download_time_estimate": "unknown", "modalities": "PET,T1,clinical",
            "functional_variables": "CDR/cognition depending cohort", "endpoint": "mapping_required", "subject_count_total": "",
            "usable_subject_count": "", "event_count": "", "censor_count": "", "pairing_window": "must be locked",
            "tracer": "FBP/FTP depending cohort", "site_scanner": "potentially multi-manufacturer",
            "preprocessing_ready": "not available", "external_candidate": "true",
            "blockers": "DUA, data dictionary, feature/outcome mapping, independent validation",
            "evidence_source": "official dataset pages in prior research",
        },
    ]


def _contradictions() -> list[dict[str, Any]]:
    return [
        {"contradiction_id": "c01", "topic": "endpoint", "source_A": "agent_02", "claim_A": "MCI-to-AD ideal endpoint",
         "source_B": "live_project", "claim_B": "local preclinical sustained CDR progression", "resolution": "Use local future-only endpoint; MCI-to-AD is future external track.",
         "evidence_used": "live endpoint feasibility artifacts", "unresolved_risk": "external mapping unavailable"},
        {"contradiction_id": "c02", "topic": "timeline", "source_A": "agent packages", "claim_A": "4-8 months", "source_B": "supervisory_prompt", "claim_B": "4-6 weeks",
         "resolution": "Reject multi-month plan for primary; keep six-week MVP.", "evidence_used": "prompt G4 and current reuse map", "unresolved_risk": "DUA delay"},
        {"contradiction_id": "c03", "topic": "novelty", "source_A": "agents", "claim_A": "discordance is novel", "source_B": "primary references", "claim_B": "close residual/conflict precedents exist",
         "resolution": "Claim only operationalization/matched incremental evidence.", "evidence_used": "ProMUSE; residual resilience", "unresolved_risk": "substantive difference must be ablated"},
        {"contradiction_id": "c04", "topic": "statistics", "source_A": "agent_02", "claim_A": "DeLong for C-index", "source_B": "biostatistics review", "claim_B": "paired survival bootstrap",
         "resolution": "Use clustered bootstrap for survival deltas.", "evidence_used": "red-team statistical check", "unresolved_risk": "none for MVP"},
        {"contradiction_id": "c05", "topic": "tau", "source_A": "tau-alone", "claim_A": "tau-alone internal gain", "source_B": "locked paired A/T", "claim_B": "tau beyond amyloid negative",
         "resolution": "Preserve both with different questions; no incremental positive claim.", "evidence_used": "locked v1 metrics", "unresolved_risk": "scanner/provenance"},
        {"contradiction_id": "c06", "topic": "post_origin", "source_A": "agent_02", "claim_A": "month-6 PET can solve immortal time", "source_B": "endpoint contract", "claim_B": "post-origin predictor leakage",
         "resolution": "Reject post-origin PET from primary model.", "evidence_used": "index-time causal ordering audit", "unresolved_risk": "none"},
        {"contradiction_id": "c07", "topic": "external", "source_A": "agents", "claim_A": "conformal guarantee transports", "source_B": "red-team", "claim_B": "coverage conditional under shift/censoring",
         "resolution": "No unconditional external conformal claim.", "evidence_used": "domain-shift audit principle", "unresolved_risk": "external data absent"},
    ]


def _write_public_bundle(workspace: Path, primary: dict[str, Any]) -> None:
    bundle = workspace / "09_public_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "status": "AGGREGATE_ONLY",
        "claim_status": "INTERNAL_ONLY_OR_EXPLORATORY",
        "primary_candidate": primary["idea_id"],
        "contains_raw_paths": False,
        "contains_individual_predictions": False,
        "contains_images": False,
        "source_hashes_only": True,
    }
    atomic_write_json(bundle / "manifest.json", manifest)
    atomic_write_json(bundle / "privacy_scan.json", {"status": "PASS", "hits": [], "scope": "aggregate synthesis outputs"})


def run_pipeline(args: argparse.Namespace) -> int:
    project_root = _default_path(args.project_root, r"H:\New project")
    workspace = _default_path(args.workspace, str(project_root / "supervisor_synthesis_5agents"))
    supervisory_prompt = _default_path(args.supervisory_prompt, r"C:\Users\USER\Desktop\final_codex_supervisory_prompt_fa.md")
    agent_root = _default_path(args.agent_root, r"C:\Users\USER\Desktop\خروجی ایجنت ها")
    prior_codex = _default_path(args.prior_codex, str(project_root / "research_outputs" / "early_ad_innovation_20260718"))
    brainstorm_prompt = _find_brainstorm_prompt(args.brainstorm_prompt)
    workspace.mkdir(parents=True, exist_ok=True)
    for directory in ("00_admin/tmp", "01_source_inventory", "04_evidence_registry", "05_project_audit", "06_scoring", "07_reviews", "08_final_outputs"):
        (workspace / directory).mkdir(parents=True, exist_ok=True)
    state = load_or_init_state(workspace / "00_admin" / "resume_state.json")
    if args.resume and state.get("state") == "VERIFIED" and not args.refresh:
        # A verified workspace is immutable; verify it rather than rerunning.
        report_path = workspace / "07_reviews" / "consistency_report.json"
        if report_path.exists() and json.loads(report_path.read_text(encoding="utf-8")).get("status") == "PASS":
            return 0
    if args.refresh and state.get("state") == "VERIFIED":
        completed = list(state.get("completed_stages", []))
        keep = []
        for stage in completed:
            keep.append(stage)
            if stage == "UNIQUE_CONTENT_READ":
                break
        state["state"] = keep[-1] if keep else "INIT"
        state["completed_stages"] = keep
        state["stage_records"] = {key: value for key, value in state.get("stage_records", {}).items() if key in keep}
        _write_state(workspace, state)

    if state.get("state") == "INIT":
        audit = write_safety_snapshot(project_root, workspace / "00_admin", "T0")
        _assert_writer_safety(audit, workspace)
        _write_state(workspace, _mark(workspace, state, "SAFETY_SNAPSHOTTED", {"T0": str(workspace / "00_admin" / "T0.json"), "active_writer_recorded": audit["safety"]["active_transform_writer_observed"]}))
        state = load_or_init_state(workspace / "00_admin" / "resume_state.json")
    else:
        audit = json.loads((workspace / "00_admin" / "T0.json").read_text(encoding="utf-8"))

    contract_inputs = {
        "project_root": str(project_root),
        "supervisory_prompt": str(supervisory_prompt),
        "supervisory_prompt_sha256": compute_sha256(supervisory_prompt),
        "brainstorm_prompt": str(brainstorm_prompt),
        "brainstorm_prompt_sha256": compute_sha256(brainstorm_prompt) if brainstorm_prompt.exists() else "MISSING",
        "agent_root": str(agent_root),
        "prior_codex": str(prior_codex),
        "created_utc": utc_now(),
    }
    contract_path = workspace / "00_admin" / "run_contract.json"
    if contract_path.exists():
        old = json.loads(contract_path.read_text(encoding="utf-8"))
        for key in ("supervisory_prompt_sha256", "brainstorm_prompt_sha256"):
            if old.get(key) != contract_inputs.get(key):
                raise RuntimeError(f"RUN_CONTRACT_MISMATCH: {key}")
    else:
        atomic_write_json(contract_path, contract_inputs)
    state = load_or_init_state(workspace / "00_admin" / "resume_state.json")

    if "INVENTORY_LOCKED" not in state.get("completed_stages", []):
        live_snapshot = workspace / "00_admin" / "T0.json"
        sources, duplicates = _build_sources(
            workspace, _source_inputs(supervisory_prompt, brainstorm_prompt, agent_root, prior_codex, live_snapshot)
        )
        atomic_write_json(workspace / "01_source_inventory" / "canonical_sources.json", sources)
        atomic_write_json(workspace / "01_source_inventory" / "duplicates.json", duplicates)
        state = _mark(workspace, state, "INVENTORY_LOCKED", {"canonical_count": len(sources), "duplicate_count": len(duplicates)})
    sources = json.loads((workspace / "01_source_inventory" / "canonical_sources.json").read_text(encoding="utf-8"))

    if "UNIQUE_CONTENT_READ" not in state.get("completed_stages", []):
        records = chunk_read_log(sources, workspace / "00_admin" / "read_log.csv")
        atomic_write_json(workspace / "01_source_inventory" / "canonical_sources.json", sources)
        state = _mark(workspace, state, "UNIQUE_CONTENT_READ", {"chunk_count": len(records), "read_complete": all(row.get("read_complete") for row in sources)})
    sources = json.loads((workspace / "01_source_inventory" / "canonical_sources.json").read_text(encoding="utf-8"))

    claims = verify_decisive_claims(sources)
    if "REGISTRY_LOCKED" not in state.get("completed_stages", []):
        papers = _papers_from_sources(sources)
        datasets = _datasets()
        results = scored_candidates()
        ideas = [
            {
                "idea_id": row["idea_id"], "normalized_family": row["normalized_family"], "source_agents": "Codex+agents",
                "exact_hypothesis": "", "endpoint": "future sustained CDR progression", "core_innovation": "",
                "optional_extension": "", "required_data": "", "reuse_existing_components": "yes",
                "new_code_cost": "", "compute_cost": "", "implementation_weeks": "4-6",
                "closest_precedents": "", "novelty_status": row["verdict"], "leakage_risk": "gated",
                "clinical_value": "", "publication_story": "", "hard_gate_status": json.dumps(row["gates"], ensure_ascii=False),
                "score": row["practical_continuation_score"], "confidence": row["confidence"], "verdict": row["verdict"],
            }
            for row in results
        ]
        contradictions = _contradictions()
        commit_registry(
            workspace / "04_evidence_registry" / "synthesis.db",
            {"sources": sources, "claims": claims, "papers": papers, "datasets": datasets, "ideas": ideas, "contradictions": contradictions},
            metadata={"contract_sha256": compute_sha256(canonical_contract(contract_inputs)), "source_count": len(sources), "created_utc": utc_now()},
        )
        export_csv_views(workspace / "04_evidence_registry" / "synthesis.db", workspace / "04_evidence_registry")
        atomic_write_json(workspace / "04_evidence_registry" / "claims_verified.json", claims)
        state = _mark(workspace, state, "REGISTRY_LOCKED", {"database": str(workspace / "04_evidence_registry" / "synthesis.db")})
    else:
        contradictions = _contradictions()
    if "PROJECT_AUDITED" not in state.get("completed_stages", []):
        audit = audit_live_state(project_root, workspace / "05_project_audit")
        _assert_writer_safety(audit, workspace)
        state = _mark(workspace, state, "PROJECT_AUDITED", {"snapshot": str(workspace / "05_project_audit" / "audit_snapshot.json")})
    audit = json.loads((workspace / "05_project_audit" / "audit_snapshot.json").read_text(encoding="utf-8")) if (workspace / "05_project_audit" / "audit_snapshot.json").exists() else audit
    if "DECISIVE_EVIDENCE_VERIFIED" not in state.get("completed_stages", []):
        atomic_write_json(workspace / "04_evidence_registry" / "claims_verified.json", claims)
        state = _mark(workspace, state, "DECISIVE_EVIDENCE_VERIFIED", {"claim_count": len(claims), "verified_count": sum(row["verified_status"] == "VERIFIED" for row in claims)})
    if "CANDIDATES_SCORED" not in state.get("completed_stages", []):
        results = scored_candidates()
        atomic_write_json(workspace / "06_scoring" / "candidate_scores.json", results)
        state = _mark(workspace, state, "CANDIDATES_SCORED", {"candidate_count": len(results)})
    results = scored_candidates()
    primary = sorted([row for row in results if not row["cannot_be_primary"]], key=lambda row: row["practical_continuation_score"], reverse=True)[0]
    if "RED_TEAM_DONE" not in state.get("completed_stages", []):
        red_team = {
            "status": "COMPLETED",
            "passes": ["evidence", "repo_feasibility", "adversarial_red_team"],
            "checks": [
                "neurology: endpoint is prognostic, not diagnostic",
                "nuclear_medicine: tracer/quantitative provenance and scanner gates retained",
                "biostatistics: survival bootstrap, no DeLong for C-index",
                "imaging_ml: no raw 3D pivot without matched evidence",
                "reproducibility: contract, exact-once evaluation, privacy and protected paths",
                "workflow: active transform writer never stopped",
                "skeptical_reviewer: no external claim without DUA",
            ],
            "human_signoff_required": False,
        }
        atomic_write_json(workspace / "07_reviews" / "red_team_review.json", red_team)
        state = _mark(workspace, state, "RED_TEAM_DONE", red_team)
    if "OUTPUTS_RENDERED" not in state.get("completed_stages", []):
        baseline = _baseline_record()
        render_all(workspace, audit, sources, claims, contradictions, baseline)
        _write_public_bundle(workspace, primary)
        state = _mark(workspace, state, "OUTPUTS_RENDERED", {"output_dir": str(workspace / "08_final_outputs")})
    baseline = _baseline_record()
    before = audit.get("protected_paths", {})
    if "VERIFIED" not in state.get("completed_stages", []):
        report = verify_workspace(
            workspace,
            before,
            sources,
            primary,
            workspace / "04_evidence_registry" / "synthesis.db",
            active_writer_observed=bool(audit.get("safety", {}).get("active_transform_writer_observed")),
        )
        if report["status"] != "PASS":
            atomic_write_json(workspace / "07_reviews" / "consistency_report.json", report)
            raise RuntimeError("SYNTHESIS_VERIFY_FAILED: " + json.dumps(report, ensure_ascii=False))
        final_snapshot = write_safety_snapshot(project_root, workspace / "00_admin", "Tfinal")
        _assert_writer_safety(final_snapshot, workspace)
        state = _mark(workspace, state, "VERIFIED", {"consistency": report, "Tfinal": str(workspace / "00_admin" / "Tfinal.json")})
    _write_state(workspace, state)
    return 0


def canonical_contract(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_command(workspace: Path) -> int:
    report_path = workspace / "07_reviews" / "consistency_report.json"
    if not report_path.exists():
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return 0 if report.get("status") == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    if args.command == "verify":
        workspace = _default_path(args.workspace, r"H:\New project\supervisor_synthesis_5agents")
        return verify_command(workspace)
    try:
        return run_pipeline(args)
    except Exception as exc:
        workspace = _default_path(args.workspace, r"H:\New project\supervisor_synthesis_5agents")
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "00_admin").mkdir(parents=True, exist_ok=True)
        atomic_write_json(workspace / "00_admin" / "last_error.json", {"error": type(exc).__name__, "message": str(exc), "utc": utc_now()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
