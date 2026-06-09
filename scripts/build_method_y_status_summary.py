from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.pet_preprocess_progress import summarize_pet_preprocess_progress
from src.utils.state_manager import StateManager


BLOCKER_GLOSSARY: dict[str, str] = {
    "REGEX_FALLBACK_MANIFEST_NOT_TRAINING_READY": (
        "PET file manifest still relies on regex fallback; training-grade private manifest is required."
    ),
    "PRIVATE_MANIFEST_NOT_TRAINING_READY": (
        "Hash-level private PET manifest is incomplete or not validated for one-PET-per-patient discipline."
    ),
    "PRIMARY_TIER_MIN_CLASS_BELOW_BLOCK_THRESHOLD": (
        "Primary-tier Method Y split has insufficient minority-class support for stable training."
    ),
    "PREPROCESS_NOT_EXECUTED": (
        "Method Y PET spatial preprocessing / cache build has not been executed."
    ),
    "NO_QC_PASS_PET_OUTPUTS": (
        "No PET preprocessing outputs passed spatial QC gates."
    ),
}

WARNING_GLOSSARY: dict[str, str] = {
    "COHORT_LABEL_AUDIT_POSSIBLE_STRONG_COUPLING": "Amyloid label may remain coupled to cohort/tracer in Method Y splits.",
    "TRACER_LABEL_AUDIT_SINGLE_ROW_OR_COLUMN_REVIEW": "Tracer vs label contingency needs manual review.",
    "GAP_LABEL_AUDIT_POSSIBLE_STRONG_COUPLING": "PET–T1 temporal gap bins may correlate with label.",
    "PET_CACHE_NOT_BUILT_YET": "Method Y PET volume cache has not been built.",
}


def load(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"status": "MISSING"}
    return json.loads(p.read_text(encoding="utf-8"))


def nested(obj: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main() -> int:
    readiness = load("results/tables/method_y_training_readiness.json")
    design = load("results/tables/method_y_multimodal_design.json")
    inventory = load("results/tables/method_y_pet_inventory.json")
    phase_y = load("results/final_results/method_y/phase_y_summary.json")
    gap = load("results/tables/method_y_gap_filter_report.json")
    manifest = load("results/tables/method_y_pet_manifest_summary.json")
    preprocess = load("results/tables/method_y_pet_preprocess_summary.json")
    linked = int(manifest.get("linked_work_items") or nested(manifest, "link_summary", "linked_work_items") or 0)
    preprocess_progress = summarize_pet_preprocess_progress(linked_work_items=linked)
    availability = design.get("availability_summary", {})
    inv_amyloid = nested(inventory, "pet_image_inventory", "amyloid_fbp") or {}
    inv_tau = nested(inventory, "pet_image_inventory", "tau_ftp") or {}

    summary: dict[str, Any] = {
        "schema_version": 2,
        "branch": "method_y_pet_assisted",
        "status": readiness.get("status", "MISSING"),
        "design_status": design.get("status", "MISSING"),
        "inventory_status": inventory.get("status", "MISSING"),
        "phase_y_status": phase_y.get("status", "MISSING"),
        "training_started": design.get("training_started", False),
        "pet_preprocessing_started": preprocess_progress["preprocessing_started"],
        "pet_preprocess_progress": preprocess_progress,
        "method_x_claim_unchanged": readiness.get("method_x_claim_unchanged", True),
        "pet_as_method_x_input": False,
        "clinical_alzheimer_diagnosis_allowed": readiness.get("clinical_alzheimer_diagnosis_allowed", False),
        "current_primary_target": nested(design, "target_policy", "current_primary_target"),
        "claim_boundary": (
            "Exploratory PET-assisted branch (Method Y). PET never enters Method X-Strict or Method X-Cog. "
            "Multimodal PET/MRI training has not started; only design, inventory, and readiness audits exist."
        ),
        "availability_summary": availability,
        "pet_image_inventory": {
            "amyloid_fbp_files": inv_amyloid.get("image_file_count"),
            "amyloid_locked_patient_matches": inv_amyloid.get("locked_catalog_patient_matches_in_memory"),
            "tau_ftp_files": inv_tau.get("image_file_count"),
            "tau_locked_patient_matches": inv_tau.get("locked_catalog_patient_matches_in_memory"),
        },
        "modality_scenarios": design.get("modality_scenarios", []),
        "execution_gates_before_training": design.get("execution_gates_before_training", []),
        "blockers": readiness.get("blockers", []),
        "blocker_glossary": {k: BLOCKER_GLOSSARY.get(k, "See Method Y readiness policy.") for k in readiness.get("blockers", [])},
        "warnings": readiness.get("warnings", []),
        "warning_glossary": {k: WARNING_GLOSSARY.get(k, "Review Method Y gap/cohort audits.") for k in readiness.get("warnings", [])},
        "required_before_training": readiness.get("required_before_training", []),
        "resolution_checklist": [
            "Build and validate private PET manifest (no regex fallback).",
            "Execute Method Y PET spatial preprocessing and pass QC gates.",
            "Build PET volume cache separate from Method X T1 cache.",
            "Confirm patient-level split with one amyloid PET per patient.",
            "Run cohort/tracer/gap confounding audits on Method Y split plan.",
            "Keep calibration/ECE/Brier/CI plan train-fold-only.",
        ],
        "gap_filter_status": gap.get("status"),
        "preprocess_summary_verdict": preprocess.get("verdict"),
    }

    out_json = Path("docs/method_y_status_summary.json")
    out_md = Path("docs/method_y_status_summary.md")
    atomic_write_json(out_json, summary)

    lines = [
        "# Method Y PET-Assisted Branch — Design & Readiness Summary\n\n",
        f"- overall training readiness: **`{summary['status']}`**\n",
        f"- multimodal design: `{summary['design_status']}`\n",
        f"- PET inventory: `{summary['inventory_status']}`\n",
        f"- phase Y summary: `{summary['phase_y_status']}`\n",
        f"- training started: `{summary['training_started']}`\n",
        f"- PET preprocessing started: `{summary['pet_preprocessing_started']}`\n",
        f"- PET SPM/MNI progress: `{summary['pet_preprocess_progress']['resume_done']}/{summary['pet_preprocess_progress']['linked_work_items']}`\n",
        f"- method_x_claim_unchanged: `{summary['method_x_claim_unchanged']}`\n",
        f"- pet_as_method_x_input: `{summary['pet_as_method_x_input']}`\n",
        f"- current primary target: `{summary['current_primary_target']}`\n\n",
        "## Claim boundary\n\n",
        f"{summary['claim_boundary']}\n\n",
        "## Design availability (aggregate)\n\n",
        "| resource | count |\n",
        "|---|---:|\n",
        f"| locked catalog rows | {availability.get('locked_catalog_rows', 'NA')} |\n",
        f"| dev rows | {availability.get('dev_rows', 'NA')} |\n",
        f"| external rows | {availability.get('external_rows', 'NA')} |\n",
        f"| amyloid FBP image files | {availability.get('amyloid_fbp_image_files', 'NA')} |\n",
        f"| amyloid locked patient matches | {availability.get('amyloid_fbp_locked_patient_matches', 'NA')} |\n",
        f"| tau FTP image files | {availability.get('tau_ftp_image_files', 'NA')} |\n",
        f"| tau locked patient matches | {availability.get('tau_ftp_locked_patient_matches', 'NA')} |\n\n",
        "## Modality scenarios (design-ready)\n\n",
        "| scenario | PET input | Method X compliant | status |\n",
        "|---|---|---|---|\n",
    ]
    for scenario in summary["modality_scenarios"]:
        lines.append(
            f"| {scenario.get('name')} | {scenario.get('pet_input')} | {scenario.get('method_x_compliant')} | "
            f"{scenario.get('status')} |\n"
        )
    lines.extend(["\n", "## Training blockers\n\n"])
    for blocker in summary["blockers"]:
        gloss = summary["blocker_glossary"].get(blocker, "")
        lines.append(f"- **`{blocker}`** — {gloss}\n")
    lines.extend(["\n", "## Warnings\n\n"])
    for warning in summary["warnings"]:
        gloss = summary["warning_glossary"].get(warning, "")
        lines.append(f"- **`{warning}`** — {gloss}\n")
    lines.extend(["\n", "## Required before training\n\n"])
    for item in summary["required_before_training"]:
        lines.append(f"- {item}\n")
    lines.extend(["\n", "## Resolution checklist\n\n"])
    for item in summary["resolution_checklist"]:
        lines.append(f"- {item}\n")
    lines.extend(["\n", "## Execution gates (from design artifact)\n\n"])
    for gate in summary["execution_gates_before_training"]:
        lines.append(f"- {gate}\n")
    atomic_write_text(out_md, "".join(lines))

    StateManager().append_event(
        "method_y_status_summary",
        "COMPLETED",
        "Method Y design/readiness summary refreshed with blocker glossary.",
    )
    print(json.dumps({"method_y_status_summary": "COMPLETED", "status": summary["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
