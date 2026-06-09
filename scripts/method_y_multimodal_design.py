from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.pet_preprocess_progress import summarize_pet_preprocess_progress
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


MODALITY_SCENARIOS = [
    {
        "name": "mri_only",
        "inputs": ["T1 MRI"],
        "status": "Method X baseline/final-comparison path",
        "method_x_compliant": True,
        "pet_input": False,
    },
    {
        "name": "pet_only",
        "inputs": ["amyloid PET image", "optional tau PET image"],
        "status": "Method Y only",
        "method_x_compliant": False,
        "pet_input": True,
    },
    {
        "name": "mri_pet",
        "inputs": ["T1 MRI", "amyloid PET image", "optional tau PET image"],
        "status": "Method Y multimodal only",
        "method_x_compliant": False,
        "pet_input": True,
    },
    {
        "name": "mri_pet_cognitive",
        "inputs": ["T1 MRI", "PET image", "cognitive/psychological summaries"],
        "status": "Method Y multimodal+cognitive only",
        "method_x_compliant": False,
        "pet_input": True,
    },
]


def load_optional_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def nested(obj: dict | None, *keys: str) -> object:
    cur: object = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="results/tables/method_y_pet_inventory.json")
    ap.add_argument("--cognitive_summary", default="results/tables/cognitive_feature_summary.json")
    ap.add_argument("--split_summary", default="splits/split_summary.json")
    ap.add_argument("--manifest_summary", default="results/tables/method_y_pet_manifest_summary.json")
    args = ap.parse_args()

    inventory = load_optional_json(args.inventory)
    cognitive = load_optional_json(args.cognitive_summary)
    splits = load_optional_json(args.split_summary)
    manifest_summary = load_optional_json(args.manifest_summary)
    linked_work_items = int(
        manifest_summary.get("linked_work_items")
        or nested(manifest_summary, "link_summary", "linked_work_items")
        or 0
    )
    preprocess_progress = summarize_pet_preprocess_progress(linked_work_items=linked_work_items)
    pet_preprocessing_started = bool(preprocess_progress["preprocessing_started"])

    amyloid_files = nested(inventory, "pet_image_inventory", "amyloid_fbp", "image_file_count")
    tau_files = nested(inventory, "pet_image_inventory", "tau_ftp", "image_file_count")
    amyloid_matches = nested(inventory, "pet_image_inventory", "amyloid_fbp", "locked_catalog_patient_matches_in_memory")
    tau_matches = nested(inventory, "pet_image_inventory", "tau_ftp", "locked_catalog_patient_matches_in_memory")
    cognitive_rows = nested(cognitive, "rows")

    design = {
        "schema_version": 1,
        "method": "Method Y-PET/MM",
        "status": "DESIGN_READY_NOT_TRAINED",
        "training_started": False,
        "pet_preprocessing_started": pet_preprocessing_started,
        "pet_preprocess_progress": preprocess_progress,
        "method_x_claim_unchanged": True,
        "method_x_pet_input_allowed": False,
        "method_y_pet_input_allowed": True,
        "modality_scenarios": MODALITY_SCENARIOS,
        "target_policy": {
            "current_primary_target": "AB-positive vs AB-negative biomarker status",
            "current_target_is_clinical_alzheimer_diagnosis": False,
            "clinical_alzheimer_output_allowed_now": False,
            "clinical_alzheimer_requires_independent_label": True,
            "allowed_future_clinical_targets": [
                "cognitively_normal_vs_mci_vs_ad_dementia",
                "clinical_diagnosis_at_visit",
                "conversion_to_mci_or_ad_dementia_on_followup",
                "adjudicated_clinical_diagnosis",
            ],
            "forbidden_interpretation": "Do not call AB-positive vs AB-negative a definitive Alzheimer disease diagnosis.",
        },
        "availability_summary": {
            "locked_catalog_rows": nested(inventory, "locked_catalog", "rows"),
            "locked_catalog_patients": nested(inventory, "locked_catalog", "patients"),
            "amyloid_fbp_image_files": amyloid_files,
            "amyloid_fbp_locked_patient_matches": amyloid_matches,
            "tau_ftp_image_files": tau_files,
            "tau_ftp_locked_patient_matches": tau_matches,
            "cognitive_feature_rows": cognitive_rows,
            "dev_rows": nested(splits, "dev_rows"),
            "external_rows": nested(splits, "external_rows"),
        },
        "execution_gates_before_training": [
            "Create separate Method Y PET preprocessing root outside Method X caches.",
            "Create no-PET-in-Method-X regression test after adding PET code.",
            "Confirm PET images are never loaded by Method X or Method X-Cog dataloaders.",
            "Define whether the target is biomarker abnormality or an independent clinical diagnosis.",
            "If clinical AD output is desired, build and validate a separate clinical-label table before training.",
            "Keep public outputs aggregate-only unless a separate privacy review approves hash-level artifacts.",
        ],
        "reporting_contract": {
            "mri_only": "Report under Method X when PET is not an input.",
            "pet_only": "Report under Method Y only.",
            "mri_pet": "Report under Method Y only; not a non-invasive Method X result.",
            "mri_pet_cognitive": "Report under Method Y only and label cognitive features as exploratory covariates.",
        },
        "privacy_contract": {
            "raw_identifiers_exported": False,
            "raw_paths_exported": False,
            "subject_level_predictions_public": False,
            "pet_images_public": False,
            "hdf5_cache_public": False,
        },
    }

    atomic_write_json("results/tables/method_y_multimodal_design.json", design)
    scenario_lines = []
    for item in MODALITY_SCENARIOS:
        scenario_lines.append(
            f"| {item['name']} | {', '.join(item['inputs'])} | {item['status']} | {item['method_x_compliant']} | {item['pet_input']} |\n"
        )
    gate_lines = [f"- {gate}\n" for gate in design["execution_gates_before_training"]]
    md = [
        "# Method Y PET/MRI Multimodal Design\n\n",
        "- status: `DESIGN_READY_NOT_TRAINED`\n",
        "- training_started: `False`\n",
        f"- PET preprocessing started: `{pet_preprocessing_started}`\n",
        f"- PET SPM/MNI preprocess: `{preprocess_progress['resume_done']}/{preprocess_progress['linked_work_items']}` complete\n",
        "- Method X claim unchanged: `True`\n",
        "- PET input allowed in Method X: `False`\n",
        "- PET input allowed only in Method Y: `True`\n\n",
        "## Availability\n\n",
        f"- locked catalog rows: `{design['availability_summary']['locked_catalog_rows']}`\n",
        f"- amyloid FBP image files: `{amyloid_files}`\n",
        f"- amyloid FBP locked-patient matches: `{amyloid_matches}`\n",
        f"- tau FTP image files: `{tau_files}`\n",
        f"- tau FTP locked-patient matches: `{tau_matches}`\n",
        f"- cognitive feature rows: `{cognitive_rows}`\n\n",
        "## Modality Scenarios\n\n",
        "| scenario | inputs | reporting status | Method X compliant | PET input |\n",
        "|---|---|---|---:|---:|\n",
        *scenario_lines,
        "\n## Target Policy\n\n",
        "Current output should be named `AB-positive vs AB-negative biomarker status`, not Alzheimer disease diagnosis. A clinical Alzheimer output is blocked until an independent clinical diagnosis/conversion label is defined and validated.\n\n",
        "## Gates Before Training\n\n",
        *gate_lines,
        "\n## Reporting Boundary\n\n",
        "PET-only, MRI+PET, and MRI+PET+cognitive are Method Y results. They must not be merged into the Method X non-invasive claim.\n",
    ]
    atomic_write_text("results/tables/method_y_multimodal_design.md", "".join(md))
    write_run_manifest("results/logs/method_y_multimodal_design_manifest.json", stage="method_y_multimodal_design", data_root=None, extra={"status": design["status"]})
    StateManager().append_event("method_y_multimodal_design", "DESIGN_READY_NOT_TRAINED", "Method Y PET/MRI multimodal design artifact generated.")
    print(json.dumps({"status": design["status"], "training_started": False, "pet_preprocessing_started": pet_preprocessing_started, "pet_preprocess_progress": preprocess_progress}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
