#!/usr/bin/env python3
"""Map proposal claims to implemented, improved, added, or limited project artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def exists(path: str) -> bool:
    return Path(path).exists()


def build_claims() -> list[dict]:
    return [
        {
            "claim_id": "P01",
            "proposal_claim": "Early Alzheimer's disease diagnosis from MRI and clinical variables.",
            "status": "SCIENTIFIC_ETHICAL_LIMIT",
            "current_mapping": "Reframed as early prediction of amyloid biomarker status and preclinical risk unless an independent clinical label is used.",
            "evidence": ["docs/research_methods_limitations_ethics_statistics.md", "governance/clinical_endpoint_progression_policy.md"],
            "caveat": "AB-positive vs AB-negative is not clinical Alzheimer diagnosis.",
        },
        {
            "claim_id": "P02",
            "proposal_claim": "Non-invasive primary model.",
            "status": "BETTER_THAN_PROPOSAL",
            "current_mapping": "Method X-Strict protects PET-free inputs: T1 MRI plus age/sex/education/APOE4/MMSE.",
            "evidence": ["governance/method_x_operating_contract.md", "results/governance/method_x_strict_feature_matrix_summary.json"],
            "caveat": "PET is label source only, never Method X input.",
        },
        {
            "claim_id": "P03",
            "proposal_claim": "Multimodal MRI, PET, and cognitive modeling.",
            "status": "ADDED_CONTROLLED_EXTENSION" if exists("scripts/train_mri_pet_cognitive_method_y.py") else "DONE",
            "current_mapping": "Method Y includes PET-only, MRI-only paired, MRI+PET fusion, and MRI+PET+cognitive controlled extension.",
            "evidence": [
                "scripts/train_mri_pet_cognitive_method_y.py",
                "results/tables/method_y_incremental_value_table.json",
                "results/tables/method_y_multimodal_design.json",
            ],
            "caveat": "Method Y is paired subset only and PET-label tautology must be reported.",
        },
        {
            "claim_id": "P04",
            "proposal_claim": "3D ResNet / transfer learning.",
            "status": "ADDED_CONTROLLED_EXTENSION",
            "current_mapping": "MONAI ResNet/EfficientNet controlled ablations exist; true pretrained 3D fine-tuning is not claimed because no validated compatible pretrained 3D weights are locked in this repo.",
            "evidence": [
                "src/models/backbones/backbone_factory.py",
                "results/final_results/backbone_ablation_battery/backbone_ablation_battery.json",
                "results/final_results/three_d_ablation/three_d_ablation_summary.json",
            ],
            "caveat": "Report as controlled ablation, not confirmed pretrained transfer learning.",
        },
        {
            "claim_id": "P05",
            "proposal_claim": "Augmentation increases samples from 1500 to about 6000.",
            "status": "ADDED_CONTROLLED_EXTENSION",
            "current_mapping": "Train-fold-only MONAI augmentation reports 4x augmented training exposures while real N remains unchanged.",
            "evidence": ["src/data/volume_augmentation.py", "results/tables/augmentation_policy.json"],
            "caveat": "6000 is augmented exposure count, not 6000 real subjects.",
        },
        {
            "claim_id": "P06",
            "proposal_claim": "Synthetic MRI-to-PET augmentation.",
            "status": "ADDED_CONTROLLED_EXTENSION",
            "current_mapping": "Supplementary fold-local policy: generator fit only inside train fold; synthetic PET added only to same train fold.",
            "evidence": ["scripts/build_synthetic_mri_pet_supplement.py", "results/tables/synthetic_mri_pet_supplement.json"],
            "caveat": "Validation/test/holdout stay real; Method X never receives synthetic PET.",
        },
        {
            "claim_id": "P07",
            "proposal_claim": "Report accuracy, sensitivity, specificity, and F1.",
            "status": "ADDED_CONTROLLED_EXTENSION",
            "current_mapping": "Threshold metrics are computed using thresholds selected from fold-internal validation data only.",
            "evidence": ["src/evaluation/threshold_metrics.py", "results/tables/method_y_incremental_value_table.json"],
            "caveat": "Test labels are not used for threshold selection.",
        },
        {
            "claim_id": "P08",
            "proposal_claim": "Longitudinal clinical progression.",
            "status": "DONE",
            "current_mapping": "Clinical endpoint branch uses CDR worsening and PACC decline as progression proxies.",
            "evidence": [
                "governance/clinical_endpoint_progression_policy.md",
                "results/final_results/clinical_endpoint/clinical_endpoint_baseline_summary.json",
            ],
            "caveat": "Progression proxy, not clinical AD diagnosis.",
        },
        {
            "claim_id": "P09",
            "proposal_claim": "Privacy-preserving public outputs.",
            "status": "BETTER_THAN_PROPOSAL",
            "current_mapping": "Public bundle is aggregate/hash-level only with raw data, HDF5, images, raw identifiers, raw paths, and subject-level predictions excluded.",
            "evidence": ["scripts/build_public_bundle.py", "public_bundle/PUBLIC_BUNDLE_MANIFEST.json"],
            "caveat": "Subject-level prediction vectors remain private.",
        },
        {
            "claim_id": "P10",
            "proposal_claim": "Advanced fusion and scientific upgrades.",
            "status": "BETTER_THAN_PROPOSAL",
            "current_mapping": "Current project preserves SynIB, MFSCA, worst-dimension audit, FLAIR sensitivity, KD, cohort-probe, calibration, DCA, privacy, and claim-control workflows.",
            "evidence": [
                "docs/literature_backbone_evidence_2023_2026.yaml",
                "src/evaluation/worst_dimension_audit.py",
                "src/features/mfsca_texture.py",
                "scripts/run_pet_guided_kd_train.py",
            ],
            "caveat": "These are retained as current upgrades, not reverted to the initial proposal.",
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="results/tables/proposal_alignment_report.json")
    ap.add_argument("--out_md", default="results/tables/proposal_alignment_report.md")
    args = ap.parse_args()

    report = {
        "stage": "proposal_alignment_report",
        "status": "COMPLETED",
        "claim_boundary": {
            "preferred_title_framing": "Early prediction of amyloid biomarker status and preclinical risk",
            "clinical_alzheimer_diagnosis_requires_independent_label": True,
            "method_x_strict_pet_free": True,
            "method_y_pet_assisted_exploratory": True,
        },
        "privacy_contract": {
            "public_aggregate_or_hash_level_only": True,
            "public_raw_paths": False,
            "public_raw_identifiers": False,
            "public_images_or_caches": False,
            "public_subject_level_predictions": False,
        },
        "claims": build_claims(),
    }
    atomic_write_json(args.out_json, report)
    lines = [
        "# Proposal Alignment Report\n\n",
        f"- status: `{report['status']}`\n",
        f"- title framing: `{report['claim_boundary']['preferred_title_framing']}`\n",
        "- clinical Alzheimer diagnosis requires independent label: `True`\n",
        "- Method X-Strict PET-free: `True`\n\n",
        "| claim | status | current mapping | caveat |\n",
        "|---|---|---|---|\n",
    ]
    for claim in report["claims"]:
        lines.append(
            f"| {claim['claim_id']} | {claim['status']} | {claim['current_mapping']} | {claim['caveat']} |\n"
        )
    atomic_write_text(args.out_md, "".join(lines))
    write_run_manifest("results/logs/proposal_alignment_report_manifest.json", stage="proposal_alignment_report", data_root=None, extra={"claims": len(report["claims"])})
    StateManager().append_event("proposal_alignment_report", "COMPLETED", "Proposal claims mapped to current implementation and controlled extensions.")
    print(json.dumps({"status": report["status"], "claims": len(report["claims"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
