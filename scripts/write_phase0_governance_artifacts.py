from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/policy/final_execution_program.json")
DEFAULT_OUT = Path("results/governance")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_label_freeze(config: dict[str, Any]) -> dict[str, Any]:
    label = config["label_freeze"]
    return {
        "schema_version": 1,
        "status": "TEMPLATE_REQUIRES_CUTOFF_BEFORE_MODELING",
        "primary_binary_label": label["primary_binary_label"],
        "centiloid_threshold": label["centiloid_threshold"],
        "sensitivity_label": label["sensitivity_label"],
        "required_fields": label["required_fields"],
        "change_policy": label["change_policy"],
        "must_freeze_before": [
            "modeling",
            "model_selection",
            "hyperparameter_tuning",
            "performance_inspection",
            "calibration",
            "threshold_selection",
        ],
    }


def build_feature_allowlists(config: dict[str, Any]) -> dict[str, Any]:
    branches = config["branches"]
    return {
        "schema_version": 1,
        "status": "POLICY_ALLOWLISTS_DEFINED",
        "method_x_allowed_inputs": config["method_x_allowed_inputs"],
        "method_x_forbidden_inputs": config["method_x_forbidden_inputs"],
        "branches": {
            name: {
                "purpose": branch.get("purpose"),
                "allowed_inputs": branch.get("allowed_inputs", []),
                "forbidden_inputs": branch.get("forbidden_inputs", []),
                "claim_boundary": branch.get("claim_boundary"),
            }
            for name, branch in branches.items()
        },
        "quarantined_predictor_families": [
            "amyloid_eligibility_variables",
            "visual_read_fields_used_to_define_labels",
            "pet_derived_predictors",
            "centiloid_as_predictor",
            "suvr_as_predictor",
            "tau_pet_as_predictor",
            "future_visits",
            "raw_ids",
            "raw_paths",
            "scanner_site_ids_for_method_x",
        ],
    }


def build_cohort_probe_protocol(config: dict[str, Any]) -> dict[str, Any]:
    protocol = config["cohort_probe_protocol"]
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_MODELING",
        "headline_validity_metric": protocol["headline_validity_metric"],
        "primary_probe": protocol["primary_probe"],
        "secondary_probe": protocol["secondary_probe"],
        "always_report": protocol["always_report"],
        "interpretation": {
            "high_linear_auc": "Cohort is linearly encoded.",
            "low_linear_high_nonlinear_auc": "Cohort is encoded nonlinearly; report as covert/structured cohort signal.",
            "deconfounding_success_rule": "Never claim deconfounding succeeded based on the linear probe alone.",
        },
    }


def build_deconfounding_scope(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_MODELING",
        "valid_settings": [
            {
                "setting": "continuous_centiloid_target",
                "reason": "Target varies within cohort and is not equivalent to substudy membership.",
            },
            {
                "setting": "sf_inclusive_screening",
                "reason": "SF-inclusive screening contains both biomarker classes where label does not equal cohort.",
            },
        ],
        "degenerate_settings": [
            {
                "setting": "binary_a4_learn_only_catalog",
                "reason": "Label approximately equals cohort; removing cohort approximately removes label.",
                "confound_controlled_cv": "N/A-DEGENERATE",
            }
        ],
        "forbidden_phrase": "confound-free prediction accuracy",
        "preferred_phrases": [
            "confound-controlled performance estimate",
            "confound-reduced performance estimate",
            "confound-aware evaluation",
        ],
        "method_x_strict_cv_status": config["branches"]["method_x_strict"]["confound_controlled_cv"],
    }


def build_manifest(config: dict[str, Any], artifact_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "PHASE0_GOVERNANCE_ARTIFACTS_WRITTEN",
        "program_version": config["version"],
        "protected_thesis_object": config["protected_thesis_object"],
        "minimum_phase2_mvp": config["phase2_minimum_mvp"],
        "optional_supplementary": config["optional_supplementary"],
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "raw_data_touched": False,
        "subject_level_outputs_exported": False,
    }


def build_manifest_md(manifest: dict[str, Any]) -> str:
    lines = [
        "# Phase 0 Governance Manifest\n\n",
        f"- status: `{manifest['status']}`\n",
        f"- protected_thesis_object: `{manifest['protected_thesis_object']}`\n",
        "- raw_data_touched: `False`\n",
        "- subject_level_outputs_exported: `False`\n",
        "\n## Artifacts\n\n",
    ]
    for name, path in manifest["artifacts"].items():
        lines.append(f"- {name}: `{path}`\n")
    lines.extend(["\n## Minimum Phase-2 MVP\n\n"])
    for item in manifest["minimum_phase2_mvp"]:
        lines.append(f"- {item}\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    config = read_json(config_path)

    artifacts = {
        "label_freeze_template": out_dir / "label_freeze_template.json",
        "feature_allowlist_summary": out_dir / "feature_allowlist_summary.json",
        "cohort_probe_protocol": out_dir / "cohort_probe_protocol.json",
        "deconfounding_scope": out_dir / "deconfounding_scope.json",
        "sf_internal_decoupling_policy": Path("governance/sf_internal_decoupling_policy.md"),
        "branch_catalog_manifest": Path("data/processed/catalogs/branch_catalog_manifest.json"),
        "cohort_label_coupling_audit": out_dir / "cohort_label_coupling_audit.json",
        "leakage_review_report": out_dir / "leakage_review_report.json",
        "phase0_governance_manifest": out_dir / "phase0_governance_manifest.json",
        "phase0_governance_manifest_md": out_dir / "phase0_governance_manifest.md",
    }

    write_json(artifacts["label_freeze_template"], build_label_freeze(config))
    write_json(artifacts["feature_allowlist_summary"], build_feature_allowlists(config))
    write_json(artifacts["cohort_probe_protocol"], build_cohort_probe_protocol(config))
    write_json(artifacts["deconfounding_scope"], build_deconfounding_scope(config))
    manifest = build_manifest(config, artifacts)
    write_json(artifacts["phase0_governance_manifest"], manifest)
    write_text(artifacts["phase0_governance_manifest_md"], build_manifest_md(manifest))

    print(json.dumps({"status": manifest["status"], "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
