"""Fail-closed, privacy-safe synthesis of locked Method Y evidence.

This module never loads PET volumes, checkpoints, or outer-test datasets. It
does not promote a completed training run into a scientific/clinical claim.
The original v3/v4 artifacts remain immutable; this is a versioned correction.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes, atomic_write_json
from .contracts import sha256_mapping
from .posttraining_correction_v4_2 import BRANCHES, CorrectionV42Error


RECEIPTS = {
    "v4_correction": "locked_artifact_correction.json",
    "v4_policy_bootstrap": "fold_stratified_paired_bootstrap.json",
    "v4_scanner": "fold_local_scanner_probes.json",
    "selection_flow": "image_cohort_selection_flow.json",
    "v3_incremental": "v3_scalar_vs_image_fold_stratified_bootstrap.json",
}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise CorrectionV42Error(reason)


def _read_object(path: Path) -> tuple[dict[str, Any], str]:
    _require(path.is_file(), f"SCIENCE_SOURCE_MISSING:{path.name}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CorrectionV42Error(f"SCIENCE_SOURCE_NOT_JSON:{path.name}") from exc
    _require(isinstance(value, dict), f"SCIENCE_SOURCE_NOT_OBJECT:{path.name}")
    return value, hashlib.sha256(raw).hexdigest()


def _verify_receipt(value: dict[str, Any], label: str) -> None:
    expected = sha256_mapping({k: v for k, v in value.items() if k != "report_sha256"})
    _require(value.get("report_sha256") == expected, f"SCIENCE_RECEIPT_HASH_MISMATCH:{label}")
    _require(value.get("science_gate_status") == "BLOCKED", f"UNEXPECTED_SCIENCE_PASS:{label}")
    _require(value.get("claim_status") == "POST_HOC_INTERNAL_EXPLORATORY",
             f"UNEXPECTED_CLAIM:{label}")


def _checked_ci(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 2,
             f"SCIENCE_CI_INVALID:{label}")
    low, high = float(value[0]), float(value[1])
    _require(math.isfinite(low) and math.isfinite(high) and low <= high,
             f"SCIENCE_CI_INVALID:{label}")
    return [low, high]


def _delta_summary(value: dict[str, Any], label: str) -> dict[str, Any]:
    _require(value.get("status") == "COMPLETED_CONDITIONAL_EXPLORATORY",
             f"SCIENCE_BOOTSTRAP_INCOMPLETE:{label}")
    _require(value.get("valid_resamples") >= 1000, f"SCIENCE_BOOTSTRAP_TOO_SMALL:{label}")
    _require(len(value.get("fold_sizes_preserved", [])) == 5,
             f"SCIENCE_FOLD_COVERAGE_INVALID:{label}")
    point_ibs = float(value["point_delta_ibs"])
    point_uno = float(value["point_delta_uno"])
    _require(math.isfinite(point_ibs) and math.isfinite(point_uno),
             f"SCIENCE_POINT_NONFINITE:{label}")
    return {
        "delta_ibs": point_ibs,
        "delta_ibs_95ci": _checked_ci(value.get("delta_ibs_95ci"), label),
        "delta_uno": point_uno,
        "delta_uno_95ci": _checked_ci(value.get("delta_uno_95ci"), label),
        "uncertainty_scope": "CONDITIONAL_FIXED_MODELS_POST_HOC_NOT_CONFIRMATORY",
    }


def synthesize(
    receipts: dict[str, dict[str, Any]],
    receipt_file_sha256: dict[str, str],
    *, quantitative: dict[str, Any],
    xai: dict[str, Any],
    external: dict[str, Any],
    v3_claim: dict[str, Any],
    scanner_support: dict[str, dict[str, Any]],
    risk_sensitivity: dict[str, Any],
    pairwise: dict[str, Any],
    joint_pairs: dict[str, Any],
    external_sha256: dict[str, str],
) -> dict[str, Any]:
    """Reduce verified evidence to an explicit technical/scientific boundary."""
    _require(set(receipts) == set(RECEIPTS), "SCIENCE_RECEIPT_COVERAGE_INVALID")
    for label, value in receipts.items():
        _verify_receipt(value, label)
    correction = receipts["v4_correction"]
    v4_boot = receipts["v4_policy_bootstrap"]
    scanner = receipts["v4_scanner"]
    selection = receipts["selection_flow"]
    v3 = receipts["v3_incremental"]
    parent_sha = receipt_file_sha256["v4_correction"]
    for label in ("v4_policy_bootstrap", "v4_scanner", "selection_flow"):
        _require(receipts[label].get("correction_receipt_sha256") == parent_sha,
                 f"SCIENCE_PARENT_RECEIPT_MISMATCH:{label}")
    _require(pairwise.get("report_sha256") == sha256_mapping(
        {k: v for k, v in pairwise.items() if k != "report_sha256"}
    ), "SCIENCE_PAIRWISE_RECEIPT_HASH_MISMATCH")
    _require(pairwise.get("correction_receipt_sha256") == parent_sha
             and pairwise.get("source_run_contract_sha256")
             == correction.get("source_run_contract_sha256"),
             "SCIENCE_PAIRWISE_PARENT_MISMATCH")
    _require(pairwise.get("status") == "COMPLETED_DESCRIPTIVE_ONLY"
             and pairwise.get("science_gate_status") == "BLOCKED"
             and pairwise.get("positive_claim_allowed") is False
             and pairwise.get("planned_pair_count_per_branch") == 5,
             "SCIENCE_PAIRWISE_STATUS_INVALID")
    _require(set(pairwise.get("branch_results", {})) == set(BRANCHES),
             "SCIENCE_PAIRWISE_BRANCH_COVERAGE_INVALID")
    _require(all(len(pairwise["branch_results"][branch]["planned_pairs"]) == 5
                 for branch in BRANCHES), "SCIENCE_PAIRWISE_PAIR_COVERAGE_INVALID")
    _require(joint_pairs.get("report_sha256") == sha256_mapping(
        {k: v for k, v in joint_pairs.items() if k != "report_sha256"}
    ), "SCIENCE_JOINT_RECEIPT_HASH_MISMATCH")
    _require(joint_pairs.get("correction_receipt_sha256") == parent_sha
             and joint_pairs.get("pairwise_receipt_sha256")
             == external_sha256["pairwise_v4_2"]
             and joint_pairs.get("source_run_contract_sha256")
             == correction.get("source_run_contract_sha256"),
             "SCIENCE_JOINT_PARENT_MISMATCH")
    _require(joint_pairs.get("input_sha256") == pairwise.get("input_sha256"),
             "SCIENCE_JOINT_SOURCE_COVERAGE_MISMATCH")
    _require(joint_pairs.get("status") == "COMPLETED_CONDITIONAL_EXPLORATORY"
             and joint_pairs.get("science_gate_status") == "BLOCKED"
             and joint_pairs.get("claim_status") == "POST_HOC_INTERNAL_EXPLORATORY"
             and joint_pairs.get("positive_claim_allowed") is False
             and joint_pairs.get("external_claim_allowed") is False
             and joint_pairs.get("iterations_per_branch", 0) >= 1000
             and joint_pairs.get("multiplicity")
             == "TEN_PLANNED_CONTRASTS_TWO_METRICS_NO_ADJUSTED_CLAIM_TEST",
             "SCIENCE_JOINT_STATUS_INVALID")
    _require(set(joint_pairs.get("branch_results", {})) == set(BRANCHES),
             "SCIENCE_JOINT_BRANCH_COVERAGE_INVALID")
    _require(correction.get("status") == "COMPLETED_FROM_HASH_VERIFIED_LOCKED_ARTIFACTS",
             "SCIENCE_V4_CORRECTION_INCOMPLETE")
    _require(v3.get("status") == "COMPLETED_CONDITIONAL_EXPLORATORY",
             "SCIENCE_V3_INCREMENTAL_INCOMPLETE")
    _require(all(correction.get(k) is False for k in ("positive_claim_allowed", "external_claim_allowed")),
             "SCIENCE_V4_CLAIM_POLICY_INVALID")
    _require(all(v3.get(k) is False for k in ("positive_claim_allowed", "external_claim_allowed")),
             "SCIENCE_V3_CLAIM_POLICY_INVALID")
    _require(
        correction.get("source_run_contract_sha256")
        == v4_boot.get("source_run_contract_sha256")
        == scanner.get("source_run_contract_sha256"),
        "SCIENCE_V4_CONTRACT_MISMATCH",
    )
    _require(scanner.get("leave_manufacturer_out_performance_status") == "NOT_EXECUTED",
             "SCIENCE_SCANNER_STATUS_UNEXPECTED")
    _require(scanner.get("pooled_oof_embedding_probe_status")
             == "INVALID_CROSS_MODEL_COORDINATE_POOLING", "SCIENCE_SCANNER_PROBE_POLICY_INVALID")
    _require(quantitative.get("science_gate_status") == "BLOCKED"
             and quantitative.get("claim_status") == "NO_CLAIM", "SCIENCE_QUANTITATIVE_STATUS_UNEXPECTED")
    _require(quantitative.get("centaur_reference", {}).get("status") == "PASS"
             and quantitative.get("centiloid_masks", {}).get("status") == "PASS",
             "SCIENCE_ASSET_STATUS_UNEXPECTED")
    _require(quantitative.get("tau_quantitative_validation", {}).get("report_audit", {}).get("status")
             == "BLOCKED", "SCIENCE_TAU_QUANTITATIVE_STATUS_UNEXPECTED")
    _require(xai.get("science_gate_status") == "FAIL"
             and xai.get("claim_status") == "NO_CLAIM", "SCIENCE_XAI_STATUS_UNEXPECTED")
    fidelity = xai.get("details", {}).get("fidelity", {})
    _require(fidelity.get("evaluated_cells") == 45
             and fidelity.get("expected_cells") == 50
             and fidelity.get("nonpass_cells") == 30,
             "SCIENCE_XAI_COVERAGE_UNEXPECTED")
    _require(external.get("status") == "EXTERNAL_PENDING"
             and external.get("science_gate_status") == "BLOCKED"
             and external.get("external_outcome_accessed") is False,
             "SCIENCE_EXTERNAL_STATUS_UNEXPECTED")
    _require(v3_claim.get("external_status") == "EXTERNAL_PENDING"
             and v3_claim.get("positive_claim_allowed") is False
             and v3_claim.get("external_claim_allowed") is False,
             "SCIENCE_V3_CLAIM_SOURCE_INVALID")
    _require(set(v3_claim.get("per_branch", {})) == set(BRANCHES),
             "SCIENCE_V3_BRANCH_CLAIM_COVERAGE_INVALID")
    _require(all(not v3_claim["per_branch"][branch].get("positive_claim_allowed")
                 for branch in BRANCHES), "SCIENCE_V3_BRANCH_CLAIM_INVALID")
    _require(set(scanner_support) == set(BRANCHES),
             "SCIENCE_SCANNER_SUPPORT_COVERAGE_INVALID")
    _require(risk_sensitivity.get("status") == "PASS"
             and risk_sensitivity.get("primary") == "logit_sum"
             and risk_sensitivity.get("sensitivity") == "mortality_last",
             "SCIENCE_RISK_SENSITIVITY_SOURCE_INVALID")
    sensitivity_rho = float(risk_sensitivity.get("spearman", {}).get("rho", float("nan")))
    _require(math.isfinite(sensitivity_rho) and -1 <= sensitivity_rho < 0,
             "SCIENCE_RISK_TARGET_AGREEMENT_UNEXPECTED")
    _require(int(risk_sensitivity["spearman"].get("n", 0)) > 100000,
             "SCIENCE_RISK_SENSITIVITY_VOXEL_COUNT_UNEXPECTED")

    branch_results: dict[str, Any] = {}
    for branch in BRANCHES:
        flow = selection["branches"][branch]
        parent = int(flow["endpoint_parent_n"])
        included = int(flow["image_included_n"])
        excluded = int(flow["image_excluded_n"])
        _require(parent == included + excluded,
                 f"SCIENCE_SELECTION_ACCOUNTING_INVALID:{branch}")
        _require(included == int(correction["branches"][branch]["subject_count"])
                 == int(v3["branches"][branch]["subject_count"]),
                 f"SCIENCE_BRANCH_MEMBERSHIP_MISMATCH:{branch}")
        nested_folds = correction["branches"][branch]["folds"]
        _require(len(nested_folds) == 5 and {r["outer_fold"] for r in nested_folds} == set(range(5)),
                 f"SCIENCE_NESTED_FOLD_COVERAGE_INVALID:{branch}")
        _require(sum(int(r["outer_test_subject_count"]) for r in nested_folds) == included,
                 f"SCIENCE_NESTED_FOLD_SIZE_INVALID:{branch}")
        _require(scanner["results"][branch]["completed_probes"] >= 1,
                 f"SCIENCE_SCANNER_PROBES_MISSING:{branch}")
        scanner_old = scanner_support[branch]
        _require(scanner_old.get("status") == "COMPLETED_WITH_CLAIM_GATE_UNRESOLVED"
                 and scanner_old.get("scanner_robustness_passed") is False,
                 f"SCIENCE_SCANNER_GATE_UNEXPECTED:{branch}")
        support = scanner_old.get("manufacturer_support", {})
        _require(isinstance(support, dict) and
                 sum(int(row["subjects"]) for row in support.values()) == included,
                 f"SCIENCE_SCANNER_SUPPORT_SUM_MISMATCH:{branch}")
        eligible = sorted(
            name for name, row in support.items()
            if name != "UNKNOWN" and int(row["subjects"]) >= 30
            and int(row["events"]) >= 10
        )
        _require(eligible == sorted(scanner_old["leave_manufacturer_out_eligibility"]
                 ["eligible_manufacturers"]),
                 f"SCIENCE_SCANNER_ELIGIBILITY_MISMATCH:{branch}")
        _require(len(eligible) == (3 if branch == "amyloid" else 1),
                 f"SCIENCE_SCANNER_SUPPORT_UNEXPECTED:{branch}")
        v3_branch = v3["branches"][branch]
        joint_branch = joint_pairs["branch_results"][branch]
        original_grid = joint_branch.get("original_evaluation_grid_days", [])
        stable_grid = joint_branch.get("support_stable_evaluation_grid_days", [])
        _require(isinstance(original_grid, list) and len(original_grid) >= 2
                 and isinstance(stable_grid, list) and len(stable_grid) >= 2
                 and stable_grid == original_grid[-len(stable_grid):]
                 and joint_branch.get("ibs_estimand") in (
                     "SOURCE_GRID_REPRODUCED",
                     "RESTRICTED_GRID_POST_HOC_SENSITIVITY_NOT_EQUIVALENT_TO_SOURCE_IBS",
                 ), f"SCIENCE_JOINT_GRID_INVALID:{branch}")
        _require(joint_branch.get("status") == "COMPLETED_CONDITIONAL_EXPLORATORY"
                 and joint_branch.get("shared_subject_draws_across_arms") is True
                 and joint_branch.get("valid_resamples")
                 == joint_pairs["iterations_per_branch"]
                 and joint_branch.get("rejected_attempts", 0)
                 <= joint_branch["valid_resamples"] / 19,
                 f"SCIENCE_JOINT_BOOTSTRAP_INVALID:{branch}")
        expected_pairs = set(pairwise["branch_results"][branch]["planned_pairs"])
        actual_pairs = joint_branch.get("planned_contrasts", {})
        _require(set(actual_pairs) == expected_pairs and len(actual_pairs) == 5,
                 f"SCIENCE_JOINT_PAIR_COVERAGE_INVALID:{branch}")
        joint_contrasts: dict[str, Any] = {}
        for name, result in actual_pairs.items():
            ibs = float(result["point_delta_ibs"])
            uno = float(result["point_delta_uno"])
            _require(math.isfinite(ibs) and math.isfinite(uno)
                     and result.get("status") == "CONDITIONAL_EXPLORATORY_ONLY"
                     and result.get("p_value") == "NOT_COMPUTED"
                     and result.get("familywise_claim") == "NOT_EVALUATED",
                     f"SCIENCE_JOINT_CONTRAST_INVALID:{branch}/{name}")
            joint_contrasts[name] = {
                "delta_ibs": ibs,
                "delta_ibs_95ci": _checked_ci(result.get("delta_ibs_95ci"), branch + name),
                "delta_uno": uno,
                "delta_uno_95ci": _checked_ci(result.get("delta_uno_95ci"), branch + name),
            }
        branch_results[branch] = {
            "endpoint_parent_n": parent,
            "image_evaluable_n": included,
            "image_excluded_n": excluded,
            "v3_image_increment_over_scalar": _delta_summary(
                v3_branch["scalar_vs_image"], f"v3/{branch}"
            ),
            "v3_ibs_estimand": v3_branch["ibs_estimand"],
            "v4_fold_local_policy_arms": [r["selected_arm"] for r in nested_folds],
            "v4_policy_increment_over_a0": _delta_summary(
                v4_boot["branches"][branch], f"v4/{branch}"
            ),
            "v4_joint_pairwise_conditional": {
                "status": "POST_HOC_CONDITIONAL_EXPLORATORY_ONLY",
                "ibs_estimand": joint_branch["ibs_estimand"],
                "evaluation_grid_days": joint_branch["support_stable_evaluation_grid_days"],
                "valid_resamples": joint_branch["valid_resamples"],
                "rejected_attempts": joint_branch["rejected_attempts"],
                "planned_contrasts": joint_contrasts,
                "multiplicity_adjusted_claim": False,
            },
            "scanner_probe_scope": "WITHIN_ONE_FOLD_MODEL_ONLY_DIAGNOSTIC",
            "manufacturer_subject_event_support": {
                name: {"subjects": int(row["subjects"]), "events": int(row["events"])}
                for name, row in sorted(support.items())
            },
            "manufacturer_eligibility_count": len(eligible),
            "leave_manufacturer_out_status": "NOT_EXECUTED",
            "positive_claim_allowed": False,
        }
    _require(correction["legacy_post_eval_corrections"]["calibration_leg"]
             == "COMPLETED_WITH_ESTIMABILITY_GATES", "SCIENCE_CALIBRATION_STATUS_INVALID")
    return {
        "schema_version": "4.2",
        "status": "LOCKED_ARTIFACT_REANALYSIS_COMPLETE_SCIENCE_BLOCKED",
        "technical_training_status": "V3_20_OF_20_AND_V4_50_OF_50_FINALIZED",
        "statistical_reanalysis_status": "COMPLETE_FOR_V3_INCREMENTAL_V4_FOLD_LOCAL_POLICY_AND_JOINT_PAIRS",
        "scientific_validation_status": "BLOCKED",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "quantitative_pet_claim_allowed": False,
        "xai_claim_allowed": False,
        "branch_results": branch_results,
        "calibration": {
            "status": "COMPLETED_WITH_ESTIMABILITY_GATES",
            "limitation": "EXISTING_HESSIAN_INTERVALS_DO_NOT_ACCOUNT_FOR_FOLD_CLUSTERING",
        },
        "legacy_v4_post_eval": {
            "pooled_branch_friedman": "INVALID_FOR_CLAIM",
            "raw_risk_score_noninferiority": "INVALID_FOR_CLAIM",
            "legacy_pairwise_ci": "SUPERSEDED_BY_JOINT_FOLD_STRATIFIED_CONDITIONAL_INTERVALS",
            "global_selected_arm_oof": "SELECTION_INFORMED_DESCRIPTIVE_ONLY",
        },
        "branchwise_matched_pairs": {
            "status": "TEN_CONDITIONAL_EXPLORATORY_JOINT_PAIR_INTERVALS_VERIFIED",
            "inferential_claim": "NONE",
        },
        "quantitative_pet": {
            "evidence_present": True,
            "official_atlas_assets": "PASS_FOR_ASSET_INTEGRITY_NOT_IMAGE_EQUIVALENCE",
            "tau_scan_level_quantitation": "BLOCKED",
            "fbp_ftp_pattern_scale": "NOT_EQUIVALENT_TO_OFFICIAL_CENTILOID_OR_CENTAUR",
        },
        "xai": {
            "science_gate_status": "FAIL",
            "evaluated_cells": 45,
            "expected_cells": 50,
            "nonpass_cells": 30,
            "risk_target_sensitivity_rho_descriptive": sensitivity_rho,
            "legacy_sensitivity_pass_meaning": "COMPUTATION_COMPLETED_NOT_AGREEMENT_PASS",
            "voxel_bootstrap_ci_status": "INVALID_FOR_SUBJECT_LEVEL_INFERENCE_SPATIAL_PSEUDOREPLICATION",
        },
        "external_validation": "EXTERNAL_PENDING",
        "remaining_gates": [
            "SCANNER_LEAVE_MANUFACTURER_OUT_PERFORMANCE_NOT_EXECUTED",
            "TAU_SECOND_MANUFACTURER_EVENT_SUPPORT_INSUFFICIENT",
            "PET_IMAGE_QUANTITATIVE_EQUIVALENCE_UNPROVEN",
            "XAI_FIDELITY_AND_COVERAGE_FAILED",
            "XAI_RISK_TARGET_SENSITIVITY_DISCORDANT",
            "EXTERNAL_COHORT_ACCESS_AND_VALIDATION_PENDING",
            "SELECTION_AND_REFIT_UNCERTAINTY_NOT_COVERED_BY_LOCKED_MODEL_BOOTSTRAP",
        ],
        "source_sha256": {
            **{label: receipt_file_sha256[label] for label in sorted(RECEIPTS)},
            **external_sha256,
        },
    }


def build_status(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).resolve()
    base = project / "results" / "method_y_revival_v4_posttraining_correction_v4_2"
    receipts: dict[str, dict[str, Any]] = {}
    receipt_sha: dict[str, str] = {}
    for label, filename in RECEIPTS.items():
        receipts[label], receipt_sha[label] = _read_object(base / filename)
    external_paths = {
        "quantitative_v2": project / "results" / "method_y_revival_v2" / "provenance"
        / "quantitative_pet_feature_validation_v2.json",
        "xai_v4": project / "results" / "method_y_revival_v4_image_factorial_v1"
        / "attribution" / "xai_science_gate_v4.json",
        "external_v2": project / "results" / "method_y_revival_v2" / "external_validation"
        / "readiness.json",
        "claim_v3": project / "results" / "method_y_revival_v3_image_v2" / "reports"
        / "IMAGE_CLAIM_POLICY.json",
        "scanner_v3_amyloid": project / "results" / "method_y_revival_v3_image_v2" / "reports"
        / "amyloid_scanner_robustness.json",
        "scanner_v3_tau": project / "results" / "method_y_revival_v3_image_v2" / "reports"
        / "tau_scanner_robustness.json",
        "risk_sensitivity_v4": project / "results" / "method_y_revival_v4_image_factorial_v1"
        / "attribution" / "risk_sensitivity_subset.json",
        "pairwise_v4_2": base / "branchwise_matched_pair_points.json",
        "joint_pairs_v4_2": base / "branchwise_joint_fold_bootstrap.json",
    }
    external_objects: dict[str, dict[str, Any]] = {}
    external_sha: dict[str, str] = {}
    for label, path in external_paths.items():
        external_objects[label], external_sha[label] = _read_object(path)
    report = synthesize(
        receipts, receipt_sha,
        quantitative=external_objects["quantitative_v2"],
        xai=external_objects["xai_v4"],
        external=external_objects["external_v2"],
        v3_claim=external_objects["claim_v3"],
        scanner_support={branch: external_objects[f"scanner_v3_{branch}"] for branch in BRANCHES},
        risk_sensitivity=external_objects["risk_sensitivity_v4"],
        pairwise=external_objects["pairwise_v4_2"],
        joint_pairs=external_objects["joint_pairs_v4_2"],
        external_sha256=external_sha,
    )
    report["source_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["report_sha256"] = sha256_mapping(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    _require(report.get("report_sha256") == sha256_mapping(
        {k: v for k, v in report.items() if k != "report_sha256"}
    ), "SCIENCE_STATUS_HASH_MISMATCH")
    rows = [
        "# Method Y: verified post-training scientific status (v4.2)",
        "",
        "Training is complete: v3 has 20/20 and v4 has 50/50 locked outer cells. "
        "Scientific validation is **blocked**; all results below are internal, post-hoc and exploratory.",
        "",
        "| Branch | Image-evaluable / endpoint | v3 image − scalar ΔIBS (95% CI) | "
        "v4 fold-local policy − A0 ΔIBS (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        item = report["branch_results"][branch]
        v3 = item["v3_image_increment_over_scalar"]
        v4 = item["v4_policy_increment_over_a0"]
        fmt = lambda value: f"{value:+.4f}"
        ci3 = ", ".join(fmt(v) for v in v3["delta_ibs_95ci"])
        ci4 = ", ".join(fmt(v) for v in v4["delta_ibs_95ci"])
        rows.append(
            f"| {branch} | {item['image_evaluable_n']} / {item['endpoint_parent_n']} | "
            f"{fmt(v3['delta_ibs'])} [{ci3}] | {fmt(v4['delta_ibs'])} [{ci4}] |"
        )
    rows.extend([
        "",
        "The v3 tau image signal is favourable on locked internal predictions, but scanner "
        "robustness and quantitative PET provenance do not permit a positive claim. "
        "The v3 amyloid IBS estimate uses a post-hoc support-restricted horizon grid; "
        "it is not the same estimand as the original full-grid IBS.",
        "",
        "v4's original global arm choice used overlapping outer-training partitions. "
        "The fold-local reconstruction is post-hoc, not an unbiased confirmatory model comparison. "
        "The pooled-branch Friedman test and non-inferiority on raw risk-score differences "
        "are invalid for a scientific claim. All ten planned pairwise point contrasts "
        "now have joint fold-stratified subject-bootstrap intervals, conditional on locked "
        "models and post-hoc analysis; no p-value or superiority claim follows. "
        "Amyloid's joint-pair IBS grid excludes day 365 after an outcome-support check, "
        "so its intervals are a new sensitivity estimand, not source-grid IBS. "
        "Calibration reports exist at estimable horizons, "
        "but their Hessian intervals do not include fold/model uncertainty.",
        "",
        "Atlas asset checks passed, but neither FBP nor FTP pattern-normalized images "
        "are validated as official Centiloid/CenTauR measurements. "
        "The XAI gate failed (45/50 cells evaluated; 30 nonpass). "
        "Its risk-target sensitivity maps were negatively correlated "
        f"(descriptive rho {report['xai']['risk_target_sensitivity_rho_descriptive']:+.3f}); "
        "the legacy voxel-bootstrap CI is not a subject-level CI. "
        "Independent external validation is pending. No diagnostic, causal, quantitative-equivalence "
        "or external-generalization claim is authorized.",
        "",
        "This report only consumes hash-verified locked artifacts. It does not retrain, "
        "re-evaluate an outer test set, or change any Method X work.",
        "",
        f"Report SHA-256: `{report['report_sha256']}`",
        "",
    ])
    return "\n".join(rows)


def write_status(project_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    report = build_status(project_root)
    out = Path(output_root)
    json_path = out / "SCIENCE_STATUS_CORRECTED.json"
    md_path = out / "SCIENCE_STATUS_CORRECTED.md"
    markdown = render_markdown(report).encode("utf-8")
    if json_path.exists():
        _require(json.loads(json_path.read_text(encoding="utf-8")) == report,
                 "SCIENCE_STATUS_OUTPUT_CONTRACT_MISMATCH")
    if md_path.exists():
        _require(md_path.read_bytes() == markdown,
                 "SCIENCE_STATUS_MARKDOWN_CONTRACT_MISMATCH")
    if not json_path.exists():
        atomic_write_json(json_path, report)
    if not md_path.exists():
        atomic_write_bytes(md_path, markdown)
    return report
