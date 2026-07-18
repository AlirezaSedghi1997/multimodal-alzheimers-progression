"""Safe, read-only preparation work that can run while FTP transforms continue."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(ROOT_TOOLS) not in sys.path:
    sys.path.insert(0, str(ROOT_TOOLS))

from core import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    compute_sha256,
    disk_snapshot,
    path_fingerprint,
    process_snapshot,
    utc_now,
)


def build_endpoint_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "contract_id": "method_y_future_progression_endpoint_v2_pretransform",
        "index_date": "baseline_valid_pet_date",
        "target_derivation": "future_clinical_only",
        "primary_endpoint": "first_confirmed_sustained_CDR_global_worsening_after_index_pet",
        "secondary_endpoint": "future_PACC_decline_only_if_longitudinal_contract_passes",
        "baseline_requirement": "CDR_global_equals_0_at_or_near_index",
        "confirmation_rule": "first_worsening_followed_by_non-reversal_or_confirmatory_visit",
        "censoring_rule": "last_reliable_followup_or_death_with_provenance",
        "branches": {
            "amyloid": {
                "tracer": "FBP",
                "features": ["global_centiloid_or_valid_suvr", "prespecified_spatial_extent_optional"],
                "target": "future_clinical_only",
            },
            "tau": {
                "tracer": "FTP",
                "features": ["CenTauR_or_valid_suvr_preserving_regional_features"],
                "target": "future_clinical_only",
            },
            "paired_at": {
                "tracers": ["FBP", "FTP"],
                "question": "tau_incremental_value_beyond_amyloid",
                "target": "future_clinical_only",
            },
        },
        "forbidden_targets": [
            "same_pet_derived_label",
            "amyloid_positivity_derived_from_same_pet",
            "centiloid_threshold_derived_from_same_pet",
            "post_origin_diagnosis_as_baseline_target",
        ],
        "forbidden_predictors": [
            "post_origin_pet",
            "future_visit_feature",
            "future_diagnosis_note",
            "cohort_imputed_label",
        ],
        "claim_status_before_external_validation": "INTERNAL_ONLY_OR_EXPLORATORY",
    }


def build_model_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "contract_id": "method_y_nested_survival_v2_pretransform",
        "outer_folds": 5,
        "inner_folds": 3,
        "seed": 20260714,
        "group_unit": "subject_key",
        "selection_metric": "integrated_brier_score",
        "primary_model": "coxnet",
        "challengers": ["rsf", "gbsa"],
        "equal_hpo_budget": True,
        "outer_test_access": "after_inner_selection_and_lock_only",
        "outer_evaluation": "exactly_once_after_checkpoint_lock",
        "preprocessing_fit_scope": "training_fold_only",
        "image_branch": "secondary_only_after_transform_roi_scanner_gates",
        "horizons_days": [547.875, 1095.75, 1640.0],
        "metrics": [
            "uno_c_index",
            "harrell_c_index",
            "dynamic_auc",
            "integrated_brier_score",
            "calibration_slope",
            "calibration_intercept",
        ],
        "uncertainty": "clustered_subject_bootstrap",
    }


def evaluate_contracts(endpoint: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if endpoint.get("target_derivation") != "future_clinical_only":
        errors.append("endpoint target is not future clinical only")
    if "same_pet_derived_label" not in endpoint.get("forbidden_targets", []):
        errors.append("same PET derived target is not forbidden")
    branches = endpoint.get("branches", {})
    if branches.get("amyloid", {}).get("tracer") != "FBP":
        errors.append("amyloid tracer must be FBP")
    if branches.get("tau", {}).get("tracer") != "FTP":
        errors.append("tau tracer must be FTP")
    if branches.get("amyloid", {}).get("tracer") == branches.get("tau", {}).get("tracer"):
        errors.append("amyloid and tau tracers must be disjoint")
    if model.get("outer_folds") != 5 or model.get("inner_folds") != 3:
        errors.append("nested fold contract must be 5 outer and 3 inner")
    if model.get("outer_test_access") != "after_inner_selection_and_lock_only":
        errors.append("outer test must be unavailable during HPO")
    if model.get("selection_metric") != "integrated_brier_score":
        errors.append("winner selection metric must be integrated Brier score")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors}


def compare_transform_status(
    manifest_summary: dict[str, Any],
    live_status: dict[str, Any] | None,
    active_writer: bool,
) -> dict[str, Any]:
    flags: list[str] = []
    if not live_status:
        flags.append("LIVE_STATUS_MISSING")
    else:
        transform = live_status.get("transform", {})
        manifest_pass = int(manifest_summary.get("status_counts", {}).get("PASS", 0))
        validated = int(transform.get("validated_count", 0))
        source_eligible = int(transform.get("source_eligible_count", 0))
        if manifest_pass != validated:
            flags.append("STALE_VALIDATED_COUNT")
        if manifest_summary.get("row_count", 0) > source_eligible:
            flags.append("MANIFEST_EXCEEDS_SOURCE_ELIGIBLE")
        if active_writer and live_status.get("status") not in {"RUNNING", "IN_PROGRESS"}:
            flags.append("STATUS_NOT_RUNNING_WHILE_WRITER_ACTIVE")
    return {
        "status": "PASS" if not flags else "RECONCILIATION_REQUIRED",
        "flags": flags,
        "manifest_pass_count": int(manifest_summary.get("status_counts", {}).get("PASS", 0)),
        "live_validated_count": int((live_status or {}).get("transform", {}).get("validated_count", 0)),
        "active_writer": active_writer,
    }


def _safe_json(path: Path) -> tuple[Any, str | None]:
    try:
        raw = path.read_bytes()
        return json.loads(raw.decode("utf-8")), compute_sha256(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def snapshot_transform(project_root: Path) -> dict[str, Any]:
    manifest_path = project_root / "results" / "method_y_revival_v2" / "transforms" / "manifest.json"
    payload, digest_or_error = _safe_json(manifest_path)
    error = None if payload is not None else digest_or_error
    rows = payload if isinstance(payload, list) else (payload or {}).get("rows", (payload or {}).get("items", []))
    rows = rows if isinstance(rows, list) else []
    counts: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict):
            status = str(row.get("status", row.get("qc_status", row.get("transform_status", "UNKNOWN"))))
            counts[status] = counts.get(status, 0) + 1
    process = process_snapshot()
    relevant = [
        row for row in process.get("processes", [])
        if any(term in str(row.get("CommandLine", "")).lower() for term in ("method_y_transforms_manual", "materialize_method_y_transforms_v2"))
    ]
    return {
        "captured_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": compute_sha256(manifest_path) if manifest_path.exists() else None,
        "manifest_parse_error": error,
        "row_count": len(rows),
        "status_counts": counts,
        "active_writer": bool(relevant),
        "active_writer_processes": relevant,
        "writer_was_stopped": False,
        "writer_was_modified": False,
    }


def _asset_registry(project_root: Path) -> dict[str, Any]:
    asset_paths = {
        "centaur_universal": project_root / "results/method_y_revival_v2/provenance/assets/unpacked/centaur/SPM/CenTauR.nii.gz",
        "centaur_frontal": project_root / "results/method_y_revival_v2/provenance/assets/unpacked/centaur/SPM/Frontal_CenTauR.nii.gz",
        "centaur_mesial": project_root / "results/method_y_revival_v2/provenance/assets/unpacked/centaur/SPM/Mesial_CenTauR.nii.gz",
        "centaur_meta": project_root / "results/method_y_revival_v2/provenance/assets/unpacked/centaur/SPM/Meta_CenTauR.nii.gz",
        "centaur_temporoparietal": project_root / "results/method_y_revival_v2/provenance/assets/unpacked/centaur/SPM/TP_CenTauR.nii.gz",
    }
    assets: list[dict[str, Any]] = []
    for asset_id, path in asset_paths.items():
        assets.append(
            {
                "asset_id": asset_id,
                "role": "quantitative_biomarker_mask",
                "branch": "tau",
                "path": str(path),
                "local_file_present": path.exists(),
                "file_sha256": compute_sha256(path) if path.exists() else None,
                "space": "UNKNOWN_UNTIL_HEADER_AUDIT",
                "license_id": "PENDING_LICENSE_LOCK",
                "license_sha256": None,
                "label_table_sha256": None,
                "redistribution_allowed": False,
                "status": "LOCAL_REFERENCE_ONLY_LICENSE_AND_SPACE_PENDING",
            }
        )
    assets.append(
        {
            "asset_id": "gaain_centiloid_std_voi_v1",
            "role": "quantitative_biomarker_mask",
            "branch": "amyloid",
            "path": None,
            "local_file_present": False,
            "file_sha256": None,
            "space": "CENTILOID_DOCUMENTED_SPACE",
            "license_id": "PENDING_ASSET_REGISTRY",
            "license_sha256": None,
            "label_table_sha256": None,
            "redistribution_allowed": False,
            "status": "BLOCKED_UNTIL_LICENSE_AND_FILE_LOCK",
        }
    )
    assets.append(
        {
            "asset_id": "harvard_oxford_anatomical_xai",
            "role": "anatomical_xai_atlas",
            "branch": "shared",
            "path": None,
            "local_file_present": False,
            "file_sha256": None,
            "space": "MNI_REQUIRED",
            "license_id": "CC_BY_SA_4_OR_LOCAL_TERMS_PENDING",
            "license_sha256": None,
            "label_table_sha256": None,
            "redistribution_allowed": False,
            "status": "TEMPLATE_ONLY",
        }
    )
    return {"schema_version": "1.0", "assets": assets, "claim_bearing_allowed": False}


def _readiness(
    project_root: Path,
    transform: dict[str, Any],
    contract_check: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "task_id": "T01_transform_resume",
            "status": "RUNNING" if transform["active_writer"] else "NOT_RUNNING",
            "safe_before_transform_completion": True,
            "mutation": "none",
            "blockers": [] if transform["active_writer"] else ["active writer not observed"],
        },
        {
            "task_id": "T02_transform_gate",
            "status": "BLOCKED_UNTIL_MANIFEST_TERMINAL",
            "safe_before_transform_completion": True,
            "mutation": "none",
            "blockers": ["manifest is live; do not train or claim"],
        },
        {
            "task_id": "T03_endpoint_contract",
            "status": "READY_PREPARED" if contract_check["status"] == "PASS" else "FAIL",
            "safe_before_transform_completion": True,
            "mutation": "workspace_only",
            "blockers": contract_check["errors"],
        },
        {
            "task_id": "T04_nested_split_contract",
            "status": "READY_TEMPLATE_ONLY",
            "safe_before_transform_completion": True,
            "mutation": "workspace_only",
            "blockers": ["final membership digest must be locked after eligible transform/QC set is terminal"],
        },
        {
            "task_id": "T05_survival_model_contract",
            "status": "READY_TEMPLATE_ONLY",
            "safe_before_transform_completion": True,
            "mutation": "workspace_only",
            "blockers": ["no training before transform and endpoint gates"],
        },
        {
            "task_id": "T06_quantitative_pet_validation",
            "status": "BLOCKED",
            "safe_before_transform_completion": False,
            "mutation": "none",
            "blockers": ["transform terminal set", "official atlas/space/license lock"],
        },
        {
            "task_id": "T07_external_validation",
            "status": "BLOCKED_DUA",
            "safe_before_transform_completion": True,
            "mutation": "workspace_only",
            "blockers": ["SCAN/NACC DUA", "ADNI DUA", "OASIS3 DUA"],
        },
        {
            "task_id": "T08_xai_contract",
            "status": "READY_TEMPLATE_ONLY",
            "safe_before_transform_completion": True,
            "mutation": "workspace_only",
            "blockers": ["locked checkpoint and licensed anatomical atlas"],
        },
        {
            "task_id": "T09_privacy_and_claim_policy",
            "status": "READY_PREPARED",
            "safe_before_transform_completion": True,
            "mutation": "workspace_only",
            "blockers": [],
        },
    ]


def run_preparation(project_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("00_admin", "01_preflight", "02_contracts", "03_audits", "04_reports", "05_tests"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    transform_t0 = snapshot_transform(project_root)
    endpoint = build_endpoint_contract()
    model = build_model_contract()
    contract_check = evaluate_contracts(endpoint, model)
    existing_run_contract, run_contract_digest_or_error = _safe_json(
        project_root / "results" / "method_y_revival_v2" / "run_contract.json"
    )
    live_status, live_status_digest_or_error = _safe_json(
        project_root / "results" / "method_y_revival_v2" / "reports" / "LIVE_STATUS_V2.json"
    )
    provenance, provenance_digest_or_error = _safe_json(
        project_root / "results" / "method_y_revival_v2" / "provenance" / "quantitative_pet_feature_provenance_reference.json"
    )
    assets = _asset_registry(project_root)
    status_consistency = compare_transform_status(
        transform_t0,
        live_status,
        transform_t0["active_writer"],
    )
    readiness = _readiness(project_root, transform_t0, contract_check)
    if status_consistency["status"] != "PASS":
        for item in readiness:
            if item["task_id"] == "T02_transform_gate":
                item["blockers"].append("LIVE_STATUS_RECONCILIATION_REQUIRED")
    preparation = {
        "schema_version": "1.0",
        "preparation_id": "method_y_pre_transform_preparation_v1",
        "created_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "read_only_inputs": True,
        "active_writer_policy": "never_stop_or_modify",
        "transform_t0": transform_t0,
        "existing_v2_contract": {
            "readable": existing_run_contract is not None,
            "sha256": compute_sha256(project_root / "results/method_y_revival_v2/run_contract.json")
            if (project_root / "results/method_y_revival_v2/run_contract.json").exists() else None,
            "parse_error": None if existing_run_contract is not None else run_contract_digest_or_error,
        },
        "existing_live_status": {
            "payload": live_status,
            "parse_error": None if live_status is not None else live_status_digest_or_error,
        },
        "existing_provenance_reference": {
            "payload": provenance,
            "parse_error": None if provenance is not None else provenance_digest_or_error,
        },
        "contract_check": contract_check,
        "status_consistency": status_consistency,
        "readiness": readiness,
        "disk": disk_snapshot(),
        "protected_paths": {
            rel: path_fingerprint(
                project_root / rel,
                max_files=5000,
                exclude_relative=("method_y_revival_v2/transforms",) if rel == "results" else (),
            )
            for rel in ("src", "scripts", "configs", "data", "results")
        },
        "completion_policy": {
            "training_allowed_now": False,
            "outer_test_allowed_now": False,
            "claim_allowed_now": False,
            "external_validation_allowed_now": False,
        },
    }
    atomic_write_json(output_root / "01_preflight" / "preparation_report.json", preparation)
    atomic_write_json(output_root / "01_preflight" / "transform_T0.json", transform_t0)
    atomic_write_json(output_root / "02_contracts" / "endpoint_contract.json", endpoint)
    atomic_write_json(output_root / "02_contracts" / "model_evaluation_contract.json", model)
    atomic_write_json(output_root / "02_contracts" / "asset_registry.json", assets)
    atomic_write_json(
        output_root / "02_contracts" / "external_validation_mapping_template.json",
        {
            "schema_version": "1.0",
            "status": "TEMPLATE_ONLY_DUA_PENDING",
            "cohorts": ["SCAN_NACC", "ADNI", "OASIS3"],
            "outcomes": {"baseline_cdr": None, "followup_cdr": None, "visit_date": None},
            "features": {"amyloid_centiloid": None, "tau_regional": None, "t1_transform": None},
            "no_outcome_opening_before_model_lock": True,
        },
    )
    atomic_write_json(
        output_root / "02_contracts" / "xai_contract.json",
        {
            "schema_version": "1.0",
            "status": "TEMPLATE_ONLY",
            "checkpoint_loading": "strict",
            "require_model_contract_hash": True,
            "require_licensed_atlas": True,
            "methods": ["integrated_gradients", "occlusion_sensitivity", "gradcam_plus_plus_if_valid_layer"],
            "sanity_checks": ["randomized_weights", "label_permutation", "top_roi_vs_random_roi_fidelity"],
            "public_output": "aggregate_only",
            "claim_status": "NO_CLAIM",
        },
    )
    atomic_write_json(output_root / "02_contracts" / "method_x_matched_benchmark_template.json", {
        "status": "TEMPLATE_ONLY",
        "backbones": ["resnet10", "resnet18", "densenet121", "efficientnet_b0", "mednext_s", "tiny"],
        "difference_allowed": ["backbone_only"],
        "common": {
            "folds": 5, "seed": 20260714, "fusion": "locked_common",
            "loss": "locked_common", "scheduler": "locked_common", "augmentation": "locked_common",
        },
        "claim_status": "TECHNICAL_ONLY_UNTIL_WITHIN_COHORT_LABEL_VARIATION",
    })
    transform_tfinal = snapshot_transform(project_root)
    atomic_write_json(output_root / "01_preflight" / "transform_Tfinal.json", transform_tfinal)
    preparation["transform_tfinal"] = transform_tfinal
    preparation["resume_state"] = "PRE_TRANSFORM_PREPARATION_VERIFIED"
    atomic_write_json(output_root / "01_preflight" / "preparation_report.json", preparation)
    readiness_lines = [
        "# گزارش کارهای قابل انجام پیش از پایان transform",
        "",
        "وضعیت این package: `PRE_TRANSFORM_PREPARATION_VERIFIED`",
        "",
        "این package هیچ transform، cache، training، outer-test یا claim را اجرا نکرده است.",
        "",
        "## کارهای انجام‌شده",
        "",
        "- snapshot دوگانه‌ی process و manifest؛",
        "- endpoint contract با future-only target؛",
        "- جداسازی FBP/FTP؛",
        "- nested 5×3 model contract؛",
        "- equal-budget CoxNet/RSF/GBSA protocol؛",
        "- asset registry و license/space gates؛",
        "- external mapping template؛",
        "- strict XAI contract؛",
        "- matched Method X benchmark template؛",
        "- protected-path fingerprint و privacy/claim policy.",
        "",
        "## کارهای عمداً مسدودشده",
        "",
        "- quantitative PET validation تا terminal transform set؛",
        "- split membership lock تا eligible set نهایی؛",
        "- training و outer-test؛",
        "- external validation تا DUA؛",
        "- XAI واقعی تا checkpoint و atlas قفل‌شده.",
        "",
        "## سیاست resume",
        "",
        "اجرای مجدد همین preflight فقط snapshot جدید تولید می‌کند و هرگز writer فعال را متوقف نمی‌کند.",
    ]
    atomic_write_text(output_root / "04_reports" / "PRE_TRANSFORM_READY.md", "\n".join(readiness_lines) + "\n")
    atomic_write_json(output_root / "00_admin" / "state.json", {
        "state": "PRE_TRANSFORM_PREPARATION_VERIFIED",
        "updated_utc": utc_now(),
        "active_writer_stopped": False,
        "project_mutated": False,
        "next_stage": "WAIT_FOR_TERMINAL_TRANSFORM_AND_RUN_POST_TRANSFORM_GATE",
    })
    return preparation
