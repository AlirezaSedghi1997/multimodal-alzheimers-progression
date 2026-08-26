from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .atomic_io import atomic_write_json, read_json
from .oof_reporting import horizon_calibration_from_oof, subgroup_incremental_report, build_subgroup_metadata
from .science_gates import paired_oof_bootstrap, incremental_value_gate
from .image_robustness_v3 import (
    leave_manufacturer_out_eligibility,
    oof_scanner_probe,
)


class ImageReportingV3Error(ValueError):
    pass


def build_calibration_reports(
    *,
    base_variant: dict[str, Any],
    augmented_variant: dict[str, Any],
    horizons_days: list[float],
) -> dict[str, Any]:
    reports = {}
    for horizon in horizons_days:
        # Use locked OOF without refit
        base_cal = horizon_calibration_from_oof(base_variant, horizon_days=float(horizon), minimum_effective_rows=10)
        aug_cal = horizon_calibration_from_oof(augmented_variant, horizon_days=float(horizon), minimum_effective_rows=10)
        reports[str(horizon)] = {"base": base_cal, "augmented": aug_cal, "horizon_days": float(horizon)}
    return {"status": "COMPLETED", "horizons": reports}


def build_subgroup_reports(
    *,
    base_variant: dict[str, Any],
    augmented_variant: dict[str, Any],
    metadata: pd.DataFrame,
    subgroup_columns: list[str],
    minimum_subjects: int = 30,
    minimum_events: int = 10,
    iterations: int = 2000,
    seed: int = 123,
) -> dict[str, Any]:
    return subgroup_incremental_report(
        base_variant=base_variant,
        augmented_variant=augmented_variant,
        metadata=metadata,
        subgroup_columns=subgroup_columns,
        minimum_subjects=minimum_subjects,
        minimum_events=minimum_events,
        iterations=iterations,
        seed=seed,
    )


def build_paired_bootstrap_report(
    *,
    base_variant: dict[str, Any],
    augmented_variant: dict[str, Any],
    output_path: Path,
    iterations: int = 2000,
    seed: int = 20260714,
) -> dict[str, Any]:
    # 2000 iterations fixed
    if int(iterations) != 2000:
        raise ImageReportingV3Error("BOOTSTRAP_ITERATIONS_MUST_BE_2000")
    from .science_gates import cached_paired_oof_bootstrap

    return cached_paired_oof_bootstrap(
        base_variant,
        augmented_variant,
        output_path=output_path,
        iterations=iterations,
        seed=seed,
    )


def claim_consistency_state(
    *,
    training_execution_allowed: bool,
    incremental_value_pass: bool,
    quantitative_equivalence_proven: bool,
    external_validation_pass: bool,
    scanner_robustness_passed: bool,
) -> dict[str, Any]:
    # training_execution_allowed = hard blockers empty
    internal_incremental_pass = bool(incremental_value_pass and scanner_robustness_passed)
    quantitative_standard_claim = bool(internal_incremental_pass and quantitative_equivalence_proven)
    external_claim = bool(quantitative_standard_claim and external_validation_pass)
    return {
        "training_execution_allowed": bool(training_execution_allowed),
        "internal_incremental_pass": internal_incremental_pass,
        "quantitative_standard_claim_allowed": quantitative_standard_claim,
        "external_claim_allowed": external_claim,
        "positive_claim_allowed": quantitative_standard_claim,  # same as internal incremental + quant
        "scanner_support_available_is_not_gate": True,
    }


def render_private_and_public(
    *,
    private_oof_predictions: list[dict[str, Any]],
    private_embeddings: list[list[float]] | None,
    output_root: Path,
    branch: str,
    variant: str,
) -> dict[str, Any]:
    private_root = output_root / "private" / "oof_predictions" / branch / variant
    private_root.mkdir(parents=True, exist_ok=True)
    public_root = output_root / "reports"
    public_root.mkdir(parents=True, exist_ok=True)
    # private contains patient hashes, predictions, embeddings
    private_payload = {
        "oof_predictions": private_oof_predictions,
        "embeddings": private_embeddings,
        "branch": branch,
        "variant": variant,
    }
    atomic_write_json(private_root / "oof_private.json", private_payload)
    # public aggregate: no patient_hash, no individual prediction, no embedding vector
    public_payload = {
        "branch": branch,
        "variant": variant,
        "aggregate_metrics": {
            "subject_count": len(private_oof_predictions),
            "event_count": sum(int(r.get("event", 0)) for r in private_oof_predictions),
        },
        "report_type": "AGGREGATE_ONLY",
    }
    public_path = public_root / f"{branch}_{variant}_aggregate.json"
    atomic_write_json(public_path, public_payload)
    return {"private_path": str(private_root / "oof_private.json"), "public_path": str(public_path)}


def privacy_scan_reports(report_root: Path) -> dict[str, Any]:
    """Recursively reject identifiers and row-level outcomes in public JSON."""
    forbidden_tokens = ["patient_hash", "image_path_private", "H:\\", "A4 Learn", "individual_prediction", "embedding vector", ".pt", ".ckpt", "private parquet"]
    forbidden_keys = {
        "patient_hash", "source_subject_id", "raw_path", "image_path_private",
        "oof_predictions", "censoring_reference", "event", "time_days",
        "original_event", "original_time_days", "risk_score",
        "survival_probabilities", "embedding", "embeddings", "oof_embeddings",
    }
    row_level_keys = {
        "censoring_reference", "event", "time_days", "original_event",
        "original_time_days", "risk_score", "survival_probabilities",
    }
    violations: list[str] = []

    def inspect(value: Any, *, path: Path, location: str = "$") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in forbidden_keys:
                    kind = "row_level" if normalized in row_level_keys else "forbidden_key"
                    violations.append(f"{path}:{kind}:{location}.{key}")
                inspect(child, path=path, location=f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, path=path, location=f"{location}[{index}]")

    for path in report_root.rglob("*.json"):
        # Only scan public reports (under reports), not private
        if "private" in str(path).lower():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            inspect(data, path=path)
            lowered = text.lower()
            for token in forbidden_tokens:
                if token.lower() in lowered:
                    violations.append(f"{path}:contains:{token}")
        except Exception as exc:
            violations.append(f"{path}:read_error:{exc}")
    status = "PASS" if not violations else "FAIL"
    return {"status": status, "violations": violations, "scanned_root": str(report_root)}


def build_scanner_robustness_report(
    *, augmented_variant: dict[str, Any], metadata: pd.DataFrame | None
) -> dict[str, Any]:
    """Run OOF-only scanner diagnostics without upgrading feasibility to proof."""
    if metadata is None or metadata.empty:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "SCANNER_METADATA_UNAVAILABLE",
            "scanner_robustness_passed": False,
        }
    required = {"patient_hash", "manufacturer"}
    if not required.issubset(metadata.columns):
        raise ImageReportingV3Error("SCANNER_METADATA_COLUMNS_MISSING")
    if metadata["patient_hash"].astype(str).duplicated().any():
        raise ImageReportingV3Error("SCANNER_METADATA_PATIENT_DUPLICATE")

    prediction_rows = [
        row
        for fold in augmented_variant.get("folds", [])
        for row in fold.get("oof_predictions", [])
    ]
    embedding_rows = [
        row
        for fold in augmented_variant.get("folds", [])
        for row in fold.get("oof_embeddings", [])
    ]
    prediction_by_patient = {str(row.get("patient_hash")): row for row in prediction_rows}
    embedding_by_patient = {str(row.get("patient_hash")): row for row in embedding_rows}
    if len(prediction_by_patient) != len(prediction_rows):
        raise ImageReportingV3Error("OOF_PREDICTION_PATIENT_DUPLICATE")
    if len(embedding_by_patient) != len(embedding_rows):
        raise ImageReportingV3Error("OOF_EMBEDDING_PATIENT_DUPLICATE")

    metadata_index = metadata.copy()
    metadata_index["patient_hash"] = metadata_index["patient_hash"].astype(str)
    metadata_index["manufacturer"] = (
        metadata_index["manufacturer"].fillna("UNKNOWN").astype(str).str.upper()
    )
    metadata_index = metadata_index.set_index("patient_hash")
    common = sorted(set(prediction_by_patient).intersection(metadata_index.index))
    support: dict[str, dict[str, int]] = {}
    for patient in common:
        manufacturer = str(metadata_index.loc[patient, "manufacturer"])
        cell = support.setdefault(manufacturer, {"subjects": 0, "events": 0})
        cell["subjects"] += 1
        cell["events"] += int(prediction_by_patient[patient].get("event", 0))
    leave_out = leave_manufacturer_out_eligibility(support)

    embedding_common = sorted(set(embedding_by_patient).intersection(metadata_index.index))
    if embedding_common:
        embeddings = np.asarray(
            [embedding_by_patient[patient].get("embedding", []) for patient in embedding_common],
            dtype=float,
        )
        manufacturers = [
            str(metadata_index.loc[patient, "manufacturer"])
            for patient in embedding_common
        ]
        probe = oof_scanner_probe(embeddings, manufacturers)
    else:
        probe = {"status": "NOT_ESTIMABLE", "reason": "OOF_EMBEDDINGS_UNAVAILABLE"}

    unknown_count = sum(
        int(values.get("subjects", 0))
        for manufacturer, values in support.items()
        if manufacturer == "UNKNOWN"
    )
    return {
        "status": "COMPLETED_WITH_CLAIM_GATE_UNRESOLVED",
        "manufacturer_support": support,
        "leave_manufacturer_out_eligibility": leave_out,
        "oof_scanner_probe": probe,
        "unknown_scanner_excluded_sensitivity": {
            "status": "ELIGIBILITY_COUNTED",
            "unknown_excluded_count": int(unknown_count),
            "remaining_count": int(len(common) - unknown_count),
        },
        "scanner_robustness_passed": False,
        "terminal_reason": "LEAVE_MANUFACTURER_OUT_PERFORMANCE_NOT_ESTABLISHED",
    }


def build_real_oof_reports(
    *,
    output_root: Path,
    branch: str,
    variants: dict[str, Any],
    metadata: pd.DataFrame | None,
    horizons_days: list[float],
    bootstrap_iterations: int = 2000,
    seed: int = 20260714,
    scanner_sensitivity_passed: bool = False,
    quantitative_equivalence_proven: bool = False,
    external_validation_pass: bool = False,
    expected_outer_folds: int = 5,
) -> dict[str, Any]:
    """Plan 012 executor over REAL locked OOF folds.

    Consumes weight-derived outer predictions (never placeholders), emits
    public aggregate reports only, and evaluates the incremental claim gate.
    The paired primary bootstrap is locked at 2000 iterations (Plan 012 STOP
    condition); absence of subgroup metadata or embeddings is recorded
    honestly, never PASS.
    """
    if int(bootstrap_iterations) != 2000:
        raise ImageReportingV3Error("BOOTSTRAP_ITERATIONS_MUST_BE_2000")
    if set(variants.keys()) != {"scalar_tabular", "scalar_tabular_plus_pattern_image"}:
        raise ImageReportingV3Error("REAL_OOF_VARIANTS_MUST_BE_MATCHED_PAIR")
    base = variants["scalar_tabular"]
    aug = variants["scalar_tabular_plus_pattern_image"]
    expected = set(range(int(expected_outer_folds)))
    for name, payload in (("base", base), ("augmented", aug)):
        observed = {int(row.get("fold", -1)) for row in payload.get("folds", [])}
        if observed != expected:
            raise ImageReportingV3Error(
                f"FAILED_INCOMPLETE_OUTER_CV:{name}:expected={sorted(expected)}:observed={sorted(observed)}"
            )
    for base_fold, aug_fold in zip(
        sorted(base["folds"], key=lambda row: int(row["fold"])),
        sorted(aug["folds"], key=lambda row: int(row["fold"])),
    ):
        base_members = {
            str(row.get("patient_hash")) for row in base_fold.get("oof_predictions", [])
        }
        aug_members = {
            str(row.get("patient_hash")) for row in aug_fold.get("oof_predictions", [])
        }
        if base_members != aug_members:
            raise ImageReportingV3Error("MATCHED_VARIANT_MEMBERSHIP_MISMATCH")

    rep_dir = Path(output_root) / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    calibration = build_calibration_reports(
        base_variant=base, augmented_variant=aug, horizons_days=horizons_days
    )
    atomic_write_json(rep_dir / f"{branch}_calibration.json", calibration)

    if metadata is not None and len(metadata):
        subgroups = build_subgroup_reports(
            base_variant=base, augmented_variant=aug, metadata=metadata,
            subgroup_columns=["sex_binary", "age_group", "treatment_active", "manufacturer"],
            minimum_subjects=30, minimum_events=10,
            iterations=int(bootstrap_iterations), seed=seed,
        )
    else:
        subgroups = {"status": "NOT_AVAILABLE_METADATA"}
    atomic_write_json(rep_dir / f"{branch}_subgroups.json", subgroups)

    scanner_audit = build_scanner_robustness_report(
        augmented_variant=aug, metadata=metadata
    )
    atomic_write_json(rep_dir / f"{branch}_scanner_robustness.json", scanner_audit)
    scanner_pass = bool(
        scanner_sensitivity_passed
        and scanner_audit.get("scanner_robustness_passed") is True
    )

    bootstrap = build_paired_bootstrap_report(
        base_variant=base, augmented_variant=aug,
        output_path=rep_dir / f"{branch}_paired_bootstrap.json",
        iterations=int(bootstrap_iterations), seed=seed,
    )
    gate = incremental_value_gate(
        delta_ibs_ci=tuple(bootstrap["delta_ibs_95ci"]),
        delta_cindex_ci=tuple(bootstrap["delta_cindex_95ci"]),
        outer_folds_completed=len(base.get("folds", [])),
        expected_outer_folds=int(expected_outer_folds),
        scanner_sensitivity_passed=scanner_pass,
    )
    incremental = {
        "branch": branch,
        "science_gate": gate,
        "bootstrap": {
            "delta_ibs_mean": bootstrap["delta_ibs_mean"],
            "delta_ibs_95ci": bootstrap["delta_ibs_95ci"],
            "delta_cindex_mean": bootstrap["delta_cindex_mean"],
            "delta_cindex_95ci": bootstrap["delta_cindex_95ci"],
            "iterations": int(bootstrap_iterations),
        },
        "oof_source": "LOCKED_CHECKPOINT_WEIGHTS",
    }
    atomic_write_json(rep_dir / f"{branch}_image_incremental.json", incremental)

    claim = claim_consistency_state(
        training_execution_allowed=True,
        incremental_value_pass=(gate["science_gate_status"] == "PASS"),
        quantitative_equivalence_proven=bool(quantitative_equivalence_proven),
        external_validation_pass=bool(external_validation_pass),
        scanner_robustness_passed=scanner_pass,
    )
    atomic_write_json(rep_dir / f"{branch}_IMAGE_CLAIM_POLICY.json", claim)

    scan = privacy_scan_reports(rep_dir)
    status = "COMPLETED_FROM_REAL_OOF" if gate["science_gate_status"] in {"PASS", "FAIL"} else "PARTIAL"
    return {
        "status": status,
        "branch": branch,
        "science_gate": gate,
        "claim": claim,
        "privacy_scan": scan,
        "public_reports": sorted(p.name for p in rep_dir.glob("*.json")),
    }


def build_aggregate_claim_policy(
    *,
    output_root: Path,
    branch_policies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-branch claim policies into the single public policy file.

    A permission is granted only when EVERY branch grants it (AND semantics).
    Blockers are indexed so no branch-specific caveat is lost.
    """
    if not branch_policies:
        raise ImageReportingV3Error("AGGREGATE_REQUIRES_AT_LEAST_ONE_BRANCH")
    rep_dir = Path(output_root) / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    def _and(field: str) -> bool:
        return all(bool(policy.get(field)) for policy in branch_policies.values())

    limitations: dict[str, list[str]] = {}
    for branch, policy in sorted(branch_policies.items()):
        for blocker in policy.get("hard_blockers", []) or []:
            limitations.setdefault(blocker, []).append(branch)
    aggregate = {
        "schema_version": "3.0",
        "scope": "AGGREGATE_OVER_BRANCHES",
        "branches": sorted(branch_policies.keys()),
        "training_execution_allowed": _and("training_execution_allowed"),
        "internal_incremental_pass": _and("internal_incremental_pass"),
        "positive_claim_allowed": _and("positive_claim_allowed"),
        "quantitative_standard_claim_allowed": _and(
            "quantitative_standard_claim_allowed"
        ),
        "external_claim_allowed": _and("external_claim_allowed"),
        "per_branch": branch_policies,
        "blocker_index": {k: sorted(set(v)) for k, v in sorted(limitations.items())},
    }
    atomic_write_json(rep_dir / "IMAGE_CLAIM_POLICY.json", aggregate)
    return aggregate
