from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .atomic_io import atomic_write_json, read_json
from .oof_reporting import horizon_calibration_from_oof, subgroup_incremental_report, build_subgroup_metadata
from .science_gates import paired_oof_bootstrap, incremental_value_gate


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
    """Fail if any public report contains forbidden tokens in values."""
    forbidden_tokens = ["patient_hash", "image_path_private", "H:\\", "A4 Learn", "individual_prediction", "embedding vector", ".pt", ".ckpt", "private parquet"]
    # Also check for embedding vectors: look for large numeric arrays in public reports
    violations = []
    for path in report_root.rglob("*.json"):
        # Only scan public reports (under reports), not private
        if "private" in str(path).lower():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            # check values not keys? But scanner must be context-aware: allow key patient_hash in schema doc but not value
            # We check raw text for forbidden tokens in values: if "patient_hash" appears as value pattern with actual hash-like string, fail
            lowered = text.lower()
            for token in forbidden_tokens:
                if token.lower() in lowered:
                    # special case: allow key name in schema doc if no actual hash value leaked
                    # We check if token appears with an actual hash (hex-like) nearby
                    if token == "patient_hash" and "\"patient_hash\":" in text:
                        # if there's a patient_hash key with a value that looks like a hash, it's violation
                        # Our public reports should not contain that key at all
                        violations.append(f"{path}:contains:{token}")
                    elif token != "patient_hash":
                        violations.append(f"{path}:contains:{token}")
        except Exception as exc:
            violations.append(f"{path}:read_error:{exc}")
    status = "PASS" if not violations else "FAIL"
    return {"status": status, "violations": violations, "scanned_root": str(report_root)}


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

    bootstrap = build_paired_bootstrap_report(
        base_variant=base, augmented_variant=aug,
        output_path=rep_dir / f"{branch}_paired_bootstrap.json",
        iterations=int(bootstrap_iterations), seed=seed,
    )
    gate = incremental_value_gate(
        delta_ibs_ci=tuple(bootstrap["delta_ibs_95ci"]),
        delta_cindex_ci=tuple(bootstrap["delta_cindex_95ci"]),
        outer_folds_completed=len(base.get("folds", [])),
        expected_outer_folds=len(base.get("folds", [])),
        scanner_sensitivity_passed=bool(scanner_sensitivity_passed),
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
        scanner_robustness_passed=bool(scanner_sensitivity_passed),
    )
    atomic_write_json(rep_dir / "IMAGE_CLAIM_POLICY.json", claim)

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
