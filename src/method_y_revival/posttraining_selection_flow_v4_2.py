"""Audit who entered the Method Y image cohort from its endpoint population."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping
from .posttraining_correction_v4_2 import BRANCHES, CorrectionV42Error


BASELINE_VARIABLES = (
    "age", "sex_binary", "education_years", "apoe4_carrier",
    "treatment_active", "baseline_pacc_raw", "baseline_fctotal96",
    "baseline_ldeltotal", "baseline_digittotal", "baseline_mmscore",
    "baseline_amyloid_centiloid",
)
PROPENSITY_VARIABLES = (
    "age", "sex_binary", "education_years", "apoe4_carrier",
    "treatment_active", "baseline_pacc_raw", "baseline_mmscore",
    "baseline_amyloid_centiloid",
)


def standardized_mean_difference(included: pd.Series, excluded: pd.Series) -> float | None:
    """Unadjusted SMD with observed values only; null means not estimable."""
    a = pd.to_numeric(included, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(excluded, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    denominator = math.sqrt((float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2)
    if denominator == 0:
        return 0.0 if math.isclose(float(np.mean(a)), float(np.mean(b))) else None
    return float((np.mean(a) - np.mean(b)) / denominator)


def _numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    a = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "observed_n": int(len(a)),
        "missing_n": int(len(values) - len(a)),
        "mean": float(np.mean(a)) if len(a) else None,
        "median": float(np.median(a)) if len(a) else None,
        "q1": float(np.quantile(a, 0.25)) if len(a) else None,
        "q3": float(np.quantile(a, 0.75)) if len(a) else None,
    }


def _propensity_audit(frame: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    labels = frame["image_included"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 5:
        return {"status": "NOT_ESTIMABLE", "reason": "INSUFFICIENT_GROUP_SUPPORT"}
    features = frame.loc[:, list(PROPENSITY_VARIABLES)].apply(
        pd.to_numeric, errors="coerce"
    )
    columns = [name for name in features if features[name].notna().any()]
    if not columns:
        return {"status": "NOT_ESTIMABLE", "reason": "NO_OBSERVED_BASELINE_FEATURES"}
    features = features[columns]
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=int(seed)),
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    oof_probability = cross_val_predict(
        pipeline, features, labels, cv=splitter, method="predict_proba"
    )[:, 1]
    if not np.isfinite(oof_probability).all():
        raise CorrectionV42Error("SELECTION_PROPENSITY_NONFINITE")
    # Fit for a descriptive inclusion-weight ESS only.  It is not used to
    # adjust, train, or reselect any Method Y outcome model.
    pipeline.fit(features, labels)
    full_probability = np.clip(pipeline.predict_proba(features)[:, 1], 0.01, 0.99)
    selected_weights = 1.0 / full_probability[labels == 1]
    ess = float(selected_weights.sum() ** 2 / np.square(selected_weights).sum())
    return {
        "status": "COMPLETED_DESCRIPTIVE_ONLY",
        "baseline_features": columns,
        "cross_validated_inclusion_auc": float(roc_auc_score(labels, oof_probability)),
        "oof_inclusion_probability_q05_q50_q95": [
            float(v) for v in np.quantile(oof_probability, [0.05, 0.5, 0.95])
        ],
        "selected_inverse_inclusion_weight_ess": ess,
        "selected_n": int(counts[1]),
        "ess_fraction_of_selected": ess / float(counts[1]),
        "outcome_used_in_propensity_model": False,
        "ipw_outcome_adjustment_performed": False,
    }


def build_selection_flow(
    project_root: str | Path, correction_path: str | Path,
    *, seed: int = 20260714,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    correction_raw = Path(correction_path).read_bytes()
    correction = json.loads(correction_raw)
    if correction.get("report_sha256") != sha256_mapping(
        {k: v for k, v in correction.items() if k != "report_sha256"}
    ):
        raise CorrectionV42Error("CORRECTION_RECEIPT_HASH_MISMATCH")
    run_root = Path(correction["source_root"])
    contract = json.loads((run_root / "run_contract.json").read_text(encoding="utf-8"))
    if contract.get("digest") != correction.get("source_run_contract_sha256"):
        raise CorrectionV42Error("SELECTION_SOURCE_CONTRACT_MISMATCH")
    branches: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for branch in BRANCHES:
        endpoint_path = (
            project / "results" / "method_y_revival_v2" / "endpoints"
            / f"{branch}_progression_endpoint.parquet"
        )
        membership_ref = contract["input_fingerprints"][f"{branch}_membership"]
        membership_path = Path(membership_ref["path"]).resolve()
        if not endpoint_path.is_file() or not membership_path.is_file():
            raise CorrectionV42Error("SELECTION_INPUT_MISSING")
        if not membership_path.is_relative_to(project):
            raise CorrectionV42Error("SELECTION_MEMBERSHIP_OUTSIDE_PROJECT")
        endpoint_sha = hashlib.sha256(endpoint_path.read_bytes()).hexdigest()
        membership_sha = hashlib.sha256(membership_path.read_bytes()).hexdigest()
        if membership_sha != membership_ref["sha256"]:
            raise CorrectionV42Error("SELECTION_MEMBERSHIP_HASH_MISMATCH")
        sources[f"{branch}_endpoint"] = endpoint_sha
        sources[f"{branch}_membership"] = membership_sha
        endpoint = pd.read_parquet(endpoint_path)
        if branch == "amyloid":
            parent = endpoint[endpoint["cohort"].astype(str) == "A4"].copy()
        else:
            parent = endpoint[
                endpoint["cohort"].astype(str).isin(["A4_treatment", "A4_placebo"])
            ].copy()
        membership = pd.read_parquet(membership_path)
        included = membership[membership["role"] == "development"].copy()
        if parent["patient_hash"].duplicated().any() or included["patient_hash"].duplicated().any():
            raise CorrectionV42Error("SELECTION_DUPLICATE_SUBJECT")
        parent_keys = set(parent["patient_hash"].astype(str))
        included_keys = set(included["patient_hash"].astype(str))
        if not included_keys or not included_keys.issubset(parent_keys):
            raise CorrectionV42Error("IMAGE_COHORT_NOT_SUBSET_OF_ENDPOINT")
        match = included.merge(
            parent[["patient_hash", "event", "time_days"]], on="patient_hash",
            suffixes=("_image", "_endpoint"), validate="one_to_one"
        )
        if not (
            (match["event_image"].astype(int) == match["event_endpoint"].astype(int)).all()
            and np.allclose(match["time_days_image"], match["time_days_endpoint"], atol=1e-8)
        ):
            raise CorrectionV42Error("IMAGE_ENDPOINT_OUTCOME_MISMATCH")
        expected_n = int(correction["branches"][branch]["subject_count"])
        if len(included_keys) != expected_n:
            raise CorrectionV42Error("IMAGE_COHORT_COUNT_MISMATCH")
        parent["image_included"] = parent["patient_hash"].astype(str).isin(included_keys).astype(int)
        selected = parent[parent["image_included"] == 1]
        excluded = parent[parent["image_included"] == 0]
        baseline: dict[str, Any] = {}
        for variable in BASELINE_VARIABLES:
            if variable not in parent.columns:
                raise CorrectionV42Error("BASELINE_VARIABLE_MISSING:" + variable)
            baseline[variable] = {
                "included": _numeric_summary(selected[variable]),
                "excluded": _numeric_summary(excluded[variable]),
                "smd_included_minus_excluded": standardized_mean_difference(
                    selected[variable], excluded[variable]
                ),
            }
        branches[branch] = {
            "parent_population": "A4_ONLY" if branch == "amyloid" else "A4_TREATMENT_AND_PLACEBO",
            "endpoint_parent_n": len(parent),
            "image_included_n": len(selected),
            "image_excluded_n": len(excluded),
            "image_inclusion_fraction": len(selected) / len(parent),
            "included_events": int(selected["event"].sum()),
            "excluded_events": int(excluded["event"].sum()),
            "included_event_fraction": float(selected["event"].mean()),
            "excluded_event_fraction": float(excluded["event"].mean()),
            "included_followup_days": _numeric_summary(selected["time_days"]),
            "excluded_followup_days": _numeric_summary(excluded["time_days"]),
            "baseline_balance": baseline,
            "inclusion_propensity": _propensity_audit(parent, seed=int(seed)),
            "interpretation": "SELECTION_BIAS_DIAGNOSTIC_NOT_OUTCOME_ADJUSTMENT",
        }
    return {
        "schema_version": "4.2",
        "status": "COMPLETED_DESCRIPTIVE_ONLY",
        "science_gate_status": "BLOCKED",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "correction_receipt_sha256": hashlib.sha256(correction_raw).hexdigest(),
        "branches": branches,
        "input_sha256": sources,
    }


def write_selection_flow(
    project_root: str | Path, correction_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    report = build_selection_flow(project_root, correction_path)
    report["source_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["report_sha256"] = sha256_mapping(report)
    output = Path(output_path)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != report:
            raise CorrectionV42Error("SELECTION_FLOW_OUTPUT_CONTRACT_MISMATCH")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    return report
