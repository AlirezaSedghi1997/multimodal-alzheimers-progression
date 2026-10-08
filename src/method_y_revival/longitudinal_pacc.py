from __future__ import annotations

import math
import hashlib
import os
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping


class LongitudinalPACCError(RuntimeError):
    """Raised when the prespecified longitudinal PACC analysis cannot be fit."""


def _key(cohort: Any, subject: Any) -> tuple[str, str]:
    return str(cohort or "").strip().upper(), str(subject or "").strip().upper()


def _visit_code(value: Any) -> str:
    text = str(value or "").strip()
    try:
        return f"{int(float(text)):03d}"
    except ValueError:
        return text.upper()


def exact_pacc_visit_days_from_adqs(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[tuple[str, str, str], float], dict[str, Any]]:
    component_codes = {"FCSRT96", "LMIIA", "DIGIT", "MMSE"}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    used = 0
    for row in rows:
        if str(row.get("QSTESTCD") or "").strip().upper() not in component_codes:
            continue
        try:
            day = float(row.get("QSDTC_DAYS_T0"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(day):
            continue
        subject = _key(row.get("SUBSTUDY"), row.get("BID"))
        visit = _visit_code(row.get("VISITCD"))
        if not all(subject) or not visit:
            continue
        grouped[(*subject, visit)].append(day)
        used += 1
    dates = {key: float(np.median(values)) for key, values in grouped.items() if values}
    return dates, {
        "status": "EXACT_PACC_VISIT_DATES_DERIVED",
        "component_codes": sorted(component_codes),
        "component_rows_used": int(used),
        "visit_dates_derived": len(dates),
        "date_rule": "MEDIAN_EXACT_QSDTC_DAYS_T0_ACROSS_PACC_COMPONENTS",
    }


def build_longitudinal_pacc_rows(
    *,
    pacc_rows: Iterable[dict[str, Any]],
    exact_visit_days: dict[tuple[str, str, str], float],
    pet_index_days: dict[tuple[str, str], float],
    covariates_by_key: dict[tuple[str, str], dict[str, Any]],
    baseline_window_days: float,
    maximum_followup_days: float,
    minimum_measurements: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counters: Counter[str] = Counter()
    by_subject: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in pacc_rows:
        subject = _key(source.get("SUBSTUDY"), source.get("BID"))
        visit = _visit_code(source.get("VISCODE"))
        try:
            pacc = float(source.get("PACC.raw"))
        except (TypeError, ValueError):
            counters["missing_pacc_measurements_excluded"] += 1
            continue
        if not math.isfinite(pacc):
            counters["missing_pacc_measurements_excluded"] += 1
            continue
        if subject not in pet_index_days or subject not in covariates_by_key:
            counters["measurements_without_locked_pet_endpoint_excluded"] += 1
            continue
        visit_key = (*subject, visit)
        if visit_key not in exact_visit_days:
            counters["measurements_without_exact_date_excluded"] += 1
            continue
        elapsed = float(exact_visit_days[visit_key]) - float(pet_index_days[subject])
        if elapsed < 0:
            counters["pre_pet_measurements_excluded"] += 1
            continue
        if elapsed > float(maximum_followup_days):
            counters["measurements_beyond_followup_excluded"] += 1
            continue
        covariates = covariates_by_key[subject]
        by_subject[subject].append(
            {
                "patient_hash": str(covariates["patient_hash"]),
                "cohort": str(covariates["cohort"]),
                "time_days_from_pet": float(elapsed),
                "time_years": float(elapsed / 365.25),
                "pacc_raw": pacc,
                "age": covariates.get("age"),
                "sex_binary": covariates.get("sex_binary"),
                "education_years": covariates.get("education_years"),
                "treatment_active": covariates.get("treatment_active"),
                "baseline_amyloid_centiloid": covariates.get(
                    "baseline_amyloid_centiloid"
                ),
            }
        )

    output: list[dict[str, Any]] = []
    for subject, rows in by_subject.items():
        rows.sort(key=lambda row: float(row["time_days_from_pet"]))
        if len(rows) < int(minimum_measurements):
            counters["subjects_excluded_insufficient_measurements"] += 1
            continue
        if float(rows[0]["time_days_from_pet"]) > float(baseline_window_days):
            counters["subjects_excluded_no_near_pet_baseline"] += 1
            continue
        output.extend(rows)
    output.sort(key=lambda row: (str(row["patient_hash"]), float(row["time_days_from_pet"])))
    audit = {
        "status": "LONGITUDINAL_PACC_ENDPOINT_BUILT",
        "eligible_subject_count": len({row["patient_hash"] for row in output}),
        "eligible_measurement_count": len(output),
        **{key: int(value) for key, value in sorted(counters.items())},
        "raw_identifiers_in_output": False,
        "exact_measurement_dates_required": True,
    }
    return output, audit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_longitudinal_pacc_from_sources(
    *,
    raw_root: str | Path,
    pacc_relative: str,
    adqs_relative: str,
    amyloid_suvr_relative: str,
    parent_endpoint: pd.DataFrame,
    parent_endpoint_sha256: str,
    output_root: str | Path,
    baseline_window_days: float,
    maximum_followup_days: float,
    minimum_measurements: int,
) -> dict[str, Any]:
    from scripts.build_method_y_tau_progression import (
        patient_hash_for_key,
        read_csv_rows,
    )

    from .endpoint_builder import load_first_amyloid_scans

    raw = Path(raw_root)
    source_paths = {
        "pacc": raw / pacc_relative,
        "adqs": raw / adqs_relative,
        "amyloid_suvr": raw / amyloid_suvr_relative,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise LongitudinalPACCError("MISSING_LONGITUDINAL_PACC_SOURCE:" + "|".join(missing))
    source_sha256 = {name: _sha256_file(path) for name, path in source_paths.items()}
    input_contract_sha256 = sha256_mapping(
        {
            "source_sha256": source_sha256,
            "parent_endpoint_sha256": parent_endpoint_sha256,
            "parent_membership": sorted(parent_endpoint["patient_hash"].astype(str)),
            "baseline_window_days": float(baseline_window_days),
            "maximum_followup_days": float(maximum_followup_days),
            "minimum_measurements": int(minimum_measurements),
            "date_rule": "MEDIAN_EXACT_QSDTC_DAYS_T0_ACROSS_PACC_COMPONENTS",
        }
    )
    root = Path(output_root)
    contract_path = root / "endpoints" / "longitudinal_pacc_endpoint_contract.json"
    endpoint_path = root / "endpoints" / "longitudinal_pacc_endpoint.csv"
    audit_path = root / "endpoints" / "longitudinal_pacc_endpoint_audit.json"
    if contract_path.is_file():
        from .atomic_io import read_json

        existing = read_json(contract_path)
        if existing.get("input_contract_sha256") != input_contract_sha256:
            raise LongitudinalPACCError("LONGITUDINAL_PACC_ENDPOINT_CONTRACT_MISMATCH")
        if not endpoint_path.is_file() or _sha256_file(endpoint_path) != existing.get(
            "endpoint_sha256"
        ):
            raise LongitudinalPACCError("LONGITUDINAL_PACC_ENDPOINT_HASH_MISMATCH")
        return {
            "status": "LONGITUDINAL_PACC_ENDPOINT_LOCKED",
            "contract": existing,
            "endpoint_path": endpoint_path,
        }

    first_scans, scan_audit = load_first_amyloid_scans(
        read_csv_rows(source_paths["amyloid_suvr"])
    )
    exact_dates, date_audit = exact_pacc_visit_days_from_adqs(
        read_csv_rows(source_paths["adqs"])
    )
    parent_by_hash = parent_endpoint.set_index("patient_hash", drop=False)
    pet_index_days: dict[tuple[str, str], float] = {}
    covariates_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_key, scan in first_scans.items():
        normalized = _key(*raw_key)
        patient_hash = patient_hash_for_key(raw_key)
        if patient_hash not in parent_by_hash.index:
            continue
        source = parent_by_hash.loc[patient_hash]
        pet_index_days[normalized] = float(scan["scan_date"])
        covariates_by_key[normalized] = {
            "patient_hash": patient_hash,
            "cohort": source["cohort"],
            "age": source.get("age"),
            "sex_binary": source.get("sex_binary"),
            "education_years": source.get("education_years"),
            "treatment_active": source.get("treatment_active"),
            "baseline_amyloid_centiloid": source.get("baseline_amyloid_centiloid"),
        }
    rows, endpoint_audit = build_longitudinal_pacc_rows(
        pacc_rows=read_csv_rows(source_paths["pacc"]),
        exact_visit_days=exact_dates,
        pet_index_days=pet_index_days,
        covariates_by_key=covariates_by_key,
        baseline_window_days=float(baseline_window_days),
        maximum_followup_days=float(maximum_followup_days),
        minimum_measurements=int(minimum_measurements),
    )
    if not rows:
        raise LongitudinalPACCError("NO_ELIGIBLE_LONGITUDINAL_PACC_ROWS")
    frame = pd.DataFrame(rows)
    _atomic_csv(endpoint_path, frame)
    contract = {
        "status": "LONGITUDINAL_PACC_ENDPOINT_LOCKED",
        "locked": True,
        "storage_format": "csv",
        "endpoint_path": endpoint_path.name,
        "endpoint_sha256": _sha256_file(endpoint_path),
        "input_contract_sha256": input_contract_sha256,
        "source_sha256": source_sha256,
        "parent_endpoint_sha256": parent_endpoint_sha256,
        "row_count": int(len(frame)),
        "subject_count": int(frame["patient_hash"].nunique()),
        "raw_identifiers_present": False,
    }
    audit = {
        "status": "LONGITUDINAL_PACC_ENDPOINT_LOCKED",
        "scan_linkage_audit": scan_audit,
        "exact_date_audit": date_audit,
        "endpoint_audit": endpoint_audit,
        "contract_sha256": sha256_mapping(contract),
    }
    atomic_write_json(contract_path, contract)
    atomic_write_json(audit_path, audit)
    return {
        "status": "LONGITUDINAL_PACC_ENDPOINT_LOCKED",
        "contract": contract,
        "audit": audit,
        "endpoint_path": endpoint_path,
    }


def _fit_mixed_model(frame: pd.DataFrame, formula: str):
    last_error: Exception | None = None
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    frame,
                    groups=frame["patient_hash"],
                    re_formula="~time_years",
                )
                result = model.fit(reml=False, method=method, maxiter=1000, disp=False)
                covariance = np.asarray(result.cov_re, dtype=float)
                valid = (bool(result.converged) and np.isfinite(result.params).all()
                         and np.isfinite(result.bse).all() and np.isfinite(result.llf)
                         and np.isfinite(covariance).all()
                         and np.linalg.eigvalsh(covariance).min() >= -1e-10)
                if valid:
                    result.method_y_fit_diagnostics = {
                        "optimizer": method,
                        "warning_categories": sorted({w.category.__name__ for w in caught}),
                        "random_effect_min_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
                        "boundary_or_singular_warning": any(
                            'boundary' in str(w.message).lower() or 'singular' in str(w.message).lower()
                            for w in caught),
                    }
                    return result
                last_error = LongitudinalPACCError(f"MIXED_MODEL_NOT_CONVERGED:{method}")
        except (ValueError, np.linalg.LinAlgError) as exc:
            last_error = exc
    raise LongitudinalPACCError(f"MIXED_MODEL_FIT_FAILED:{last_error}")


def _model_summary(result, *, interaction: str | None = None) -> dict[str, Any]:
    residual = np.asarray(result.resid, dtype=float)
    if not np.isfinite(residual).all():
        raise LongitudinalPACCError("NONFINITE_MIXED_MODEL_RESIDUALS")
    summary: dict[str, Any] = {
        "converged": bool(result.converged),
        "n_observations": int(result.nobs),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "log_likelihood": float(result.llf),
        "fit_diagnostics": getattr(result, "method_y_fit_diagnostics", {}),
        "residual_diagnostics": {
            "mean": float(residual.mean()),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "quantiles_01_50_99": np.quantile(residual, [.01, .5, .99]).tolist(),
            "scope": "DESCRIPTIVE_NOT_A_MISSING_AT_RANDOM_TEST",
        },
    }
    if interaction is not None:
        if interaction not in result.params:
            raise LongitudinalPACCError("PRESPECIFIED_PET_TIME_INTERACTION_MISSING")
        interval = result.conf_int().loc[interaction]
        summary.update(
            interaction_parameter=interaction,
            interaction_estimate=float(result.params[interaction]),
            interaction_standard_error=float(result.bse[interaction]),
            interaction_95ci=[float(interval.iloc[0]), float(interval.iloc[1])],
            interaction_p_value=float(result.pvalues[interaction]),
        )
    return summary


def fit_longitudinal_pacc_models(
    *, endpoint: pd.DataFrame, output_root: str | Path, write_report: bool = True,
    slope_covariate_sensitivity: bool = False,
) -> dict[str, Any]:
    required = {
        "patient_hash",
        "cohort",
        "time_years",
        "pacc_raw",
        "age",
        "sex_binary",
        "education_years",
        "treatment_active",
        "baseline_amyloid_centiloid",
    }
    missing = sorted(required - set(endpoint.columns))
    if missing:
        raise LongitudinalPACCError(f"MISSING_LONGITUDINAL_PACC_COLUMNS:{missing}")
    frame = endpoint.loc[endpoint["cohort"].astype(str) == "A4"].copy()
    initial_subjects, initial_rows = int(frame["patient_hash"].nunique()), len(frame)
    missing_counts = {k: int(frame[k].isna().sum()) for k in sorted(required - {"patient_hash"})}
    frame = frame.dropna(subset=sorted(required - {"cohort"}))
    if not np.isfinite(frame[["time_years", "pacc_raw", "age", "sex_binary",
                              "education_years", "treatment_active",
                              "baseline_amyloid_centiloid"]].to_numpy(dtype=float)).all():
        raise LongitudinalPACCError("NONFINITE_LONGITUDINAL_PACC_INPUT")
    if (frame["time_years"] < 0).any() or (frame["time_years"] > 4.5).any():
        raise LongitudinalPACCError("PACC_FOLLOWUP_OUTSIDE_LOCKED_WINDOW")
    if frame.duplicated(["patient_hash", "time_years"]).any():
        raise LongitudinalPACCError("DUPLICATE_PATIENT_TIME_MEASUREMENT")
    invariant = ["age", "sex_binary", "education_years", "treatment_active",
                 "baseline_amyloid_centiloid"]
    if (frame.groupby("patient_hash")[invariant].nunique(dropna=False) > 1).any().any():
        raise LongitudinalPACCError("BASELINE_COVARIATE_VARIES_WITHIN_SUBJECT")
    counts = frame.groupby("patient_hash").size()
    frame = frame.loc[frame["patient_hash"].isin(counts[counts >= 2].index)].copy()
    if frame["patient_hash"].nunique() < 30:
        raise LongitudinalPACCError("INSUFFICIENT_LONGITUDINAL_PACC_SUBJECTS")
    subject_pet = frame.groupby("patient_hash")["baseline_amyloid_centiloid"].first()
    pet_sd = float(subject_pet.std(ddof=0))
    if not math.isfinite(pet_sd) or pet_sd <= 0:
        raise LongitudinalPACCError("NONVARIABLE_AMYLOID_PREDICTOR")
    pet_mean = float(subject_pet.mean())
    frame["pet_z"] = (frame["baseline_amyloid_centiloid"] - pet_mean) / pet_sd

    clinical_formula = (
        "pacc_raw ~ time_years + age + sex_binary + education_years + "
        "treatment_active + time_years:treatment_active"
    )
    if slope_covariate_sensitivity:
        clinical_formula += " + time_years:age + time_years:sex_binary + time_years:education_years"
    amyloid_formula = clinical_formula + " + pet_z + time_years:pet_z"
    clinical = _fit_mixed_model(frame, clinical_formula)
    amyloid = _fit_mixed_model(frame, amyloid_formula)
    report = {
        "status": "LONGITUDINAL_PACC_MODELS_COMPLETED",
        "science_gate_status": "FAIL",
        "claim_status": "EXPLORATORY",
        "causal_claim_allowed": False,
        "analysis_role": "POST_HOC_EXECUTION_OF_PRESPECIFIED_SECONDARY_MODEL_FORM",
        "missingness_audit": {"initial_subjects": initial_subjects,
                              "initial_measurements": initial_rows,
                              "missing_measurements_by_column": missing_counts,
                              "assumption": "MAR_AND_CORRECT_LINEAR_MEAN_NOT_VERIFIED"},
        "primary_parameter": "time_years:pet_z",
        "model_formula": amyloid_formula,
        "slope_covariate_sensitivity": bool(slope_covariate_sensitivity),
        "subject_count": int(frame["patient_hash"].nunique()),
        "measurement_count": int(len(frame)),
        "pet_standardization": {
            "fit_cohort": "A4_ONLY",
            "mean": pet_mean,
            "standard_deviation": pet_sd,
        },
        "clinical_model": _model_summary(clinical),
        "amyloid_model": _model_summary(amyloid, interaction="time_years:pet_z"),
        "blockers": [
            "AMYLOID_SCANNER_ROBUSTNESS_NOT_PASSED",
        ],
        "external_validation": {
            "status": "EXTERNAL_UNAVAILABLE_NONBLOCKING",
            "required_for_internal_completion": False,
            "external_generalization_claim_allowed": False,
        },
    }
    report["report_sha256"] = sha256_mapping(report)
    if write_report:
        atomic_write_json(Path(output_root) / "reports" / "longitudinal_pacc_results.json", report)
    return report
