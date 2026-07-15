from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.util import Surv

from .atomic_io import atomic_write_json, read_json
from .contracts import RunContract
from .evaluation import OuterEvaluationStore, valid_survival_horizons
from .regional_features import FORBIDDEN_FEATURE_TOKENS
from .science_gates import cached_paired_oof_bootstrap, incremental_value_gate
from .splits import build_nested_subject_splits
from .survival_models import select_inner_winner


class RegionalTrainingError(ValueError):
    """Raised when regional survival training cannot satisfy the nested contract."""


def survival_evaluation_grid(
    train_times: np.ndarray, test_times: np.ndarray, *, points: int = 32
) -> np.ndarray:
    train = np.asarray(train_times, dtype=float)
    test = np.asarray(test_times, dtype=float)
    train = train[np.isfinite(train) & (train > 0)]
    test = test[np.isfinite(test) & (test > 0)]
    if train.size == 0 or test.size == 0:
        raise RegionalTrainingError("NO_POSITIVE_FINITE_SURVIVAL_TIME")
    lower = max(float(train.min()), float(test.min()))
    upper = min(float(train.max()), float(test.max()))
    if lower >= upper:
        raise RegionalTrainingError("NO_COMMON_SURVIVAL_SUPPORT")
    epsilon = max((upper - lower) * 1e-6, 1e-6)
    return np.linspace(lower + epsilon, upper - epsilon, num=max(2, int(points)))


def _make_y(frame: pd.DataFrame):
    return Surv.from_arrays(
        event=frame["event"].astype(bool).to_numpy(),
        time=frame["time_days"].astype(float).to_numpy(),
    )


def _administratively_censor(frame: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    censored = frame.copy()
    original_time = censored["time_days"].astype(float)
    censored["event"] = (
        censored["event"].astype(bool) & (original_time < float(cutoff))
    ).astype(int)
    censored["time_days"] = np.minimum(original_time, float(cutoff))
    return censored


def _pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("cox", CoxPHSurvivalAnalysis(alpha=float(alpha), n_iter=300)),
        ]
    )


def _evaluate_model(
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    requested_horizons: Iterable[float],
    min_at_risk: int,
) -> dict[str, Any]:
    raw_upper = min(float(train["time_days"].max()), float(test["time_days"].max()))
    cutoff = raw_upper - max(raw_upper * 1e-6, 1e-6)
    evaluation_train = _administratively_censor(train, cutoff)
    evaluation_test = _administratively_censor(test, cutoff)
    y_train = _make_y(evaluation_train)
    y_test = _make_y(evaluation_test)
    risk = np.asarray(model.predict(test[features]), dtype=float)
    grid = survival_evaluation_grid(
        evaluation_train["time_days"].to_numpy(),
        evaluation_test["time_days"].to_numpy(),
    )
    survival_functions = model.predict_survival_function(test[features])
    survival = np.vstack([function(grid) for function in survival_functions])
    ibs = float(integrated_brier_score(y_train, y_test, survival, grid))
    uno = float(concordance_index_ipcw(y_train, y_test, risk, tau=float(grid[-1]))[0])
    horizons = valid_survival_horizons(
        train_times=train["time_days"].to_numpy(),
        test_times=test["time_days"].to_numpy(),
        requested_horizons=requested_horizons,
        min_at_risk=min_at_risk,
    )
    effective_events = evaluation_test["event"].astype(int).to_numpy()
    effective_times = evaluation_test["time_days"].astype(float).to_numpy()
    original_events = test["event"].astype(int).to_numpy()
    original_times = test["time_days"].astype(float).to_numpy()
    patient_hashes = test["patient_hash"].astype(str).to_numpy()
    oof_predictions = [
        {
            "patient_hash": str(patient_hashes[index]),
            "event": int(effective_events[index]),
            "time_days": float(effective_times[index]),
            "original_event": int(original_events[index]),
            "original_time_days": float(original_times[index]),
            "risk_score": float(risk[index]),
            "survival_probabilities": [float(value) for value in survival[index]],
        }
        for index in range(len(test))
    ]
    return {
        "uno_c_index": uno,
        "integrated_brier_score": ibs,
        "evaluation_grid_days": grid.tolist(),
        "horizons": {str(key): value for key, value in horizons.items()},
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_events": int(train["event"].sum()),
        "test_events": int(test["event"].sum()),
        "administrative_censoring_cutoff_days": cutoff,
        "test_rows_administratively_censored": int(
            np.sum(test["time_days"].astype(float).to_numpy() >= cutoff)
        ),
        "oof_predictions": oof_predictions,
        "censoring_reference": [
            {"event": int(row.event), "time_days": float(row.time_days)}
            for row in evaluation_train[["event", "time_days"]].itertuples(index=False)
        ],
    }


def _atomic_joblib(path: Path, model: Pipeline) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(model, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_alpha(
    frame: pd.DataFrame,
    *,
    features: list[str],
    inner_folds: list[dict[str, Any]],
    alpha_grid: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_subject = frame.set_index("patient_hash", drop=False)
    candidates: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        scores: list[float] = []
        for fold in inner_folds:
            train = by_subject.loc[fold["train_subjects"]]
            validation = by_subject.loc[fold["validation_subjects"]]
            model = _pipeline(alpha)
            model.fit(train[features], _make_y(train))
            score = _evaluate_model(
                model,
                train,
                validation,
                features,
                requested_horizons=[],
                min_at_risk=1,
            )["integrated_brier_score"]
            scores.append(float(score))
        standard_error = (
            float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else 0.0
        )
        candidates.append(
            {
                "candidate_id": f"cox_alpha_{alpha:g}",
                "alpha": float(alpha),
                "mean_ibs": float(np.mean(scores)),
                "standard_error": standard_error,
                "complexity_rank": int(alpha_grid.index(alpha)),
                "inner_fold_ibs": scores,
            }
        )
    return select_inner_winner(candidates), candidates


def run_regional_nested_cv(
    *,
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    output_root: str | Path,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    alpha_grid: list[float],
    requested_horizons: list[float],
    min_at_risk: int,
    branch: str = "tau",
) -> dict[str, Any]:
    if branch not in {"amyloid", "tau", "paired_at"}:
        raise RegionalTrainingError(f"UNSUPPORTED_REGIONAL_BRANCH:{branch}")
    required = {"patient_hash", "event", "time_days", "treatment_arm"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RegionalTrainingError(f"MISSING_TRAINING_COLUMNS:{missing}")
    if frame["patient_hash"].duplicated().any():
        raise RegionalTrainingError("DUPLICATE_TRAINING_SUBJECT")
    for variant, features in feature_sets.items():
        missing_features = sorted(set(features).difference(frame.columns))
        if missing_features:
            raise RegionalTrainingError(f"MISSING_FEATURES:{variant}:{missing_features}")
        for feature in features:
            if any(token in feature.casefold() for token in FORBIDDEN_FEATURE_TOKENS):
                raise RegionalTrainingError(f"FORBIDDEN_MODEL_FEATURE:{feature}")
    split_rows = [
        {
            "subject_key": row.patient_hash,
            "event": int(row.event),
            "treatment_arm": row.treatment_arm,
        }
        for row in frame.itertuples(index=False)
    ]
    splits = build_nested_subject_splits(
        split_rows, outer_folds=outer_folds, inner_folds=inner_folds, seed=seed
    )
    root = Path(output_root)
    atomic_write_json(root / "splits" / f"{branch}_split_contract.json", splits)
    training_contract = RunContract.from_mapping(
        {
            "split_digest": splits["split_digest"],
            "feature_sets": feature_sets,
            "alpha_grid": alpha_grid,
            "seed": seed,
            "model_family": "CoxPHSurvivalAnalysis",
            "branch": branch,
        }
    )
    by_subject = frame.set_index("patient_hash", drop=False)
    variant_reports: dict[str, Any] = {}
    all_selected: dict[str, list[dict[str, Any]]] = {}
    for variant, features in feature_sets.items():
        fold_reports: list[dict[str, Any]] = []
        selections: list[dict[str, Any]] = []
        for outer in splits["outer_folds"]:
            train = by_subject.loc[outer["train_subjects"]]
            test = by_subject.loc[outer["test_subjects"]]
            fold_directory = root / "development" / branch / variant / f"fold_{outer['fold']}"
            checkpoint = fold_directory / "checkpoint.joblib"
            selection_path = fold_directory / "inner_selection.json"
            if checkpoint.exists() and selection_path.exists():
                selection = read_json(selection_path)
                if selection.get("run_contract_sha256") != training_contract.digest:
                    raise RegionalTrainingError("REGIONAL_CHECKPOINT_RUN_CONTRACT_MISMATCH")
                checkpoint_digest = _sha256_file(checkpoint)
                if selection.get("checkpoint_sha256") != checkpoint_digest:
                    raise RegionalTrainingError("REGIONAL_CHECKPOINT_HASH_MISMATCH")
                winner = selection["winner"]
                candidates = selection["candidates"]
                alpha = float(winner["alpha"])
                model = joblib.load(checkpoint)
            elif checkpoint.exists() or selection_path.exists():
                raise RegionalTrainingError("INCOMPLETE_REGIONAL_FOLD_CHECKPOINT")
            else:
                winner, candidates = _select_alpha(
                    train,
                    features=features,
                    inner_folds=outer["inner_folds"],
                    alpha_grid=alpha_grid,
                )
                alpha = float(winner["alpha"])
                model = _pipeline(alpha)
                model.fit(train[features], _make_y(train))
                checkpoint_digest = _atomic_joblib(checkpoint, model)
                atomic_write_json(
                    selection_path,
                    {
                        "winner": winner,
                        "candidates": candidates,
                        "source": "inner_validation_ibs_only",
                        "run_contract_sha256": training_contract.digest,
                        "checkpoint_sha256": checkpoint_digest,
                    },
                )
            store = OuterEvaluationStore(
                root / "outer_evaluation" / branch / variant,
                fold=int(outer["fold"]),
                run_contract_digest=training_contract.digest,
            )
            metrics = store.evaluate_once(
                checkpoint_digest=checkpoint_digest,
                evaluator=lambda model=model, train=train, test=test, features=features: _evaluate_model(
                    model,
                    train,
                    test,
                    features,
                    requested_horizons,
                    min_at_risk,
                ),
            )
            selections.append({"fold": int(outer["fold"]), "alpha": alpha, "winner": winner})
            fold_reports.append(
                {
                    "fold": int(outer["fold"]),
                    "status": "COMPLETED",
                    "selected_alpha": alpha,
                    **metrics,
                }
            )
        variant_reports[variant] = {
            "outer_folds_completed": len(fold_reports),
            "mean_uno_c_index": float(np.mean([row["uno_c_index"] for row in fold_reports])),
            "mean_integrated_brier_score": float(
                np.mean([row["integrated_brier_score"] for row in fold_reports])
            ),
            "folds": fold_reports,
        }
        all_selected[variant] = selections
    locked = {
        "model_family": "CoxPHSurvivalAnalysis",
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "seed": int(seed),
        "split_digest": splits["split_digest"],
        "run_contract_sha256": training_contract.digest,
        "selection_source": "inner_validation_ibs_only",
        "fold_selected_configs": all_selected,
        "feature_sets": feature_sets,
    }
    atomic_write_json(root / "locked" / f"{branch}_locked_config.json", locked)
    report = {
        "status": "OUTER_EVALUATION_DONE",
        "science_gate_status": "NOT_EVALUATED",
        "claim_status": "EXPLORATORY",
        "variants": variant_reports,
        "locked_config": locked,
    }
    atomic_write_json(root / "reports" / f"{branch}_regional_development.json", report)
    return report


def run_tau_regional_development_from_locked_endpoint(
    *, config: dict[str, Any], output_root: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(output_root)
    endpoint_contract_path = root / "endpoints" / "tau_endpoint_contract.json"
    if not endpoint_contract_path.is_file():
        raise RegionalTrainingError("LOCKED_TAU_ENDPOINT_REQUIRED")
    endpoint_contract = read_json(endpoint_contract_path)
    if not endpoint_contract.get("locked"):
        raise RegionalTrainingError("TAU_ENDPOINT_NOT_LOCKED")
    endpoint_path = root / "endpoints" / str(endpoint_contract["endpoint_path"])
    if endpoint_contract.get("storage_format") == "parquet":
        endpoint = pd.read_parquet(endpoint_path)
    elif endpoint_contract.get("storage_format") == "csv_fallback":
        endpoint = pd.read_csv(endpoint_path)
    else:
        raise RegionalTrainingError("UNSUPPORTED_ENDPOINT_STORAGE_FORMAT")
    regional_path = Path(project_root) / config["paths"]["tau_regional_features"]
    regional = pd.read_csv(regional_path)
    if regional["patient_hash"].duplicated().any():
        raise RegionalTrainingError("DUPLICATE_REGIONAL_FEATURE_SUBJECT")
    frame = endpoint.merge(regional, on="patient_hash", how="inner", validate="one_to_one")
    if len(frame) != len(endpoint):
        raise RegionalTrainingError(
            f"REGIONAL_ENDPOINT_MEMBERSHIP_MISMATCH:endpoint={len(endpoint)}:joined={len(frame)}"
        )
    base_candidates = [
        "treatment_active",
        "age",
        "sex_binary",
        "education_years",
        "apoe4_carrier",
        "baseline_amyloid_centiloid",
        "baseline_pacc_raw",
        "baseline_fctotal96",
        "baseline_ldeltotal",
        "baseline_digittotal",
        "baseline_mmscore",
        "baseline_cdrsb",
    ]
    base_features = [feature for feature in base_candidates if feature in frame.columns]
    pet_features = [feature for feature in regional.columns if feature != "patient_hash"]
    feature_sets = {
        "clinical_cognition": base_features,
        "clinical_cognition_pet": [*base_features, *pet_features],
    }
    report = run_regional_nested_cv(
        frame=frame,
        feature_sets=feature_sets,
        output_root=root,
        outer_folds=int(config["validation"]["outer_folds"]),
        inner_folds=int(config["validation"]["inner_folds"]),
        seed=int(config["seed"]),
        alpha_grid=[float(value) for value in config["hpo"]["cox"]["alpha"]],
        requested_horizons=[
            float(value) for value in config["validation"]["requested_horizons_days"]
        ],
        min_at_risk=int(config["validation"]["minimum_at_risk_per_horizon"]),
        branch="tau",
    )
    base = report["variants"]["clinical_cognition"]
    augmented = report["variants"]["clinical_cognition_pet"]
    bootstrap = cached_paired_oof_bootstrap(
        base,
        augmented,
        output_path=root / "reports" / "tau_paired_bootstrap.json",
        iterations=int(config["validation"]["bootstrap_iterations"]),
        seed=int(config["seed"]) + 1000,
    )
    science_gate = incremental_value_gate(
        delta_ibs_ci=tuple(bootstrap["delta_ibs_95ci"]),
        delta_cindex_ci=tuple(bootstrap["delta_cindex_95ci"]),
        outer_folds_completed=int(augmented["outer_folds_completed"]),
        expected_outer_folds=int(config["validation"]["outer_folds"]),
        scanner_sensitivity_passed=False,
    )
    preliminary = {
        "status": "PAIRED_BOOTSTRAP_COMPLETED_SCANNER_SENSITIVITY_PENDING",
        "science_gate_status": science_gate["science_gate_status"],
        "claim_status": science_gate["claim_status"],
        "delta_mean_uno_c_index": augmented["mean_uno_c_index"] - base["mean_uno_c_index"],
        "delta_mean_integrated_brier_score": (
            augmented["mean_integrated_brier_score"]
            - base["mean_integrated_brier_score"]
        ),
        "regional_feature_count": len(pet_features),
        "endpoint_membership_count": len(frame),
        "required_before_positive_claim": [
            "SCANNER_SENSITIVITY_PASS",
            "EXTERNAL_VALIDATION",
        ],
    }
    atomic_write_json(root / "reports" / "tau_added_value_preliminary.json", preliminary)
    report["preliminary_added_value"] = preliminary
    report["paired_bootstrap"] = bootstrap
    report["science_gate"] = science_gate
    report["science_gate_status"] = science_gate["science_gate_status"]
    report["claim_status"] = science_gate["claim_status"]
    atomic_write_json(root / "reports" / "tau_regional_development.json", report)
    return report


def _read_locked_endpoint(root: Path, branch: str) -> pd.DataFrame:
    contract_path = root / "endpoints" / f"{branch}_endpoint_contract.json"
    if not contract_path.is_file():
        raise RegionalTrainingError(f"LOCKED_{branch.upper()}_ENDPOINT_REQUIRED")
    contract = read_json(contract_path)
    if not contract.get("locked"):
        raise RegionalTrainingError(f"{branch.upper()}_ENDPOINT_NOT_LOCKED")
    endpoint_path = root / "endpoints" / str(contract["endpoint_path"])
    if not endpoint_path.is_file() or _sha256_file(endpoint_path) != contract.get("endpoint_sha256"):
        raise RegionalTrainingError(f"{branch.upper()}_ENDPOINT_HASH_MISMATCH")
    if contract.get("storage_format") == "parquet":
        return pd.read_parquet(endpoint_path)
    if contract.get("storage_format") == "csv_fallback":
        return pd.read_csv(endpoint_path)
    raise RegionalTrainingError("UNSUPPORTED_ENDPOINT_STORAGE_FORMAT")


def run_amyloid_regional_development_from_locked_endpoint(
    *, config: dict[str, Any], output_root: str | Path
) -> dict[str, Any]:
    root = Path(output_root)
    endpoint = _read_locked_endpoint(root, "amyloid")
    if "cohort" not in endpoint.columns:
        raise RegionalTrainingError("AMYLOID_ENDPOINT_COHORT_REQUIRED")
    development = endpoint.loc[endpoint["cohort"].astype(str) == "A4"].copy()
    transport = endpoint.loc[endpoint["cohort"].astype(str) == "LEARN"].copy()
    if development.empty:
        raise RegionalTrainingError("NO_A4_AMYLOID_DEVELOPMENT_SUBJECTS")
    if "baseline_amyloid_centiloid" not in development.columns:
        raise RegionalTrainingError("CONTINUOUS_CENTILOID_REQUIRED")
    base_candidates = [
        "treatment_active",
        "age",
        "sex_binary",
        "education_years",
        "apoe4_carrier",
        "baseline_pacc_raw",
        "baseline_fctotal96",
        "baseline_ldeltotal",
        "baseline_digittotal",
        "baseline_mmscore",
        "baseline_cdrsb",
    ]
    base_features = [feature for feature in base_candidates if feature in development.columns]
    feature_sets = {
        "clinical_cognition": base_features,
        "clinical_cognition_amyloid": [*base_features, "baseline_amyloid_centiloid"],
    }
    report = run_regional_nested_cv(
        frame=development,
        feature_sets=feature_sets,
        output_root=root,
        outer_folds=int(config["validation"]["outer_folds"]),
        inner_folds=int(config["validation"]["inner_folds"]),
        seed=int(config["seed"]),
        alpha_grid=[float(value) for value in config["hpo"]["cox"]["alpha"]],
        requested_horizons=[
            float(value) for value in config["validation"]["requested_horizons_days"]
        ],
        min_at_risk=int(config["validation"]["minimum_at_risk_per_horizon"]),
        branch="amyloid",
    )
    base = report["variants"]["clinical_cognition"]
    augmented = report["variants"]["clinical_cognition_amyloid"]
    bootstrap = cached_paired_oof_bootstrap(
        base,
        augmented,
        output_path=root / "reports" / "amyloid_paired_bootstrap.json",
        iterations=int(config["validation"]["bootstrap_iterations"]),
        seed=int(config["seed"]) + 2000,
    )
    science_gate = incremental_value_gate(
        delta_ibs_ci=tuple(bootstrap["delta_ibs_95ci"]),
        delta_cindex_ci=tuple(bootstrap["delta_cindex_95ci"]),
        outer_folds_completed=int(augmented["outer_folds_completed"]),
        expected_outer_folds=int(config["validation"]["outer_folds"]),
        scanner_sensitivity_passed=False,
    )
    report.update(
        {
            "development_cohort": "A4",
            "development_membership_count": int(len(development)),
            "internal_transport_cohort": "LEARN",
            "internal_transport_membership_count": int(len(transport)),
            "internal_transport_status": "PENDING_LOCKED_GLOBAL_MODEL",
            "science_gate_status": science_gate["science_gate_status"],
            "claim_status": science_gate["claim_status"],
            "paired_bootstrap": bootstrap,
            "science_gate": science_gate,
            "preliminary_added_value": {
                "status": "PAIRED_BOOTSTRAP_COMPLETED_SCANNER_SENSITIVITY_PENDING",
                "delta_mean_uno_c_index": (
                    augmented["mean_uno_c_index"] - base["mean_uno_c_index"]
                ),
                "delta_mean_integrated_brier_score": (
                    augmented["mean_integrated_brier_score"]
                    - base["mean_integrated_brier_score"]
                ),
                "same_pet_derived_target_used": False,
            },
        }
    )
    atomic_write_json(root / "reports" / "amyloid_regional_development.json", report)
    return report


def run_paired_at_regional_development_from_locked_endpoint(
    *, config: dict[str, Any], output_root: str | Path
) -> dict[str, Any]:
    root = Path(output_root)
    endpoint = _read_locked_endpoint(root, "paired_at")
    base_candidates = [
        "treatment_active",
        "age",
        "sex_binary",
        "education_years",
        "apoe4_carrier",
        "baseline_pacc_raw",
        "baseline_fctotal96",
        "baseline_ldeltotal",
        "baseline_digittotal",
        "baseline_mmscore",
        "baseline_cdrsb",
    ]
    base = [feature for feature in base_candidates if feature in endpoint.columns]
    amyloid = [*base, "baseline_amyloid_centiloid"]
    tau_features = sorted(
        column for column in endpoint.columns if str(column).startswith("tau_persi__")
    )
    if not tau_features:
        raise RegionalTrainingError("NO_PAIRED_AT_TAU_REGIONAL_FEATURES")
    feature_sets = {
        "clinical_cognition": base,
        "clinical_cognition_amyloid": amyloid,
        "clinical_cognition_amyloid_tau": [*amyloid, *tau_features],
    }
    report = run_regional_nested_cv(
        frame=endpoint,
        feature_sets=feature_sets,
        output_root=root,
        outer_folds=int(config["validation"]["outer_folds"]),
        inner_folds=int(config["validation"]["inner_folds"]),
        seed=int(config["seed"]),
        alpha_grid=[float(value) for value in config["hpo"]["cox"]["alpha"]],
        requested_horizons=[
            float(value) for value in config["validation"]["requested_horizons_days"]
        ],
        min_at_risk=int(config["validation"]["minimum_at_risk_per_horizon"]),
        branch="paired_at",
    )
    bootstrap = cached_paired_oof_bootstrap(
        report["variants"]["clinical_cognition_amyloid"],
        report["variants"]["clinical_cognition_amyloid_tau"],
        output_path=root / "reports" / "paired_at_tau_beyond_amyloid_bootstrap.json",
        iterations=int(config["validation"]["bootstrap_iterations"]),
        seed=int(config["seed"]) + 4000,
    )
    report.update(
        {
            "tau_beyond_amyloid_bootstrap": bootstrap,
            "science_gate_status": "BLOCKED",
            "claim_status": "EXPLORATORY",
            "blockers": [
                "PAIRED_AT_POWER_SIMULATION_NOT_COMPLETED",
                "TAU_SCANNER_SENSITIVITY_INSUFFICIENT_EVENT_SUPPORT",
                "QUANTITATIVE_PET_FEATURE_PROVENANCE_NOT_FORMALLY_VALIDATED",
            ],
        }
    )
    atomic_write_json(root / "reports" / "paired_at_regional_development.json", report)
    return report
