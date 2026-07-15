from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .atomic_io import atomic_write_json, read_json
from .contracts import RunContract
from .evaluation import OuterEvaluationStore
from .regional_features import FORBIDDEN_FEATURE_TOKENS
from .regional_training import (
    _atomic_joblib,
    _evaluate_model,
    _make_y,
    _pipeline,
    _select_alpha,
    _sha256_file,
)
from .scanner_audit import scanner_sensitivity_feasibility
from .science_gates import cached_paired_oof_bootstrap


class ScannerSensitivityError(ValueError):
    """Raised when a leave-manufacturer-out contract cannot be evaluated safely."""


def _inner_memberships(frame: pd.DataFrame, *, folds: int, seed: int) -> list[dict[str, Any]]:
    labels = (
        frame["event"].astype(int).astype(str)
        + "|"
        + frame["treatment_arm"].astype(str)
    )
    minimum_class = min(Counter(labels).values())
    if minimum_class < int(folds):
        raise ScannerSensitivityError(
            f"INSUFFICIENT_INNER_STRATUM_SUPPORT:required={folds}:minimum={minimum_class}"
        )
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    memberships: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(np.arange(len(frame)), labels)
    ):
        memberships.append(
            {
                "fold": fold,
                "train_subjects": frame.iloc[train_index]["patient_hash"].astype(str).tolist(),
                "validation_subjects": frame.iloc[validation_index]["patient_hash"].astype(str).tolist(),
            }
        )
    return memberships


def run_leave_manufacturer_out_sensitivity(
    *,
    frame: pd.DataFrame,
    linkage_rows: Iterable[dict[str, Any]],
    feature_sets: dict[str, list[str]],
    base_variant: str,
    augmented_variant: str,
    output_root: str | Path,
    branch: str,
    inner_folds: int,
    seed: int,
    alpha_grid: list[float],
    requested_horizons: list[float],
    min_at_risk: int,
    bootstrap_iterations: int,
    minimum_subjects: int,
    minimum_events: int,
    minimum_manufacturers: int,
) -> dict[str, Any]:
    if branch not in {"amyloid", "tau"}:
        raise ScannerSensitivityError(f"UNSUPPORTED_SCANNER_BRANCH:{branch}")
    if base_variant not in feature_sets or augmented_variant not in feature_sets:
        raise ScannerSensitivityError("MATCHED_VARIANTS_REQUIRED")
    for variant, features in feature_sets.items():
        missing = sorted(set(features).difference(frame.columns))
        if missing:
            raise ScannerSensitivityError(f"MISSING_FEATURES:{variant}:{missing}")
        for feature in features:
            if any(token in feature.casefold() for token in FORBIDDEN_FEATURE_TOKENS):
                raise ScannerSensitivityError(f"FORBIDDEN_MODEL_FEATURE:{feature}")
    linkage = list(linkage_rows)
    feasibility = scanner_sensitivity_feasibility(
        frame,
        linkage,
        development_cohort="A4" if branch == "amyloid" else None,
        minimum_subjects=minimum_subjects,
        minimum_events=minimum_events,
        minimum_manufacturers=minimum_manufacturers,
    )
    if feasibility["status"] != "READY_FOR_LEAVE_MANUFACTURER_OUT":
        return {
            "status": feasibility["status"],
            "scanner_sensitivity_passed": False,
            "science_gate_status": "BLOCKED",
            "claim_status": "EXPLORATORY",
            "feasibility": feasibility,
            "unknown_imputed": False,
        }

    linkage_frame = pd.DataFrame(linkage)[
        ["patient_hash", "manufacturer", "scanner_sensitivity_eligible"]
    ]
    joined = frame.merge(linkage_frame, on="patient_hash", how="inner", validate="one_to_one")
    if branch == "amyloid":
        joined = joined.loc[joined["cohort"].astype(str) == "A4"].copy()
    manufacturers = list(feasibility["eligible_manufacturers"])
    analysis = joined.loc[
        joined["scanner_sensitivity_eligible"].astype(bool)
        & joined["manufacturer"].astype(str).isin(manufacturers)
    ].copy()
    root = Path(output_root)
    membership = {
        manufacturer: sorted(
            analysis.loc[analysis["manufacturer"] == manufacturer, "patient_hash"].astype(str)
        )
        for manufacturer in manufacturers
    }
    contract = RunContract.from_mapping(
        {
            "branch": branch,
            "analysis": "leave_manufacturer_out",
            "manufacturer_membership": membership,
            "feature_sets": feature_sets,
            "inner_folds": int(inner_folds),
            "seed": int(seed),
            "alpha_grid": [float(value) for value in alpha_grid],
        }
    )
    variant_reports: dict[str, Any] = {}
    for variant, features in feature_sets.items():
        fold_reports: list[dict[str, Any]] = []
        for fold_index, manufacturer in enumerate(manufacturers):
            train = analysis.loc[analysis["manufacturer"] != manufacturer].copy()
            test = analysis.loc[analysis["manufacturer"] == manufacturer].copy()
            inner = _inner_memberships(
                train,
                folds=int(inner_folds),
                seed=int(seed) + fold_index,
            )
            directory = (
                root
                / "scanner_sensitivity"
                / branch
                / variant
                / f"holdout_{manufacturer.casefold()}"
            )
            checkpoint = directory / "checkpoint.joblib"
            selection_path = directory / "inner_selection.json"
            if checkpoint.exists() and selection_path.exists():
                selection = read_json(selection_path)
                if selection.get("run_contract_sha256") != contract.digest:
                    raise ScannerSensitivityError("SCANNER_CHECKPOINT_RUN_CONTRACT_MISMATCH")
                checkpoint_digest = _sha256_file(checkpoint)
                if selection.get("checkpoint_sha256") != checkpoint_digest:
                    raise ScannerSensitivityError("SCANNER_CHECKPOINT_HASH_MISMATCH")
                model = joblib.load(checkpoint)
                winner = selection["winner"]
            elif checkpoint.exists() or selection_path.exists():
                raise ScannerSensitivityError("INCOMPLETE_SCANNER_SENSITIVITY_CHECKPOINT")
            else:
                winner, candidates = _select_alpha(
                    train,
                    features=features,
                    inner_folds=inner,
                    alpha_grid=alpha_grid,
                )
                model = _pipeline(float(winner["alpha"]))
                model.fit(train[features], _make_y(train))
                checkpoint_digest = _atomic_joblib(checkpoint, model)
                atomic_write_json(
                    selection_path,
                    {
                        "winner": winner,
                        "candidates": candidates,
                        "selection_source": "inner_validation_ibs_only",
                        "heldout_manufacturer": manufacturer,
                        "run_contract_sha256": contract.digest,
                        "checkpoint_sha256": checkpoint_digest,
                    },
                )
            store = OuterEvaluationStore(
                root / "scanner_sensitivity" / branch / variant / "outer_evaluation",
                fold=fold_index,
                run_contract_digest=contract.digest,
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
            fold_reports.append(
                {
                    "fold": fold_index,
                    "heldout_manufacturer": manufacturer,
                    "status": "COMPLETED",
                    "selected_alpha": float(winner["alpha"]),
                    **metrics,
                }
            )
        variant_reports[variant] = {
            "outer_folds_completed": len(fold_reports),
            "mean_uno_c_index": float(np.mean([fold["uno_c_index"] for fold in fold_reports])),
            "mean_integrated_brier_score": float(
                np.mean([fold["integrated_brier_score"] for fold in fold_reports])
            ),
            "folds": fold_reports,
        }
    bootstrap = cached_paired_oof_bootstrap(
        variant_reports[base_variant],
        variant_reports[augmented_variant],
        output_path=root / "reports" / f"{branch}_scanner_paired_bootstrap.json",
        iterations=int(bootstrap_iterations),
        seed=int(seed) + 3000,
    )
    scanner_passed = (
        float(bootstrap["delta_ibs_95ci"][1]) < 0.0
        and float(bootstrap["delta_cindex_95ci"][0]) >= 0.0
        and all(
            report["outer_folds_completed"] == len(manufacturers)
            for report in variant_reports.values()
        )
    )
    report = {
        "status": "COMPLETED",
        "analysis": "leave_manufacturer_out",
        "branch": branch,
        "heldout_manufacturers": manufacturers,
        "feature_sets": feature_sets,
        "variants": variant_reports,
        "paired_bootstrap": bootstrap,
        "scanner_sensitivity_passed": scanner_passed,
        "science_gate_status": "PASS" if scanner_passed else "FAIL",
        "claim_status": "NO_CLAIM",
        "unknown_imputed": False,
        "excluded_manufacturers": sorted(
            set(feasibility["manufacturer_support"]) - set(manufacturers)
        ),
        "feasibility": feasibility,
        "run_contract_sha256": contract.digest,
    }
    atomic_write_json(root / "reports" / f"{branch}_scanner_sensitivity.json", report)
    return report
