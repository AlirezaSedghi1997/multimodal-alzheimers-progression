"""Outcome-blind split construction for a post-hoc, internal scanner-transport audit.

The eligibility thresholds are fixed in the caller's versioned policy.  No
outer-test performance is used to choose manufacturers, hyperparameters, or
models.  Patient identifiers in the returned plan are private artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .contracts import sha256_mapping


class ImageLMOError(RuntimeError):
    pass


def build_lmo_plan(
    development: pd.DataFrame,
    *,
    manufacturer_order: list[str],
    minimum_subjects: int,
    minimum_events: int,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    required = {"patient_hash", "manufacturer", "event", "time_days"}
    if required - set(development.columns):
        raise ImageLMOError("DEVELOPMENT_COLUMNS_MISSING")
    frame = development.copy()
    if frame[["patient_hash", "event", "time_days"]].isna().any().any():
        raise ImageLMOError("MISSING_ENDPOINT_OR_SUBJECT")
    frame["patient_hash"] = frame["patient_hash"].astype(str)
    frame["manufacturer"] = frame["manufacturer"].fillna("UNKNOWN").astype(str).str.upper()
    if (frame["patient_hash"].str.strip() == "").any():
        raise ImageLMOError("EMPTY_SUBJECT_KEY")
    if frame["patient_hash"].duplicated().any():
        raise ImageLMOError("DUPLICATE_DEVELOPMENT_SUBJECT")
    if not frame["event"].isin([0, 1]).all():
        raise ImageLMOError("NONBINARY_EVENT")
    if not np.isfinite(frame["time_days"].astype(float).to_numpy()).all() or not (frame["time_days"].astype(float) > 0).all():
        raise ImageLMOError("FOLLOWUP_TIME_INVALID")
    if len(manufacturer_order) < 2 or len(set(manufacturer_order)) != len(manufacturer_order):
        raise ImageLMOError("MANUFACTURER_ORDER_INVALID")
    if inner_folds < 2 or minimum_subjects < 1 or minimum_events < 1:
        raise ImageLMOError("ELIGIBILITY_POLICY_INVALID")

    support = {
        str(name): {
            "subjects": int(len(group)),
            "events": int(group["event"].astype(int).sum()),
        }
        for name, group in frame.groupby("manufacturer", sort=True)
    }
    eligible = [
        name
        for name in manufacturer_order
        if support.get(name, {}).get("subjects", 0) >= minimum_subjects
        and support.get(name, {}).get("events", 0) >= minimum_events
    ]
    if eligible != manufacturer_order:
        raise ImageLMOError("PRESPECIFIED_MANUFACTURER_SUPPORT_FAILED")
    analysis = frame[frame["manufacturer"].isin(eligible)].copy()
    all_keys = set(analysis["patient_hash"])
    excluded = {
        name: counts for name, counts in support.items() if name not in eligible
    }
    outer: list[dict[str, Any]] = []
    test_count = {key: 0 for key in all_keys}
    for outer_index, heldout in enumerate(eligible):
        test = analysis[analysis["manufacturer"] == heldout].copy()
        train = analysis[analysis["manufacturer"] != heldout].copy()
        train_keys = set(train["patient_hash"])
        test_keys = set(test["patient_hash"])
        if train_keys & test_keys or train_keys | test_keys != all_keys:
            raise ImageLMOError("OUTER_PARTITION_INVALID")
        if set(train["event"].astype(int)) != {0, 1} or set(test["event"].astype(int)) != {0, 1}:
            raise ImageLMOError("OUTER_EVENT_SUPPORT_FAILED")
        for key in test_keys:
            test_count[key] += 1
        train = train.sort_values("patient_hash").reset_index(drop=True)
        strata = train["manufacturer"].astype(str) + "|" + train["event"].astype(int).astype(str)
        if int(strata.value_counts().min()) < inner_folds:
            raise ImageLMOError("INNER_STRATUM_TOO_SMALL")
        splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + outer_index)
        inner: list[dict[str, Any]] = []
        validation_count = {key: 0 for key in train_keys}
        for inner_index, (fit_idx, val_idx) in enumerate(splitter.split(train, strata)):
            fit = train.iloc[fit_idx]
            val = train.iloc[val_idx]
            fit_keys = set(fit["patient_hash"])
            val_keys = set(val["patient_hash"])
            if fit_keys & val_keys or fit_keys | val_keys != train_keys:
                raise ImageLMOError("INNER_PARTITION_INVALID")
            if set(fit["event"].astype(int)) != {0, 1} or set(val["event"].astype(int)) != {0, 1}:
                raise ImageLMOError("INNER_EVENT_SUPPORT_FAILED")
            for key in val_keys:
                validation_count[key] += 1
            inner.append({
                "fold": inner_index,
                "train_subjects": sorted(fit_keys),
                "validation_subjects": sorted(val_keys),
            })
        if any(value != 1 for value in validation_count.values()):
            raise ImageLMOError("INNER_VALIDATION_COVERAGE_INVALID")
        outer.append({
            "fold": outer_index,
            "heldout_manufacturer": heldout,
            "train_subjects": sorted(train_keys),
            "test_subjects": sorted(test_keys),
            "train_count": len(train_keys),
            "train_events": int(train["event"].sum()),
            "test_count": len(test_keys),
            "test_events": int(test["event"].sum()),
            "inner_folds": inner,
        })
    if any(value != 1 for value in test_count.values()):
        raise ImageLMOError("OUTER_TEST_COVERAGE_INVALID")
    body = {
        "schema_version": "1.0",
        "analysis_role": "POST_HOC_INTERNAL_SCANNER_TRANSPORT_EXPLORATORY",
        "manufacturer_order": eligible,
        "minimum_subjects": minimum_subjects,
        "minimum_events": minimum_events,
        "inner_folds": inner_folds,
        "seed": seed,
        "included_subjects": len(all_keys),
        "excluded_support": excluded,
        "outer_folds": outer,
    }
    return {**body, "plan_sha256": sha256_mapping(body)}


def public_plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    """Only aggregate feasibility facts; never serialize private memberships."""
    return {
        "analysis_role": plan["analysis_role"],
        "manufacturer_order": list(plan["manufacturer_order"]),
        "included_subjects": int(plan["included_subjects"]),
        "excluded_support": dict(plan["excluded_support"]),
        "outer_support": [
            {
                "manufacturer": row["heldout_manufacturer"],
                "train_count": row["train_count"],
                "train_events": row["train_events"],
                "test_count": row["test_count"],
                "test_events": row["test_events"],
            }
            for row in plan["outer_folds"]
        ],
        "plan_sha256": plan["plan_sha256"],
    }
