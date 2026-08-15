from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .contracts import sha256_mapping


class SplitContractError(ValueError):
    """Raised when subject-level nested validation cannot be constructed safely."""


def _binary_event(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) not in {0.0, 1.0}
    ):
        raise SplitContractError("BINARY_EVENT_REQUIRED")
    return int(value)


def _subject_table(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"event": set(), "treatment_arm": set()}
    )
    for row in rows:
        subject = str(row["subject_key"])
        values[subject]["event"].add(str(_binary_event(row["event"])))
        values[subject]["treatment_arm"].add(str(row.get("treatment_arm") or "NONE"))
    table: list[dict[str, Any]] = []
    for subject in sorted(values):
        if len(values[subject]["event"]) != 1 or len(values[subject]["treatment_arm"]) != 1:
            raise SplitContractError(f"INCONSISTENT_SUBJECT_LABELS:{subject}")
        event = next(iter(values[subject]["event"]))
        treatment = next(iter(values[subject]["treatment_arm"]))
        table.append({"subject_key": subject, "stratum": f"{event}|{treatment}"})
    return table


def _fold_rows(
    table: list[dict[str, Any]], *, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(table) < n_splits:
        raise SplitContractError("INSUFFICIENT_SUBJECTS_FOR_SPLITS")
    labels = np.asarray([row["stratum"] for row in table])
    subjects = np.asarray([row["subject_key"] for row in table])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        folds = list(splitter.split(subjects, labels, groups=subjects))
    except ValueError as exc:
        raise SplitContractError(f"STRATIFIED_GROUP_SPLIT_FAILED:{exc}") from exc
    for train_index, test_index in folds:
        if set(subjects[train_index]).intersection(subjects[test_index]):
            raise SplitContractError("SUBJECT_LEAKAGE_ACROSS_FOLD")
        if len(set(labels[train_index])) < 2 or len(set(labels[test_index])) < 2:
            raise SplitContractError("FOLD_MISSING_REQUIRED_STRATA")
    return folds


def build_nested_subject_splits(
    rows: Iterable[dict[str, Any]], *, outer_folds: int, inner_folds: int, seed: int
) -> dict[str, Any]:
    table = _subject_table(rows)
    subjects = np.asarray([row["subject_key"] for row in table])
    outer_payload: list[dict[str, Any]] = []
    for outer_number, (train_index, test_index) in enumerate(
        _fold_rows(table, n_splits=outer_folds, seed=seed)
    ):
        outer_train_table = [table[int(index)] for index in train_index]
        train_subjects = sorted(subjects[train_index].tolist())
        test_subjects = sorted(subjects[test_index].tolist())
        inner_payload: list[dict[str, Any]] = []
        for inner_number, (inner_train, inner_validation) in enumerate(
            _fold_rows(
                outer_train_table,
                n_splits=inner_folds,
                seed=seed + outer_number + 1,
            )
        ):
            inner_subjects = np.asarray([row["subject_key"] for row in outer_train_table])
            inner_payload.append(
                {
                    "fold": inner_number,
                    "train_subjects": sorted(inner_subjects[inner_train].tolist()),
                    "validation_subjects": sorted(inner_subjects[inner_validation].tolist()),
                }
            )
        outer_payload.append(
            {
                "fold": outer_number,
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
                "inner_folds": inner_payload,
            }
        )
    payload: dict[str, Any] = {
        "seed": int(seed),
        "outer_fold_count": int(outer_folds),
        "inner_fold_count": int(inner_folds),
        "group_unit": "subject_key",
        "outer_folds": outer_payload,
    }
    payload["split_digest"] = sha256_mapping(payload)
    return payload
