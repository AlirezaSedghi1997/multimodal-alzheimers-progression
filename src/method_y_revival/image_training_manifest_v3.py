from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import Any

import pandas as pd


class ImageMembershipError(ValueError):
    """Raised when image membership violates an immutable data-integrity rule."""


_REQUIRED_COLUMNS = {
    "patient_hash",
    "image_key",
    "tracer",
    "event",
    "time_days",
    "cohort",
    "manufacturer",
    "qc_status",
    "image_geometry_valid",
    "image_finite",
    "role",
}
_EXPECTED_TRACER = {"amyloid": "FBP", "tau": "FTP"}
_FORBIDDEN_MODEL_FEATURES = {
    "event",
    "time_days",
    "cdr_worsening_event",
    "pacc_decline_event",
    "baseline_pacc_change",
    "cdevent",
    "cdoleevent",
    "followup_cdr_visits",
    "single_visit_cdglobal_ge_0_5",
}


def validate_model_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(column).strip() for column in columns)
    if not normalized or any(not column for column in normalized):
        raise ImageMembershipError("MODEL_FEATURE_COLUMNS_REQUIRED")
    forbidden = sorted(
        column for column in normalized if column.lower() in _FORBIDDEN_MODEL_FEATURES
    )
    if forbidden:
        raise ImageMembershipError(
            "FORBIDDEN_FUTURE_DERIVED_FEATURE:" + ",".join(forbidden)
        )
    if len(set(normalized)) != len(normalized):
        raise ImageMembershipError("DUPLICATE_MODEL_FEATURE_COLUMN")
    return normalized


def _members(fold: Mapping[str, Any], *names: str) -> list[str]:
    for name in names:
        value = fold.get(name)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _canonical_membership_digest(frame: pd.DataFrame) -> str:
    fields = [
        "patient_hash",
        "image_key",
        "tracer",
        "event",
        "time_days",
        "cohort",
        "manufacturer",
        "role",
    ]
    rows = frame.loc[:, fields].sort_values("patient_hash", kind="stable")
    payload = rows.to_dict(orient="records")
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_membership(
    frame: pd.DataFrame,
    *,
    splits: Mapping[str, Any],
    branch: str,
    minimum_outer_events: int,
    minimum_inner_events: int,
) -> dict[str, Any]:
    """Audit a private subject-level image table without exporting its rows."""

    if not isinstance(frame, pd.DataFrame):
        raise ImageMembershipError("MEMBERSHIP_DATAFRAME_REQUIRED")
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ImageMembershipError(f"MEMBERSHIP_COLUMNS_MISSING:{missing}")
    branch_name = str(branch).strip().lower()
    if branch_name not in _EXPECTED_TRACER:
        raise ImageMembershipError(f"UNSUPPORTED_IMAGE_BRANCH:{branch_name}")
    if frame.empty:
        raise ImageMembershipError("EMPTY_IMAGE_MEMBERSHIP")
    if frame["patient_hash"].astype(str).duplicated().any():
        raise ImageMembershipError("DUPLICATE_PATIENT_HASH")
    if frame["image_key"].astype(str).duplicated().any():
        raise ImageMembershipError("DUPLICATE_IMAGE_KEY")

    expected = _EXPECTED_TRACER[branch_name]
    invalid_tracer = frame["tracer"].astype(str).str.upper() != expected
    invalid_qc = frame["qc_status"].astype(str) != "PASS"
    invalid_finite = ~frame["image_finite"].astype(bool)
    invalid_geometry = ~frame["image_geometry_valid"].astype(bool)
    invalid_rows = invalid_tracer | invalid_qc | invalid_finite | invalid_geometry

    patient_index = frame.set_index(frame["patient_hash"].astype(str), drop=False)
    event_by_subject = patient_index["event"].astype(int)
    if not event_by_subject.isin([0, 1]).all():
        raise ImageMembershipError("EVENT_MUST_BE_BINARY")
    times = pd.to_numeric(patient_index["time_days"], errors="coerce")
    if times.isna().any() or (times <= 0).any():
        raise ImageMembershipError("FOLLOWUP_TIME_MUST_BE_POSITIVE_FINITE")

    if splits.get("group_unit") != "subject_key":
        raise ImageMembershipError("SPLIT_GROUP_UNIT_MUST_BE_SUBJECT_KEY")
    outer_folds = splits.get("outer_folds")
    if not isinstance(outer_folds, list) or len(outer_folds) != 5:
        raise ImageMembershipError("FIVE_OUTER_FOLDS_REQUIRED")

    subject_set = set(event_by_subject.index)
    outer_reports: list[dict[str, Any]] = []
    overlap_count = 0
    outer_test_events: list[int] = []
    inner_validation_events: list[int] = []
    hard_blockers: list[str] = []
    outer_test_occurrences: Counter[str] = Counter()
    for expected_fold, outer in enumerate(outer_folds):
        if int(outer.get("fold", -1)) != expected_fold:
            raise ImageMembershipError("OUTER_FOLD_IDS_MUST_BE_CONTIGUOUS")
        train = set(_members(outer, "train_subjects", "train_subject_keys")) & subject_set
        test = set(_members(outer, "test_subjects", "test_subject_keys")) & subject_set
        if not train.isdisjoint(test):
            overlap_count += len(train & test)
            hard_blockers.append("SUBJECT_LEAKAGE_DETECTED")
        if train | test != subject_set:
            hard_blockers.append(f"OUTER_{expected_fold}_MEMBERSHIP_INCOMPLETE")
            raise ImageMembershipError(f"OUTER_{expected_fold}_MEMBERSHIP_INCOMPLETE")
        outer_test_occurrences.update(test)
        test_events = int(event_by_subject.loc[sorted(test)].sum()) if test else 0
        train_events = int(event_by_subject.loc[sorted(train)].sum()) if train else 0
        outer_test_events.append(test_events)
        inner_reports: list[dict[str, Any]] = []
        inner_folds = outer.get("inner_folds")
        if not isinstance(inner_folds, list) or len(inner_folds) != 3:
            raise ImageMembershipError("THREE_INNER_FOLDS_REQUIRED")
        for expected_inner, inner in enumerate(inner_folds):
            inner_train = set(
                _members(inner, "train_subjects", "train_subject_keys")
            ) & train
            validation = set(
                _members(
                    inner,
                    "validation_subjects",
                    "validation_subject_keys",
                    "val_subjects",
                )
            ) & train
            if not inner_train.isdisjoint(validation):
                overlap_count += len(inner_train & validation)
                hard_blockers.append("SUBJECT_LEAKAGE_DETECTED")
            if inner_train | validation != train:
                hard_blockers.append(
                    f"OUTER_{expected_fold}_INNER_{expected_inner}_PARTITION_INCOMPLETE"
                )
                raise ImageMembershipError(
                    f"INNER_PARTITION_INCOMPLETE:fold={expected_fold}:inner={expected_inner}"
                )
            validation_events = (
                int(event_by_subject.loc[sorted(validation)].sum()) if validation else 0
            )
            inner_validation_events.append(validation_events)
            inner_reports.append(
                {
                    "fold": expected_inner,
                    "train_subject_count": len(inner_train),
                    "train_event_count": (
                        int(event_by_subject.loc[sorted(inner_train)].sum())
                        if inner_train
                        else 0
                    ),
                    "validation_subject_count": len(validation),
                    "validation_event_count": validation_events,
                }
            )
        outer_reports.append(
            {
                "fold": expected_fold,
                "train_subject_count": len(train),
                "train_event_count": train_events,
                "test_subject_count": len(test),
                "test_event_count": test_events,
                "inner_folds": inner_reports,
            }
        )

    if set(outer_test_occurrences.keys()) != subject_set or any(
        c != 1 for c in outer_test_occurrences.values()
    ):
        hard_blockers.append("OUTER_TEST_EXACTLY_ONCE_VIOLATED")
        raise ImageMembershipError("OUTER_TEST_EXACTLY_ONCE_VIOLATED")

    minimum_outer_observed = min(outer_test_events, default=0)
    minimum_inner_observed = min(inner_validation_events, default=0)
    if overlap_count:
        hard_blockers.append("SUBJECT_LEAKAGE_DETECTED")
    if int(invalid_rows.sum()):
        hard_blockers.append("INVALID_IMAGE_ROWS_INCLUDED")
    if minimum_outer_observed < int(minimum_outer_events):
        hard_blockers.append(f"OUTER_EVENT_SUPPORT_BELOW_{int(minimum_outer_events)}")
    if minimum_inner_observed < int(minimum_inner_events):
        hard_blockers.append(f"INNER_EVENT_SUPPORT_BELOW_{int(minimum_inner_events)}")

    scanner_support = (
        frame.groupby(frame["manufacturer"].fillna("UNKNOWN").astype(str))["event"]
        .agg(["count", "sum"])
        .rename(columns={"count": "subjects", "sum": "events"})
        .sort_values("subjects", ascending=False)
    )
    return {
        "schema_version": "3.0",
        "status": "PASS" if not hard_blockers else "BLOCKED",
        "branch": branch_name,
        "tracer": expected,
        "eligible_subject_count": int(len(frame)),
        "eligible_event_count": int(event_by_subject.sum()),
        "invalid_images_included": int(invalid_rows.sum()),
        "all_images_finite": not bool(invalid_finite.any()),
        "all_images_geometry_valid": not bool(invalid_geometry.any()),
        "subject_overlap_count": int(overlap_count),
        "minimum_outer_test_events": int(minimum_outer_observed),
        "minimum_inner_validation_events": int(minimum_inner_observed),
        "minimum_required_outer_events": int(minimum_outer_events),
        "minimum_required_inner_events": int(minimum_inner_events),
        "outer_folds": outer_reports,
        "manufacturer_support": {
            str(index): {
                "subjects": int(row.subjects),
                "events": int(row.events),
            }
            for index, row in scanner_support.iterrows()
        },
        "membership_sha256": _canonical_membership_digest(frame),
        "hard_blockers": list(dict.fromkeys(hard_blockers)),
        "private_rows_exported": False,
    }

