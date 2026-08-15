"""Fail-closed pretraining contracts for Method Y v2.

The public records produced by this module contain counts and cryptographic
digests only.  Subject memberships stay in a versioned private directory and
are never suitable for Git publication.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping
from .endpoint_builder import sha256_file
from .splits import build_nested_subject_splits


class PretrainingContractError(ValueError):
    """Raised when a pretraining evidence chain cannot be locked safely."""


BRANCH_TO_ENDPOINT = {
    "amyloid_source_regional": "amyloid",
    "tau_source_regional": "tau",
    "paired_source_regional": "paired_at",
}
FORBIDDEN_PREDICTOR_EXACT = frozenset(
    {
        "baseline_pacc_change",
        "baseline_cdglobal",
        "baseline_cdrsb",
        "amyloid_pos",
        "baseline_amyloid_status_cl24",
        "event",
        "time_days",
        "censoring_rule",
        "followup_cdr_visits",
        "single_visit_cdglobal_ge_0_5",
    }
)
FORBIDDEN_PREDICTOR_TOKENS = (
    "cdr",
    "faq",
    "diagnos",
    "outcome",
    "followup",
    "future",
    "progress",
    "event",
    "conversion",
)
PRIVATE_TOKENS = (
    "patient_hash",
    "subject_key",
    "source_subject_id",
    "raw_path",
    "sidecar_path",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER_IDENTIFIERS = frozenset(
    {
        "nan",
        "none",
        "null",
        "unknown",
        "missing",
        "unavailable",
        "not available",
        "not_available",
        "n/a",
        "na",
        "tbd",
        "placeholder",
    }
)
SCANNER_MINIMUM_SUBJECTS = 30
SCANNER_MINIMUM_EVENTS = 10
SCANNER_MINIMUM_ELIGIBLE_MANUFACTURERS = 2


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.lower()) is not None


def _json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PretrainingContractError(f"INVALID_JSON_SNAPSHOT:{path.name}") from exc
    if not isinstance(payload, dict):
        raise PretrainingContractError(f"JSON_OBJECT_REQUIRED:{path.name}")
    return payload, hashlib.sha256(raw).hexdigest()


def _canonical_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            raise PretrainingContractError("NONFINITE_ENDPOINT_VALUE")
        return format(number, ".17g")
    if pd.isna(value):
        return None
    text = str(value)
    return None if text == "" else text


def canonical_frame_digest(frame: pd.DataFrame, *, key: str = "patient_hash") -> str:
    """Return a deterministic digest of one exact logical table snapshot."""

    if key not in frame.columns:
        raise PretrainingContractError(f"CANONICAL_KEY_MISSING:{key}")
    if frame[key].astype(str).duplicated().any():
        raise PretrainingContractError(f"CANONICAL_KEY_DUPLICATED:{key}")
    columns = sorted(str(column) for column in frame.columns)
    ordered = frame.sort_values(key, kind="mergesort")[columns]
    rows = [
        {column: _canonical_scalar(value) for column, value in zip(columns, row, strict=True)}
        for row in ordered.itertuples(index=False, name=None)
    ]
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def canonical_longitudinal_frame_digest(frame: pd.DataFrame) -> str:
    """Hash a repeated-measures table after stable numeric serialization.

    CSV parsing can move a binary float by one ULP (for example when a derived
    ``time_years`` value is written and read again).  Fourteen significant
    decimal digits preserve the scientific value while making the digest
    stable across that lossless-on-disk round trip.  The verifier separately
    enforces tight numeric equality between the locked and rebuilt tables.
    """

    required = {"patient_hash", "time_days_from_pet"}
    if not required.issubset(frame.columns):
        raise PretrainingContractError("PACC_CANONICAL_KEYS_MISSING")
    working = frame.copy()
    subjects = working["patient_hash"].astype("string").str.strip()
    times = pd.to_numeric(working["time_days_from_pet"], errors="coerce")
    if (
        subjects.isna().any()
        or subjects.eq("").any()
        or subjects.str.casefold().isin(PLACEHOLDER_IDENTIFIERS).any()
        or times.isna().any()
        or not np.isfinite(times.to_numpy(dtype=float)).all()
    ):
        raise PretrainingContractError("PACC_CANONICAL_KEYS_INVALID")
    working["patient_hash"] = subjects
    working["time_days_from_pet"] = times
    # ``time_years`` is a deterministic display derivative, not an
    # independent measurement.  Recompute it from the locked day value so a
    # one-ULP CSV/parquet round trip cannot create a false row-digest failure.
    # ``verify_pacc_secondary_endpoint`` has already required the supplied
    # value to agree with this derivation at rtol/atol 1e-12.
    if "time_years" in working.columns:
        working["time_years"] = times / 365.25
    if working.duplicated(["patient_hash", "time_days_from_pet"]).any():
        raise PretrainingContractError("PACC_DUPLICATE_SUBJECT_TIME")
    columns = sorted(str(column) for column in working.columns)
    ordered = working.sort_values(
        ["patient_hash", "time_days_from_pet"], kind="mergesort"
    )[columns]
    def longitudinal_scalar(value: Any) -> Any:
        if isinstance(value, (np.floating, float)):
            number = float(value)
            if math.isnan(number):
                return None
            if math.isinf(number):
                raise PretrainingContractError("NONFINITE_ENDPOINT_VALUE")
            return format(number, ".14g")
        return _canonical_scalar(value)

    rows = [
        {
            column: longitudinal_scalar(value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in ordered.itertuples(index=False, name=None)
    ]
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def membership_digest(values: Iterable[str]) -> str:
    members = sorted(str(value) for value in values)
    if not members or len(members) != len(set(members)):
        raise PretrainingContractError("EMPTY_OR_DUPLICATE_MEMBERSHIP")
    return hashlib.sha256(
        json.dumps(members, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_endpoint_artifact(raw: bytes, storage_format: str) -> pd.DataFrame:
    if storage_format == "parquet":
        try:
            return pd.read_parquet(io.BytesIO(raw))
        except Exception as exc:  # pragma: no cover - exact backend varies
            raise PretrainingContractError("ENDPOINT_PARQUET_UNREADABLE") from exc
    if storage_format in {"csv", "csv_fallback"}:
        try:
            return pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            raise PretrainingContractError("ENDPOINT_CSV_UNREADABLE") from exc
    raise PretrainingContractError(f"UNSUPPORTED_ENDPOINT_STORAGE:{storage_format}")


def load_locked_endpoint_snapshot(
    source_root: str | Path, branch: str
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    """Read contract and endpoint once each, binding parsing to hashed bytes."""

    root = Path(source_root)
    if branch not in {"amyloid", "tau", "paired_at"}:
        raise PretrainingContractError(f"UNKNOWN_ENDPOINT_BRANCH:{branch}")
    contract_path = root / "endpoints" / f"{branch}_endpoint_contract.json"
    contract, contract_file_sha = _json_snapshot(contract_path)
    declared_contract_sha = contract.get("contract_sha256")
    unsigned_contract = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if (
        not _is_sha256(declared_contract_sha)
        or declared_contract_sha != sha256_mapping(unsigned_contract)
    ):
        raise PretrainingContractError(
            f"ENDPOINT_CONTRACT_SELF_HASH_INVALID:{branch}"
        )
    artifact_name = contract.get("endpoint_path")
    if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
        raise PretrainingContractError(f"INVALID_ENDPOINT_BASENAME:{branch}")
    artifact_path = root / "endpoints" / artifact_name
    try:
        artifact_raw = artifact_path.read_bytes()
    except OSError as exc:
        raise PretrainingContractError(f"ENDPOINT_ARTIFACT_MISSING:{branch}") from exc
    artifact_sha = hashlib.sha256(artifact_raw).hexdigest()
    if artifact_sha != contract.get("endpoint_sha256"):
        raise PretrainingContractError(f"ENDPOINT_ARTIFACT_HASH_MISMATCH:{branch}")
    frame = _read_endpoint_artifact(artifact_raw, str(contract.get("storage_format", "")))
    return contract, frame, {
        "contract_file_sha256": contract_file_sha,
        "endpoint_artifact_sha256": artifact_sha,
    }


def _exact_binary_values(
    values: pd.Series, *, error: str, require_both: bool = False
) -> pd.Series:
    raw_values = values.tolist()
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) not in {0.0, 1.0}
        for value in raw_values
    ):
        raise PretrainingContractError(error)
    numeric = pd.Series(
        [int(value) for value in raw_values], index=values.index, dtype="int64"
    )
    if require_both and set(numeric.unique()) != {0, 1}:
        raise PretrainingContractError(error)
    return numeric


def _finite_numeric(
    values: pd.Series, *, error: str
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise PretrainingContractError(error)
    return numeric


def _validate_treatment(frame: pd.DataFrame, *, required: pd.Series) -> None:
    if not {"treatment_arm", "treatment_active"}.issubset(frame.columns):
        raise PretrainingContractError("TREATMENT_FIELDS_MISSING")
    arm = frame["treatment_arm"].fillna("").astype(str).str.strip().str.casefold()
    active_raw = frame["treatment_active"]
    active = pd.to_numeric(active_raw, errors="coerce")
    present = active.notna()
    if (
        (~np.isfinite(active[present].to_numpy(dtype=float))).any()
        or (~active[present].isin([0.0, 1.0])).any()
    ):
        raise PretrainingContractError("TREATMENT_BINARY_ENCODING_INVALID")
    if active_raw[present].map(
        lambda value: isinstance(value, bool) or not isinstance(value, Real)
    ).any():
        raise PretrainingContractError("TREATMENT_BINARY_ENCODING_INVALID")
    placeholders = {"", "na", "nan", "none", "null", "unknown"}
    if active[required].isna().any() or arm[required].isin(placeholders).any():
        raise PretrainingContractError("REQUIRED_TREATMENT_ENCODING_MISSING")
    placebo = arm.str.contains("placebo", regex=False)
    treated = arm.str.contains("solanezumab", regex=False) | arm.str.contains(
        "active", regex=False
    )
    if ((active[placebo & present]) != 0.0).any():
        raise PretrainingContractError("PLACEBO_TREATMENT_ENCODING_INVALID")
    if ((active[treated & present]) != 1.0).any():
        raise PretrainingContractError("ACTIVE_TREATMENT_ENCODING_INVALID")
    if (present & ~(placebo | treated)).any():
        raise PretrainingContractError("TREATMENT_ARM_UNRECOGNIZED")


def validate_locked_endpoint(
    source_root: str | Path,
    branch: str,
    *,
    rebuilt_source_root: str | Path | None = None,
    rebuilt_frame: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate a locked future-only CDR endpoint and optional independent rebuild."""

    contract, frame, hashes = load_locked_endpoint_snapshot(source_root, branch)
    required = {
        "patient_hash",
        "event",
        "time_days",
        "cohort",
        "treatment_arm",
        "treatment_active",
        "baseline_gap_days_abs",
        "index_source",
        "censoring_rule",
        "followup_cdr_visits",
        "baseline_cdglobal",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise PretrainingContractError(
            f"ENDPOINT_REQUIRED_COLUMNS_MISSING:{branch}:{','.join(sorted(missing))}"
        )
    patient_raw = frame["patient_hash"]
    patient = patient_raw.astype("string").str.strip()
    if (
        patient_raw.isna().any()
        or patient.isna().any()
        or patient.eq("").any()
        or patient.str.casefold().isin(PLACEHOLDER_IDENTIFIERS).any()
        or patient.duplicated().any()
    ):
        raise PretrainingContractError(f"ENDPOINT_MEMBERSHIP_INVALID:{branch}")
    event = _exact_binary_values(
        frame["event"],
        error=f"BINARY_EVENT_SUPPORT_INVALID:{branch}",
        require_both=True,
    )
    time_days = _finite_numeric(
        frame["time_days"], error=f"NONPOSITIVE_ENDPOINT_TIME:{branch}"
    )
    if (time_days <= 0).any():
        raise PretrainingContractError(f"NONPOSITIVE_ENDPOINT_TIME:{branch}")
    if (time_days > 1640.0 + 1e-9).any():
        raise PretrainingContractError(f"FOLLOWUP_WINDOW_EXCEEDED:{branch}")
    baseline_cdr = _finite_numeric(
        frame["baseline_cdglobal"], error=f"BASELINE_CDR_NOT_ZERO:{branch}"
    )
    if (baseline_cdr != 0.0).any():
        raise PretrainingContractError(f"BASELINE_CDR_NOT_ZERO:{branch}")
    baseline_gap = _finite_numeric(
        frame["baseline_gap_days_abs"],
        error=f"BASELINE_PET_CDR_WINDOW_INVALID:{branch}",
    )
    if (baseline_gap < 0).any() or (baseline_gap > 183).any():
        raise PretrainingContractError(f"BASELINE_PET_CDR_WINDOW_INVALID:{branch}")
    followups = _finite_numeric(
        frame["followup_cdr_visits"],
        error=f"POST_INDEX_FOLLOWUP_REQUIRED:{branch}",
    )
    if (followups < 1).any():
        raise PretrainingContractError(f"POST_INDEX_FOLLOWUP_REQUIRED:{branch}")
    if (followups[event == 1] < 2).any():
        raise PretrainingContractError(f"EVENT_WITHOUT_CONFIRMING_VISIT:{branch}")
    expected_censoring = np.where(
        event.to_numpy() == 1, "event_time", "right_censored"
    )
    if not np.array_equal(
        frame["censoring_rule"].astype(str).to_numpy(), expected_censoring
    ):
        raise PretrainingContractError(f"CENSORING_RULE_INCONSISTENT:{branch}")
    index_source = set(frame["index_source"].astype(str))
    expected_index = (
        {"max_fbp_scan_and_baseline_cdr_dates"}
        if branch == "amyloid"
        else {"max_ftp_scan_and_baseline_cdr_dates"}
        if branch == "tau"
        else {"max_fbp_ftp_and_baseline_cdr_dates"}
    )
    if index_source != expected_index:
        raise PretrainingContractError(f"INDEX_DATE_RULE_INVALID:{branch}")
    if branch == "paired_at":
        if "paired_fbp_ftp_gap_days" not in frame:
            raise PretrainingContractError("PAIRED_SCAN_GAP_MISSING")
        paired_gap = _finite_numeric(
            frame["paired_fbp_ftp_gap_days"], error="PAIRED_SCAN_WINDOW_INVALID"
        )
        if (paired_gap < 0).any() or (paired_gap > 183).any():
            raise PretrainingContractError("PAIRED_SCAN_WINDOW_INVALID")
    cohort_raw = frame["cohort"]
    cohort = cohort_raw.astype("string").str.strip()
    if cohort_raw.isna().any() or cohort.isna().any() or cohort.eq("").any():
        raise PretrainingContractError(f"ENDPOINT_COHORT_INVALID:{branch}")
    observed_cohorts = set(cohort)
    if branch == "amyloid":
        if observed_cohorts != {"A4", "LEARN"}:
            raise PretrainingContractError("AMYLOID_COHORT_SCOPE_INVALID")
        treatment_required = cohort.eq("A4")
    else:
        if observed_cohorts != {"A4_placebo", "A4_treatment"}:
            raise PretrainingContractError(f"A4_TREATMENT_COHORT_SCOPE_INVALID:{branch}")
        treatment_required = pd.Series(True, index=frame.index)
    _validate_treatment(frame, required=treatment_required)
    if bool(frame.get("uses_raw_identifier", pd.Series(False, index=frame.index)).astype(bool).any()):
        raise PretrainingContractError(f"RAW_IDENTIFIER_FLAG_TRUE:{branch}")
    if bool(frame.get("uses_raw_path", pd.Series(False, index=frame.index)).astype(bool).any()):
        raise PretrainingContractError(f"RAW_PATH_FLAG_TRUE:{branch}")

    definition = contract.get("endpoint_definition")
    if not isinstance(definition, Mapping):
        raise PretrainingContractError(f"ENDPOINT_DEFINITION_MISSING:{branch}")
    if definition.get("event_time_rule") != "first_increase_if_confirmed_by_next_visit":
        raise PretrainingContractError(f"SUSTAINED_EVENT_RULE_INVALID:{branch}")
    if float(definition.get("followup_window_days", -1)) != 1640.0:
        raise PretrainingContractError(f"FOLLOWUP_CONTRACT_INVALID:{branch}")
    if branch != "paired_at" and float(definition.get("baseline_cdr_required", -1)) != 0.0:
        raise PretrainingContractError(f"BASELINE_CDR_CONTRACT_INVALID:{branch}")
    if contract.get("locked") is not True:
        raise PretrainingContractError(f"ENDPOINT_NOT_LOCKED:{branch}")
    declared_rows = contract.get("row_count")
    if (
        isinstance(declared_rows, bool)
        or not isinstance(declared_rows, Integral)
        or int(declared_rows) != len(frame)
    ):
        raise PretrainingContractError(f"ENDPOINT_ROW_COUNT_MISMATCH:{branch}")
    declared_events = contract.get("event_count")
    if (
        isinstance(declared_events, bool)
        or not isinstance(declared_events, Integral)
        or int(declared_events) != int(event.sum())
    ):
        raise PretrainingContractError(f"ENDPOINT_EVENT_COUNT_MISMATCH:{branch}")
    source_hashes = contract.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise PretrainingContractError(f"ENDPOINT_SOURCE_HASHES_MISSING:{branch}")
    if any(not _is_sha256(value) for value in source_hashes.values()):
        raise PretrainingContractError(f"ENDPOINT_SOURCE_HASH_INVALID:{branch}")

    row_digest = canonical_frame_digest(frame)
    rebuild = {
        "performed": rebuilt_source_root is not None,
        "row_digest_match": None,
        "source_definition_binding_match": None,
    }
    if rebuilt_source_root is not None and rebuilt_frame is not None:
        raise PretrainingContractError("ONLY_ONE_REBUILD_SOURCE_ALLOWED")
    if rebuilt_source_root is not None:
        rebuilt_contract, rebuilt_frame, _ = load_locked_endpoint_snapshot(
            rebuilt_source_root, branch
        )
        rebuild = {
            "performed": True,
            "row_digest_match": canonical_frame_digest(rebuilt_frame) == row_digest,
            "source_definition_binding_match": (
                rebuilt_contract.get("source_hashes") == contract.get("source_hashes")
                and rebuilt_contract.get("endpoint_definition")
                == contract.get("endpoint_definition")
            ),
        }
        if not all(rebuild.values()):
            raise PretrainingContractError(f"INDEPENDENT_ENDPOINT_REBUILD_MISMATCH:{branch}")
    elif rebuilt_frame is not None:
        rebuild = {
            "performed": True,
            "row_digest_match": canonical_frame_digest(rebuilt_frame) == row_digest,
            # The locked source hashes/definition and exact artifact bytes were
            # already verified above.  The independent reconstruction is
            # intentionally in-memory so Parquet writer-version differences
            # cannot masquerade as a scientific endpoint mismatch.
            "source_definition_binding_match": True,
        }
        if rebuild["row_digest_match"] is not True:
            raise PretrainingContractError(f"INDEPENDENT_ENDPOINT_REBUILD_MISMATCH:{branch}")

    cohort_counts = {
        str(name): int(count)
        for name, count in cohort.value_counts(sort=False).sort_index().items()
    }
    event_counts = {
        str(name): int(event.loc[group.index].sum())
        for name, group in frame.groupby(cohort, sort=True)
    }
    report = {
        "status": "PASS",
        "branch": branch,
        "locked": True,
        "contract_sha256": contract.get("contract_sha256"),
        "endpoint_definition_sha256": sha256_mapping(dict(definition)),
        "source_snapshot_digest": sha256_mapping(dict(source_hashes)),
        **hashes,
        "row_digest": row_digest,
        "membership_digest": membership_digest(patient),
        "row_count": int(len(frame)),
        "event_count": int(event.sum()),
        "cohort_counts": cohort_counts,
        "event_counts_by_cohort": event_counts,
        "baseline_cdr_required": 0.0,
        "future_only": True,
        "pre_pet_outcome_used": False,
        "sustained_confirmation_required": True,
        "terminal_unconfirmed_increase_is_event": False,
        "positive_time_required": True,
        "followup_window_days": 1640.0,
        "treatment_integrity": "PASS",
        "censoring_integrity": "PASS",
        "source_hashes": dict(sorted(source_hashes.items())),
        "independent_rebuild": rebuild,
        "private_rows_published": False,
    }
    return report, frame


def predictor_temporality_contract(
    columns: Iterable[str],
    *,
    timing_evidence: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Classify endpoint columns for use as claim-bearing baseline predictors."""

    evidence = {str(key): str(value) for key, value in (timing_evidence or {}).items()}
    allowed: list[str] = []
    forbidden: dict[str, str] = {}
    timing_blocked: dict[str, str] = {}
    for raw_name in sorted({str(name) for name in columns}):
        name = raw_name.casefold()
        if raw_name in PRIVATE_TOKENS or name in PRIVATE_TOKENS:
            forbidden[raw_name] = "PRIVATE_OR_MEMBERSHIP_FIELD"
            continue
        if name in FORBIDDEN_PREDICTOR_EXACT or any(
            token in name for token in FORBIDDEN_PREDICTOR_TOKENS
        ):
            forbidden[raw_name] = "OUTCOME_DERIVED_OR_CIRCULAR"
            continue
        if name == "baseline_pacc_change":
            forbidden[raw_name] = "FUTURE_DERIVED_CHANGE_SCORE"
            continue
        if name == "baseline_pacc_raw" or any(
            token in name
            for token in ("pacc", "cognition", "mmscore", "fctotal", "ldeltotal", "digittotal")
        ):
            if evidence.get(raw_name) == "EXACT_NONFUTURE":
                allowed.append(raw_name)
            else:
                timing_blocked[raw_name] = "EXACT_NONFUTURE_TIMING_NOT_PROVEN"
            continue
        if evidence.get(raw_name) in {"TIME_INVARIANT", "EXACT_NONFUTURE"}:
            allowed.append(raw_name)
        elif name in {
            "age",
            "sex_binary",
            "education_years",
            "apoe4_carrier",
            "treatment_active",
        }:
            timing_blocked[raw_name] = "EXACT_NONFUTURE_TIMING_NOT_PROVEN"
        else:
            forbidden[raw_name] = "NOT_IN_PRESPECIFIED_BASELINE_ALLOWLIST"
    payload = {
        "schema_version": "1.0",
        "status": "PASS",
        "locked": True,
        "policy": "unknown_timing_fails_closed",
        "allowed_predictors": allowed,
        "forbidden_predictors": forbidden,
        "timing_blocked_predictors": timing_blocked,
        "baseline_pacc_change_allowed": False,
        "baseline_cdr_covariates_allowed": False,
        "outcome_or_followup_fields_allowed": False,
        "forbidden_name_patterns": list(FORBIDDEN_PREDICTOR_TOKENS),
        "fold_local_processing_required": True,
    }
    payload["allowed_predictor_digest"] = sha256_mapping(
        {"allowed_predictors": allowed}
    )
    payload["contract_sha256"] = sha256_mapping(payload)
    return payload


def _event_map(frame: pd.DataFrame) -> dict[str, int]:
    events = _exact_binary_values(
        frame["event"], error="BINARY_EVENT_SUPPORT_INVALID:SPLIT"
    )
    return dict(
        zip(frame["patient_hash"].astype(str), events.astype(int), strict=True)
    )


def _fold_event_support(subjects: Iterable[str], events: Mapping[str, int]) -> dict[str, int]:
    values = [events[str(subject)] for subject in subjects]
    return {
        "event_0": int(sum(value == 0 for value in values)),
        "event_1": int(sum(value == 1 for value in values)),
    }


def validate_nested_split(
    split: Mapping[str, Any], endpoint: pd.DataFrame
) -> dict[str, Any]:
    """Validate complete 5x3 membership and event support, not just strata."""

    if (
        split.get("group_unit") != "subject_key"
        or split.get("outer_fold_count") != 5
        or split.get("inner_fold_count") != 3
    ):
        raise PretrainingContractError("SPLIT_SHAPE_CONTRACT_INVALID")
    expected_digest = sha256_mapping(
        {key: value for key, value in split.items() if key != "split_digest"}
    )
    if split.get("split_digest") != expected_digest:
        raise PretrainingContractError("SPLIT_DIGEST_INVALID")
    members = set(endpoint["patient_hash"].astype(str))
    events = _event_map(endpoint)
    seen_test: list[str] = []
    public_folds: list[dict[str, Any]] = []
    outer = split.get("outer_folds")
    if not isinstance(outer, list) or len(outer) != 5:
        raise PretrainingContractError("OUTER_FOLD_COUNT_INVALID")
    for expected_fold, fold in enumerate(outer):
        if fold.get("fold") != expected_fold:
            raise PretrainingContractError("OUTER_FOLD_NUMBER_INVALID")
        train = set(map(str, fold.get("train_subjects", [])))
        test = set(map(str, fold.get("test_subjects", [])))
        if train & test or train | test != members:
            raise PretrainingContractError("OUTER_MEMBERSHIP_OR_LEAKAGE_INVALID")
        train_support = _fold_event_support(train, events)
        test_support = _fold_event_support(test, events)
        if min(train_support.values()) < 1 or min(test_support.values()) < 1:
            raise PretrainingContractError("OUTER_FOLD_EVENT_CLASS_MISSING")
        seen_test.extend(test)
        inner_public: list[dict[str, Any]] = []
        inner_folds = fold.get("inner_folds")
        if not isinstance(inner_folds, list) or len(inner_folds) != 3:
            raise PretrainingContractError("INNER_FOLD_COUNT_INVALID")
        seen_validation: list[str] = []
        for expected_inner, inner in enumerate(inner_folds):
            if inner.get("fold") != expected_inner:
                raise PretrainingContractError("INNER_FOLD_NUMBER_INVALID")
            inner_train = set(map(str, inner.get("train_subjects", [])))
            validation = set(map(str, inner.get("validation_subjects", [])))
            if inner_train & validation or inner_train | validation != train:
                raise PretrainingContractError("INNER_MEMBERSHIP_OR_LEAKAGE_INVALID")
            train_inner_support = _fold_event_support(inner_train, events)
            validation_support = _fold_event_support(validation, events)
            if min(train_inner_support.values()) < 1 or min(validation_support.values()) < 1:
                raise PretrainingContractError("INNER_FOLD_EVENT_CLASS_MISSING")
            seen_validation.extend(validation)
            inner_public.append(
                {
                    "fold": expected_inner,
                    "train_subject_count": len(inner_train),
                    "validation_subject_count": len(validation),
                    "train_event_support": train_inner_support,
                    "validation_event_support": validation_support,
                }
            )
        if len(seen_validation) != len(train) or set(seen_validation) != train:
            raise PretrainingContractError("INNER_VALIDATION_PARTITION_INCOMPLETE")
        public_folds.append(
            {
                "fold": expected_fold,
                "train_subject_count": len(train),
                "test_subject_count": len(test),
                "train_event_support": train_support,
                "test_event_support": test_support,
                "inner_folds": inner_public,
            }
        )
    if len(seen_test) != len(members) or set(seen_test) != members:
        raise PretrainingContractError("OUTER_TEST_PARTITION_INCOMPLETE")
    return {
        "outer_fold_count": 5,
        "inner_fold_count": 3,
        "grouping_unit": "stable_pseudonymous_person",
        "complete_outer_test_partition": True,
        "subject_overlap_detected": False,
        "all_inner_and_outer_partitions_have_both_event_classes": True,
        "folds": public_folds,
    }


def build_private_split_artifacts(
    endpoint_frames: Mapping[str, pd.DataFrame],
    endpoint_reports: Mapping[str, Mapping[str, Any]],
    *,
    private_output_root: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Create deterministic private split memberships and a public verification."""

    root = Path(private_output_root)
    root.mkdir(parents=True, exist_ok=True)
    branch_frames = {
        "amyloid_source_regional": endpoint_frames["amyloid"].loc[
            endpoint_frames["amyloid"]["cohort"].astype(str) == "A4"
        ].copy(),
        "tau_source_regional": endpoint_frames["tau"].copy(),
        "paired_source_regional": endpoint_frames["paired_at"].copy(),
    }
    reports: dict[str, Any] = {}
    for branch, frame in branch_frames.items():
        rows = [
            {
                "subject_key": str(row.patient_hash),
                "event": int(row.event),
                "treatment_arm": str(row.treatment_arm),
            }
            for row in frame[["patient_hash", "event", "treatment_arm"]].itertuples(
                index=False
            )
        ]
        split = build_nested_subject_splits(
            rows, outer_folds=5, inner_folds=3, seed=int(seed)
        )
        validation = validate_nested_split(split, frame)
        private_payload = {
            "schema_version": "2.0",
            "locked": True,
            "branch": branch,
            "analysis_role": (
                "PRIMARY_DEVELOPMENT_A4"
                if branch == "amyloid_source_regional"
                else "A4_EXPLORATORY_DEVELOPMENT"
            ),
            "endpoint_contract_sha256": endpoint_reports[
                BRANCH_TO_ENDPOINT[branch]
            ]["contract_sha256"],
            "endpoint_membership_digest": membership_digest(frame["patient_hash"]),
            "split": split,
        }
        name = f"{branch}_nested_splits_v2_private.json"
        private_path = root / name
        atomic_write_json(private_path, private_payload)
        reports[branch] = {
            "status": "PASS",
            "locked": True,
            "analysis_role": private_payload["analysis_role"],
            "development_cohort": "A4",
            "subject_count": int(len(frame)),
            "event_count": int(frame["event"].astype(int).sum()),
            "endpoint_contract_sha256": private_payload["endpoint_contract_sha256"],
            "membership_digest": private_payload["endpoint_membership_digest"],
            "split_digest": split["split_digest"],
            "private_artifact_basename": name,
            "private_artifact_sha256": sha256_file(private_path),
            **validation,
        }

    amyloid = endpoint_frames["amyloid"]
    learn = amyloid.loc[amyloid["cohort"].astype(str) == "LEARN"].copy()
    if learn.empty:
        raise PretrainingContractError("LEARN_TRANSPORT_MEMBERSHIP_EMPTY")
    transport_payload = {
        "schema_version": "2.0",
        "locked": True,
        "branch": "amyloid_learn_transport",
        "analysis_role": "LOCKED_INTERNAL_TRANSPORT_ONLY_NO_HPO",
        "endpoint_contract_sha256": endpoint_reports["amyloid"]["contract_sha256"],
        "membership_digest": membership_digest(learn["patient_hash"]),
        "members": sorted(learn["patient_hash"].astype(str)),
    }
    transport_name = "amyloid_learn_transport_membership_v2_private.json"
    transport_path = root / transport_name
    atomic_write_json(transport_path, transport_payload)
    reports["amyloid_learn_transport"] = {
        "status": "PASS",
        "locked": True,
        "analysis_role": "LOCKED_INTERNAL_TRANSPORT_ONLY_NO_HPO",
        "development_or_hpo_allowed": False,
        "evaluation_allowed_only_after_global_model_lock": True,
        "subject_count": int(len(learn)),
        "event_count": int(learn["event"].astype(int).sum()),
        "endpoint_contract_sha256": endpoint_reports["amyloid"]["contract_sha256"],
        "membership_digest": transport_payload["membership_digest"],
        "private_artifact_basename": transport_name,
        "private_artifact_sha256": sha256_file(transport_path),
    }
    return {
        "schema_version": "2.0",
        "status": "PASS",
        "seed": int(seed),
        "legacy_amyloid_all_cohort_split_status": "SUPERSEDED_FOR_PRIMARY_DEVELOPMENT_READ_ONLY",
        "branches": reports,
        "private_memberships_tracked": False,
    }


def verify_pacc_secondary_endpoint(
    source_root: str | Path,
    *,
    expected_source_hashes: Mapping[str, str],
    expected_input_contract_sha256: str,
    rebuilt_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    root = Path(source_root) / "endpoints"
    contract, contract_file_sha = _json_snapshot(
        root / "longitudinal_pacc_endpoint_contract.json"
    )
    if contract.get("locked") is not True:
        raise PretrainingContractError("PACC_CONTRACT_NOT_LOCKED")
    if contract.get("status") != "LONGITUDINAL_PACC_ENDPOINT_LOCKED":
        raise PretrainingContractError("PACC_CONTRACT_STATUS_INVALID")
    if contract.get("raw_identifiers_present") is not False:
        raise PretrainingContractError("PACC_PRIVATE_FLAG_INVALID")
    if (
        not _is_sha256(expected_input_contract_sha256)
        or contract.get("input_contract_sha256")
        != expected_input_contract_sha256
    ):
        raise PretrainingContractError("PACC_INPUT_CONTRACT_HASH_INVALID")
    artifact_name = contract.get("endpoint_path")
    if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
        raise PretrainingContractError("PACC_ARTIFACT_BASENAME_INVALID")
    raw = (root / artifact_name).read_bytes()
    artifact_sha = hashlib.sha256(raw).hexdigest()
    if artifact_sha != contract.get("endpoint_sha256"):
        raise PretrainingContractError("PACC_ARTIFACT_HASH_MISMATCH")
    frame = pd.read_csv(io.BytesIO(raw))
    required = {"patient_hash", "time_days_from_pet", "time_years", "pacc_raw"}
    if not required.issubset(frame.columns):
        raise PretrainingContractError("PACC_COLUMNS_MISSING")
    patient_raw = frame["patient_hash"]
    subjects = patient_raw.astype("string").str.strip()
    if (
        patient_raw.isna().any()
        or subjects.isna().any()
        or subjects.eq("").any()
        or subjects.str.casefold().isin(PLACEHOLDER_IDENTIFIERS).any()
    ):
        raise PretrainingContractError("PACC_MEMBERSHIP_INVALID")
    time_days = _finite_numeric(
        frame["time_days_from_pet"], error="PACC_FUTURE_WINDOW_INVALID"
    )
    time_years = _finite_numeric(
        frame["time_years"], error="PACC_TIME_YEARS_INVALID"
    )
    if (time_days <= 0).any() or (time_days > 1640.0).any():
        raise PretrainingContractError("PACC_FUTURE_WINDOW_INVALID")
    _finite_numeric(frame["pacc_raw"], error="PACC_MEASUREMENT_INVALID")
    if not np.allclose(
        time_years.to_numpy(dtype=float),
        time_days.to_numpy(dtype=float) / 365.25,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise PretrainingContractError("PACC_TIME_YEARS_INCONSISTENT")
    identity_time = pd.DataFrame(
        {"patient_hash": subjects, "time_days_from_pet": time_days}
    )
    if identity_time.duplicated(["patient_hash", "time_days_from_pet"]).any():
        raise PretrainingContractError("PACC_DUPLICATE_SUBJECT_TIME")
    if (subjects.value_counts() < 2).any():
        raise PretrainingContractError("PACC_MINIMUM_MEASUREMENTS_NOT_MET")
    if int(contract.get("row_count", -1)) != len(frame):
        raise PretrainingContractError("PACC_ROW_COUNT_MISMATCH")
    if int(contract.get("subject_count", -1)) != subjects.nunique():
        raise PretrainingContractError("PACC_SUBJECT_COUNT_MISMATCH")
    source_hashes = contract.get("source_sha256")
    if (
        not isinstance(source_hashes, Mapping)
        or not source_hashes
        or any(not _is_sha256(value) for value in source_hashes.values())
        or not isinstance(expected_source_hashes, Mapping)
        or not expected_source_hashes
        or any(not _is_sha256(value) for value in expected_source_hashes.values())
        or dict(source_hashes) != dict(expected_source_hashes)
    ):
        raise PretrainingContractError("PACC_SOURCE_HASHES_INVALID")
    amyloid_contract, amyloid_frame, _ = load_locked_endpoint_snapshot(
        Path(source_root), "amyloid"
    )
    if contract.get("parent_endpoint_sha256") != amyloid_contract.get("endpoint_sha256"):
        raise PretrainingContractError("PACC_PARENT_ENDPOINT_BINDING_INVALID")
    if not set(subjects).issubset(set(amyloid_frame["patient_hash"].astype(str))):
        raise PretrainingContractError("PACC_MEMBERSHIP_OUTSIDE_LOCKED_AMYLOID_ENDPOINT")
    audit, audit_file_sha = _json_snapshot(
        root / "longitudinal_pacc_endpoint_audit.json"
    )
    if audit.get("contract_sha256") != sha256_mapping(contract):
        raise PretrainingContractError("PACC_AUDIT_CONTRACT_HASH_INVALID")
    endpoint_audit = audit.get("endpoint_audit")
    exact_date_audit = audit.get("exact_date_audit")
    if not (
        audit.get("status") == "LONGITUDINAL_PACC_ENDPOINT_LOCKED"
        and isinstance(endpoint_audit, Mapping)
        and endpoint_audit.get("status") == "LONGITUDINAL_PACC_ENDPOINT_BUILT"
        and endpoint_audit.get("exact_measurement_dates_required") is True
        and int(endpoint_audit.get("eligible_measurement_count", -1)) == len(frame)
        and int(endpoint_audit.get("eligible_subject_count", -1)) == subjects.nunique()
        and endpoint_audit.get("raw_identifiers_in_output") is False
        and isinstance(exact_date_audit, Mapping)
        and exact_date_audit.get("date_rule")
        == "MEDIAN_EXACT_QSDTC_DAYS_T0_ACROSS_PACC_COMPONENTS"
        and sorted(str(value).upper() for value in exact_date_audit.get("component_codes", []))
        == ["DIGIT", "FCSRT96", "LMIIA", "MMSE"]
    ):
        raise PretrainingContractError("PACC_AUDIT_BINDING_INVALID")
    if rebuilt_frame is None:
        raise PretrainingContractError("PACC_INDEPENDENT_REBUILD_REQUIRED")
    if set(str(column) for column in rebuilt_frame.columns) != set(
        str(column) for column in frame.columns
    ):
        raise PretrainingContractError("PACC_REBUILD_MISMATCH:COLUMNS")
    rebuilt_subjects = rebuilt_frame["patient_hash"].astype("string").str.strip()
    rebuilt_days = _finite_numeric(
        rebuilt_frame["time_days_from_pet"], error="PACC_REBUILD_MISMATCH:TIME"
    )
    rebuilt_years = _finite_numeric(
        rebuilt_frame["time_years"], error="PACC_REBUILD_MISMATCH:YEARS"
    )
    _finite_numeric(
        rebuilt_frame["pacc_raw"], error="PACC_REBUILD_MISMATCH:MEASUREMENT"
    )
    if (
        rebuilt_subjects.isna().any()
        or rebuilt_subjects.eq("").any()
        or rebuilt_subjects.str.casefold().isin(PLACEHOLDER_IDENTIFIERS).any()
        or (rebuilt_days <= 0).any()
        or (rebuilt_days > 1640.0).any()
        or not np.allclose(
            rebuilt_years.to_numpy(dtype=float),
            rebuilt_days.to_numpy(dtype=float) / 365.25,
            rtol=1e-12,
            atol=1e-12,
        )
        or (rebuilt_subjects.value_counts() < 2).any()
    ):
        raise PretrainingContractError("PACC_REBUILD_MISMATCH:ROWS")
    row_digest = canonical_longitudinal_frame_digest(frame)
    rebuilt_row_digest = canonical_longitudinal_frame_digest(rebuilt_frame)
    if rebuilt_row_digest != row_digest:
        raise PretrainingContractError("PACC_REBUILD_MISMATCH:ROW_DIGEST")
    verified_input_contract = {
        "schema_version": "2.0",
        "source_sha256": dict(sorted(source_hashes.items())),
        "parent_amyloid_contract_sha256": amyloid_contract["contract_sha256"],
        "parent_amyloid_endpoint_sha256": amyloid_contract["endpoint_sha256"],
        "parent_amyloid_membership_digest": membership_digest(
            amyloid_frame["patient_hash"].astype(str)
        ),
        "measurement_column": "PACC.raw",
        "exact_date_rule": exact_date_audit["date_rule"],
        "component_codes": ["DIGIT", "FCSRT96", "LMIIA", "MMSE"],
        "post_pet_baseline_window_days": 183.0,
        "followup_window_days": 1640.0,
        "minimum_longitudinal_measurements": 2,
    }
    verified_input_digest = sha256_mapping(verified_input_contract)
    return {
        "status": "PASS",
        "role": "SECONDARY_LONGITUDINAL_ENDPOINT",
        "locked": True,
        "contract_file_sha256": contract_file_sha,
        "legacy_input_contract_sha256": contract.get("input_contract_sha256"),
        "verified_input_contract_sha256": verified_input_digest,
        "verified_input_contract": verified_input_contract,
        "endpoint_artifact_sha256": artifact_sha,
        "audit_file_sha256": audit_file_sha,
        "source_snapshot_digest": sha256_mapping(dict(source_hashes)),
        "row_digest": row_digest,
        "independent_rebuild": {
            "performed": True,
            "row_digest_match": True,
            "rebuilt_row_digest": rebuilt_row_digest,
        },
        "row_count": int(len(frame)),
        "subject_count": int(subjects.nunique()),
        "measurement_time_rule": "0 < time_days_from_pet <= 1640",
        "all_measurements_post_pet": True,
        "outcome_driven_redesign_performed": False,
        "private_rows_published": False,
    }


_MANUFACTURER_ALIASES = {
    "GE": "GE",
    "GENERAL_ELECTRIC": "GE",
    "GENERAL_ELECTRIC_HEALTHCARE": "GE",
    "PHILIPS": "PHILIPS",
    "SIEMENS": "SIEMENS",
    "SIEMENS_HEALTHINEERS": "SIEMENS",
    "TOSHIBA": "TOSHIBA_CANON",
    "CANON": "TOSHIBA_CANON",
    "TOSHIBA_CANON": "TOSHIBA_CANON",
    "UNKNOWN": "UNKNOWN",
}


def _canonical_manufacturer_mapping(
    values: Mapping[str, Any], *, allow_unknown: bool
) -> dict[str, tuple[str, Any]]:
    canonical: dict[str, tuple[str, Any]] = {}
    for raw_name, value in values.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise PretrainingContractError("SCANNER_MANUFACTURER_LABEL_INVALID")
        normalized = re.sub(r"[^A-Z0-9]+", "_", raw_name.upper()).strip("_")
        name = _MANUFACTURER_ALIASES.get(normalized)
        if name is None or (name == "UNKNOWN" and not allow_unknown):
            raise PretrainingContractError(
                f"SCANNER_MANUFACTURER_LABEL_INVALID:{raw_name}"
            )
        if name in canonical:
            raise PretrainingContractError(
                f"SCANNER_MANUFACTURER_DUPLICATE_ALIAS:{name}"
            )
        canonical[name] = (raw_name, value)
    return canonical


def _exact_nonnegative_int(value: Any, *, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise PretrainingContractError(error)
    return int(value)


def scanner_contract_from_aggregate(
    scanner_audit_path: str | Path,
) -> dict[str, Any]:
    audit, audit_sha = _json_snapshot(Path(scanner_audit_path))
    if audit.get("status") != "COMPLETED":
        raise PretrainingContractError("SCANNER_AUDIT_NOT_COMPLETED")
    if audit.get("unknown_scanner_imputed") is not False:
        raise PretrainingContractError("SCANNER_UNKNOWN_WAS_IMPUTED")
    output: dict[str, Any] = {
        "schema_version": "2.0",
        "status": "PASS_WITH_BRANCH_LIMITATIONS",
        "source_aggregate_sha256": audit_sha,
        "unknown_policy": "KEEP_LITERAL_UNKNOWN_AND_EXCLUDE_FROM_SENSITIVITY",
        "unknown_imputed": False,
        "manufacturer_pooling": False,
        "ge_philips_pooled": False,
        "metadata_feasibility_is_not_post_model_sensitivity": True,
        "branches": {},
    }
    for branch in ("amyloid", "tau"):
        key = f"{branch}_sensitivity_feasibility"
        source = audit.get(key)
        if not isinstance(source, Mapping):
            raise PretrainingContractError(f"SCANNER_FEASIBILITY_MISSING:{branch}")
        support = source.get("manufacturer_support")
        if not isinstance(support, Mapping) or not support:
            raise PretrainingContractError(f"SCANNER_FEASIBILITY_MISSING:{branch}")
        support_by_name = _canonical_manufacturer_mapping(
            support, allow_unknown=False
        )
        if source.get("unknown_imputed") is not False:
            raise PretrainingContractError(f"SCANNER_UNKNOWN_IMPUTED:{branch}")
        metadata = audit.get(branch)
        if not isinstance(metadata, Mapping):
            raise PretrainingContractError(f"SCANNER_METADATA_SUMMARY_MISSING:{branch}")
        manufacturer_counts = metadata.get("manufacturer_counts")
        if not isinstance(manufacturer_counts, Mapping) or not manufacturer_counts:
            raise PretrainingContractError(f"SCANNER_METADATA_COUNTS_MISSING:{branch}")
        metadata_by_name = _canonical_manufacturer_mapping(
            manufacturer_counts, allow_unknown=True
        )
        for name, (_, value) in metadata_by_name.items():
            _exact_nonnegative_int(
                value, error=f"SCANNER_METADATA_COUNT_INVALID:{branch}:{name}"
            )
        unknown_count = (
            _exact_nonnegative_int(
                metadata_by_name["UNKNOWN"][1],
                error=f"SCANNER_METADATA_COUNT_INVALID:{branch}:UNKNOWN",
            )
            if "UNKNOWN" in metadata_by_name
            else 0
        )
        minimum_subjects = _exact_nonnegative_int(
            source.get("minimum_subjects_per_manufacturer"),
            error=f"SCANNER_THRESHOLD_INVALID:{branch}",
        )
        minimum_events = _exact_nonnegative_int(
            source.get("minimum_events_per_manufacturer"),
            error=f"SCANNER_THRESHOLD_INVALID:{branch}",
        )
        minimum_manufacturers = _exact_nonnegative_int(
            source.get("minimum_eligible_manufacturers"),
            error=f"SCANNER_THRESHOLD_INVALID:{branch}",
        )
        if (
            minimum_subjects != SCANNER_MINIMUM_SUBJECTS
            or minimum_events != SCANNER_MINIMUM_EVENTS
            or minimum_manufacturers
            != SCANNER_MINIMUM_ELIGIBLE_MANUFACTURERS
        ):
            raise PretrainingContractError(
                f"SCANNER_FEASIBILITY_INCONSISTENT:{branch}:THRESHOLDS"
            )
        public_support: dict[str, dict[str, Any]] = {}
        recomputed_eligible: list[str] = []
        for name, (_, values) in sorted(support_by_name.items()):
            if not isinstance(values, Mapping):
                raise PretrainingContractError(
                    f"SCANNER_SUPPORT_INVALID:{branch}:{name}"
                )
            subjects = _exact_nonnegative_int(
                values.get("subjects"),
                error=f"SCANNER_SUPPORT_INVALID:{branch}:{name}",
            )
            events = _exact_nonnegative_int(
                values.get("events"),
                error=f"SCANNER_SUPPORT_INVALID:{branch}:{name}",
            )
            metadata_entry = metadata_by_name.get(name)
            if metadata_entry is None:
                raise PretrainingContractError(
                    f"SCANNER_FEASIBILITY_INCONSISTENT:{branch}:{name}:MISSING_METADATA"
                )
            metadata_subjects = _exact_nonnegative_int(
                metadata_entry[1],
                error=f"SCANNER_METADATA_COUNT_INVALID:{branch}:{name}",
            )
            if events > subjects or subjects > metadata_subjects:
                raise PretrainingContractError(
                    f"SCANNER_FEASIBILITY_INCONSISTENT:{branch}:{name}:COUNT_ORDER"
                )
            expected_eligible = (
                subjects >= minimum_subjects and events >= minimum_events
            )
            if values.get("gate_eligible") is not expected_eligible:
                raise PretrainingContractError(
                    f"SCANNER_FEASIBILITY_INCONSISTENT:{branch}:{name}"
                )
            if expected_eligible:
                recomputed_eligible.append(name)
            public_support[name] = {
                "subjects": subjects,
                "events": events,
                "gate_eligible": expected_eligible,
            }
        declared_eligible_raw = source.get("eligible_manufacturers")
        if not isinstance(declared_eligible_raw, list):
            raise PretrainingContractError(
                f"SCANNER_FEASIBILITY_INCONSISTENT:{branch}:ELIGIBLE_LIST"
            )
        declared_eligible_map = _canonical_manufacturer_mapping(
            {str(name): None for name in declared_eligible_raw},
            allow_unknown=False,
        )
        declared_eligible = sorted(declared_eligible_map)
        if declared_eligible != sorted(recomputed_eligible):
            raise PretrainingContractError(
                f"SCANNER_FEASIBILITY_INCONSISTENT:{branch}:ELIGIBLE_LIST"
            )
        expected_status = (
            "READY_FOR_LEAVE_MANUFACTURER_OUT"
            if len(recomputed_eligible) >= minimum_manufacturers
            else "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT"
        )
        if source.get("status") != expected_status:
            raise PretrainingContractError(
                f"SCANNER_FEASIBILITY_INCONSISTENT:{branch}:STATUS"
            )
        output["branches"][branch] = {
            "scanner_feasibility_status": expected_status,
            "scanner_sensitivity_status": "NOT_RUN_REQUIRES_LOCKED_OOF_PREDICTIONS",
            "unknown_metadata_count": unknown_count,
            "unknown_metadata_retained_as_unknown": True,
            "site_imputed": False,
            "manufacturer_support": public_support,
            "eligible_manufacturers": sorted(recomputed_eligible),
            "minimum_subjects_per_manufacturer": minimum_subjects,
            "minimum_events_per_manufacturer": minimum_events,
            "minimum_eligible_manufacturers": minimum_manufacturers,
            "prespecified_post_training_audits": (
                ["leave_manufacturer_out", "leave_site_out"]
                if branch == "amyloid"
                else ["leave_manufacturer_out_if_feasible", "leave_site_out_if_feasible"]
            ),
            "claim_allowed": False,
            "claim_blocker": (
                "POST_MODEL_SCANNER_SENSITIVITY_NOT_RUN"
                if branch == "amyloid"
                else "TAU_SECOND_MANUFACTURER_EVENT_SUPPORT_INSUFFICIENT"
            ),
        }
    if output["branches"]["tau"]["scanner_feasibility_status"] != (
        "BLOCKED_INSUFFICIENT_MANUFACTURER_EVENT_SUPPORT"
    ):
        raise PretrainingContractError("TAU_SCANNER_LIMITATION_NOT_PRESERVED")
    output["branches"]["paired_at"] = {
        "scanner_feasibility_status": "BLOCKED_BY_TAU_MANUFACTURER_SUPPORT",
        "scanner_sensitivity_status": "NOT_RUN_REQUIRES_LOCKED_OOF_PREDICTIONS",
        "claim_allowed": False,
        "claim_blocker": "TAU_SECOND_MANUFACTURER_EVENT_SUPPORT_INSUFFICIENT",
        "prespecified_post_training_audits": [
            "leave_manufacturer_out_if_feasible",
            "leave_site_out_if_feasible",
        ],
    }
    return output


def external_intake_contracts() -> dict[str, Any]:
    cohorts: dict[str, Any] = {}
    for cohort in ("SCAN_NACC", "ADNI", "OASIS3"):
        cohorts[cohort] = {
            "root_locator": f"ENV:METHOD_Y_{cohort}_ROOT",
            "root_policy": "ABSOLUTE_H_DRIVE_READ_ONLY_REQUIRED_WHEN_AVAILABLE",
            "dua_status": "PENDING",
            "external_data_available": False,
            "data_dictionary_available": False,
            "mapping_locked": False,
            "outcome_access_allowed": False,
            "model_locked": False,
            "external_evaluation_allowed": False,
            "external_evaluation_committed": False,
            "required_modalities": [
                "FBP_OR_DOCUMENTED_AMYLOID_PET",
                "FTP_FOR_TAU_BRANCH",
                "T1_NEAR_PET_IF_IMAGE_ANALYSIS",
            ],
            "required_metadata": [
                "stable_pseudonymous_linkage_key",
                "pet_and_clinical_visit_dates",
                "tracer",
                "scanner_manufacturer_model_site",
                "baseline_and_followup_cdr_global",
                "last_contact_or_death_date",
                "documented_regional_suvr_or_centiloid",
            ],
        }
    payload = {
        "schema_version": "2.0",
        "status": "EXTERNAL_PENDING",
        "science_gate_status": "BLOCKED",
        "claim_allowed": False,
        "cohorts": cohorts,
        "outcome_embargo": {
            "enforced": True,
            "outcomes_may_be_opened_only_after": [
                "DUA_PASS",
                "DATA_DICTIONARY_HASH_LOCKED",
                "VARIABLE_MAPPING_LOCKED",
                "FEATURE_INTERSECTION_LOCKED",
                "INTERNAL_MODEL_LOCKED",
            ],
            "refit_allowed": False,
            "feature_reselection_allowed": False,
            "outcome_based_harmonization_allowed": False,
            "threshold_reselection_allowed": False,
            "initial_evaluation_exactly_once": True,
        },
        "canonical_endpoint": {
            "name": "future_sustained_confirmed_cdr_progression",
            "baseline_cdr_required": 0.0,
            "event_requires_later_confirmation": True,
            "terminal_unconfirmed_increase_is_event": False,
            "time_zero_after_pet": True,
        },
        "internal_development_blocked_by_external_access": False,
        "absolute_local_paths_published": False,
        "external_outcomes_accessed": False,
    }
    payload["contract_sha256"] = sha256_mapping(payload)
    return payload


def assert_public_contract_privacy(payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    forbidden = (
        "H:\\",
        "H:/",
        "patient_hash",
        "subject_key",
        "source_subject_id",
        "raw_path",
    )
    hits = [token for token in forbidden if token.casefold() in text.casefold()]
    if hits:
        raise PretrainingContractError(f"PUBLIC_CONTRACT_PRIVACY_VIOLATION:{hits}")


def commit_pretraining_state(
    output_root: str | Path,
    artifacts: Mapping[str, Path],
    *,
    implementation_commit: str,
    run_contract_context: Mapping[str, Any],
) -> dict[str, Any]:
    hashes = {name: sha256_file(path) for name, path in sorted(artifacts.items())}
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise PretrainingContractError("IMPLEMENTATION_COMMIT_INVALID")
    run_contract = {
        **dict(run_contract_context),
        "schema_version": "2.0",
        "implementation_commit": implementation_commit,
        "artifact_sha256": hashes,
        "reconstruction_mode": "LOCKED_RAW_SOURCE_INDEPENDENT_REBUILD",
        "release_eligible": True,
    }
    run_contract_sha256 = sha256_mapping(run_contract)
    state = {
        "schema_version": "2.0",
        "status": "AWAITING_AGENT_MULTI_PASS_QC",
        "stage": "ENDPOINT_SPLIT_SCANNER_EXTERNAL_VERIFIED",
        "artifact_sha256": hashes,
        "implementation_commit": implementation_commit,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "release_eligible": True,
        "training_started": False,
        "outer_evaluation_started": False,
        "external_outcomes_accessed": False,
        "agent_multi_pass_qc_status": "AGENT_MULTI_PASS_QC_PENDING",
    }
    atomic_write_json(Path(output_root) / "pretraining_validation_state.json", state)
    return state
