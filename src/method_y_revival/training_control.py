from __future__ import annotations

import hashlib
import math
import platform
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

from .atomic_io import atomic_write_json, read_json
from .contracts import RunContract, RunContractMismatch, sha256_mapping
from .model_family_adapters import build_estimator_pipeline
from .survival_models import select_inner_winner


class TrainingControlError(ValueError):
    """Raised when inner model selection cannot satisfy its locked contract."""


class SyntheticTrainingInterruption(RuntimeError):
    """Test-only interruption boundary raised after a committed work unit."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FAMILIES = ("coxnet", "rsf", "gbsa")
_CANONICAL_BACKEND_ID = "canonical_sksurv_integrated_brier_score_v1"
_FOLD_LOCAL_POLICY = {
    "imputation": "median_fit_on_inner_train_only",
    "scaling": "standard_scaler_fit_on_inner_train_only",
    "feature_selection": "fit_on_inner_train_only_if_prespecified",
    "harmonization": "fit_on_inner_train_only_if_prespecified",
    "post_winner_refit": "fit_on_entire_outer_train_only",
}


def _require_sha256(value: Any, name: str) -> str:
    digest = str(value).lower()
    if not _SHA256.fullmatch(digest):
        raise TrainingControlError(f"INVALID_{name.upper()}_SHA256")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_file_digest(path: Path, label: str) -> str:
    if not path.is_file():
        raise TrainingControlError(f"{label.upper()}_SOURCE_FILE_REQUIRED")
    return _sha256_file(path)


def _dependency_digest() -> str:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "scikit-learn", "scikit-survival", "scipy"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise TrainingControlError(
                f"REQUIRED_DEPENDENCY_VERSION_MISSING:{distribution}"
            ) from exc
    return sha256_mapping(versions)


def _core_implementation_contract(
    execution_backend: Mapping[str, Any],
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    backend = _validated_backend_contract(execution_backend)
    metric_module = Path(sys.modules[integrated_brier_score.__module__].__file__).resolve()
    python_runtime = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag,
    }
    return {
        "code_sha256": _module_file_digest(Path(__file__).resolve(), "training_control"),
        "dependency_sha256": _dependency_digest(),
        "metric_sha256": _module_file_digest(metric_module, "integrated_brier_score"),
        "model_factory_sha256": _module_file_digest(
            module_root / "model_family_adapters.py", "model_factory"
        ),
        "preprocessing_sha256": sha256_mapping(_FOLD_LOCAL_POLICY),
        "python_runtime_sha256": sha256_mapping(python_runtime),
        "execution_backend": backend,
    }


def canonical_execution_backend_contract() -> dict[str, Any]:
    """Return the only production backend accepted by the inner controller."""

    module_root = Path(__file__).resolve().parent
    implementation = {
        "backend_id": _CANONICAL_BACKEND_ID,
        "training_control_sha256": _module_file_digest(
            Path(__file__).resolve(), "training_control"
        ),
        "model_factory_sha256": _module_file_digest(
            module_root / "model_family_adapters.py", "model_factory"
        ),
        "metric_module_sha256": _module_file_digest(
            Path(sys.modules[integrated_brier_score.__module__].__file__).resolve(),
            "integrated_brier_score",
        ),
    }
    return {
        "backend_id": _CANONICAL_BACKEND_ID,
        "implementation_sha256": sha256_mapping(implementation),
        "test_only": False,
    }


def _validated_backend_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TrainingControlError("EXECUTION_BACKEND_CONTRACT_REQUIRED")
    backend_id = payload.get("backend_id")
    if not isinstance(backend_id, str) or not backend_id.strip():
        raise TrainingControlError("EXECUTION_BACKEND_ID_REQUIRED")
    normalized = {
        "backend_id": backend_id,
        "implementation_sha256": _require_sha256(
            payload.get("implementation_sha256"), "execution_backend"
        ),
        "test_only": payload.get("test_only"),
    }
    if not isinstance(normalized["test_only"], bool):
        raise TrainingControlError("EXECUTION_BACKEND_TEST_ONLY_FLAG_REQUIRED")
    if backend_id == _CANONICAL_BACKEND_ID:
        if normalized != canonical_execution_backend_contract():
            raise TrainingControlError("CANONICAL_EXECUTION_BACKEND_DIGEST_MISMATCH")
    elif normalized["test_only"] is not True:
        raise TrainingControlError("NONCANONICAL_RUNTIME_BACKEND_FORBIDDEN")
    return normalized


def _validate_registry(registry: dict[str, Any]) -> list[dict[str, Any]]:
    families = registry.get("families")
    if not isinstance(families, dict) or set(families) != set(_FAMILIES):
        raise TrainingControlError("MATCHED_MODEL_FAMILIES_REQUIRED")
    rows_by_family = {family: families[family] for family in _FAMILIES}
    if any(not isinstance(rows, list) or not rows for rows in rows_by_family.values()):
        raise TrainingControlError("NONEMPTY_FAMILY_CANDIDATES_REQUIRED")
    budgets = {len(rows) for rows in rows_by_family.values()}
    if len(budgets) != 1 or budgets != {int(registry.get("budget_per_family", -1))}:
        raise TrainingControlError("UNEQUAL_HPO_BUDGET")
    for family, rows in rows_by_family.items():
        if any(candidate.get("family") != family for candidate in rows):
            raise TrainingControlError("CANDIDATE_FAMILY_MISMATCH")
    candidates = [candidate for family in _FAMILIES for candidate in rows_by_family[family]]
    identifiers = [str(candidate.get("candidate_id", "")) for candidate in candidates]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise TrainingControlError("UNIQUE_CANDIDATE_IDS_REQUIRED")
    if any(candidate.get("hpo_budget_units") != 1 for candidate in candidates):
        raise TrainingControlError("EQUAL_HPO_BUDGET_UNITS_REQUIRED")
    if registry.get("selection_metric") != "mean_inner_integrated_brier_score":
        raise TrainingControlError("INNER_IBS_SELECTION_REQUIRED")
    if registry.get("selection_rule") != "one_standard_error_simplest_candidate":
        raise TrainingControlError("ONE_STANDARD_ERROR_SELECTION_REQUIRED")
    if registry.get("preprocessing_policy") != _FOLD_LOCAL_POLICY:
        raise TrainingControlError("FOLD_LOCAL_PREPROCESSING_REQUIRED")
    return candidates


def _normalized_subjects(values: Iterable[str], label: str) -> list[str]:
    subjects = [str(value).strip() for value in values]
    if not subjects or any(not value for value in subjects):
        raise TrainingControlError(f"{label.upper()}_SUBJECT_KEYS_REQUIRED")
    if len(subjects) != len(set(subjects)):
        raise TrainingControlError(f"DUPLICATE_{label.upper()}_SUBJECT_KEY")
    return sorted(subjects)


def _locked_memberships(
    *,
    outer_train_subject_keys: Iterable[str],
    outer_test_subject_keys: Iterable[str],
    inner_fold_ids: list[int],
    inner_fold_memberships: Mapping[str | int, Mapping[str, Iterable[str]]],
) -> dict[str, Any]:
    outer_train = _normalized_subjects(outer_train_subject_keys, "outer_train")
    outer_test = _normalized_subjects(outer_test_subject_keys, "outer_test")
    if set(outer_train).intersection(outer_test):
        raise TrainingControlError("OUTER_TRAIN_TEST_SUBJECT_OVERLAP")
    normalized_folds: dict[str, dict[str, list[str]]] = {}
    validation_counts = {subject: 0 for subject in outer_train}
    for fold in inner_fold_ids:
        raw = inner_fold_memberships.get(str(fold), inner_fold_memberships.get(fold))
        if not isinstance(raw, Mapping):
            raise TrainingControlError("INNER_FOLD_MEMBERSHIP_REQUIRED")
        train = _normalized_subjects(raw.get("train_subject_keys", []), "inner_train")
        validation = _normalized_subjects(
            raw.get("validation_subject_keys", []), "inner_validation"
        )
        if set(train).intersection(validation):
            raise TrainingControlError("INNER_TRAIN_VALIDATION_SUBJECT_OVERLAP")
        if set(train).union(validation) != set(outer_train):
            raise TrainingControlError("INNER_FOLD_MUST_PARTITION_OUTER_TRAIN")
        if set(train).union(validation).intersection(outer_test):
            raise TrainingControlError("OUTER_TEST_SUBJECT_INNER_HPO_MEMBERSHIP")
        for subject in validation:
            validation_counts[subject] += 1
        normalized_folds[str(fold)] = {
            "train_subject_keys": train,
            "validation_subject_keys": validation,
        }
    if set(validation_counts.values()) != {1}:
        raise TrainingControlError("INNER_VALIDATION_MEMBERSHIP_NOT_EXACTLY_ONCE")
    return {
        "outer_train_subject_keys": outer_train,
        "outer_test_subject_keys": outer_test,
        "inner_folds": normalized_folds,
        "outer_train_membership_sha256": sha256_mapping(
            {"subject_keys": outer_train}
        ),
        "outer_test_membership_sha256": sha256_mapping({"subject_keys": outer_test}),
    }


def build_training_run_contract(
    *,
    branch: str,
    outer_fold: int,
    seed: int,
    endpoint_sha256: str,
    feature_sha256: str,
    split_digest: str,
    inner_fold_ids: Iterable[int],
    candidate_registry: dict[str, Any],
    outer_train_subject_keys: Iterable[str],
    outer_test_subject_keys: Iterable[str],
    inner_fold_memberships: Mapping[str | int, Mapping[str, Iterable[str]]],
    execution_backend_contract: Mapping[str, Any] | None = None,
) -> RunContract:
    """Build one immutable outer-train/inner-selection contract."""

    if branch not in {
        "amyloid_source_regional",
        "tau_source_regional",
        "paired_source_regional",
    }:
        raise TrainingControlError(f"UNSUPPORTED_SOURCE_REGIONAL_BRANCH:{branch}")
    fold = int(outer_fold)
    if fold not in range(5):
        raise TrainingControlError("EXACT_FIVE_OUTER_FOLDS_REQUIRED")
    fold_ids = [int(value) for value in inner_fold_ids]
    if sorted(fold_ids) != [0, 1, 2]:
        raise TrainingControlError("EXACT_THREE_INNER_FOLDS_REQUIRED")
    candidates = _validate_registry(candidate_registry)
    candidate_seeds = {int(candidate.get("seed")) for candidate in candidates}
    if candidate_seeds != {int(seed)}:
        raise TrainingControlError("CANDIDATE_SEED_MISMATCH")
    backend = _validated_backend_contract(
        execution_backend_contract or canonical_execution_backend_contract()
    )
    memberships = _locked_memberships(
        outer_train_subject_keys=outer_train_subject_keys,
        outer_test_subject_keys=outer_test_subject_keys,
        inner_fold_ids=sorted(fold_ids),
        inner_fold_memberships=inner_fold_memberships,
    )
    payload = {
        "schema_version": "2.1",
        "stage": "INNER_HPO_CONTROL",
        "branch": branch,
        "outer_fold": fold,
        "seed": int(seed),
        "endpoint_sha256": _require_sha256(endpoint_sha256, "endpoint"),
        "feature_sha256": _require_sha256(feature_sha256, "feature"),
        "split_digest": _require_sha256(split_digest, "split_digest"),
        "inner_fold_ids": sorted(fold_ids),
        "candidate_registry": candidate_registry,
        "subject_memberships": memberships,
        "implementation_contract": _core_implementation_contract(backend),
        "selection_data_scope": "locked_outer_train_inner_partitions_only",
        "outer_test_access": "FORBIDDEN_DURING_INNER_HPO",
        "checkpointing": "atomic_after_every_candidate_fold",
        "resume_policy": "exact_contract_and_implementation_only",
    }
    return RunContract.from_mapping(payload)


@dataclass(frozen=True)
class InnerSelectionData:
    subject_keys: tuple[str, ...]
    features: np.ndarray
    event: np.ndarray
    time_days: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        *,
        subject_keys: Iterable[str],
        features: Any,
        event: Any,
        time_days: Any,
    ) -> "InnerSelectionData":
        subjects = tuple(str(value).strip() for value in subject_keys)
        matrix = np.asarray(features, dtype=float)
        events = np.asarray(event)
        times = np.asarray(time_days, dtype=float)
        if not subjects or any(not value for value in subjects):
            raise TrainingControlError("INNER_DATA_SUBJECT_KEYS_REQUIRED")
        if len(subjects) != len(set(subjects)):
            raise TrainingControlError("DUPLICATE_INNER_DATA_SUBJECT_KEY")
        if matrix.ndim != 2 or matrix.shape[0] != len(subjects):
            raise TrainingControlError("INNER_DATA_FEATURE_SHAPE_MISMATCH")
        if events.ndim != 1 or times.ndim != 1 or len(events) != len(subjects) or len(
            times
        ) != len(subjects):
            raise TrainingControlError("INNER_DATA_OUTCOME_SHAPE_MISMATCH")
        if np.isinf(matrix).any():
            raise TrainingControlError("INFINITE_INNER_DATA_FEATURE")
        if not np.isfinite(times).all() or np.any(times <= 0):
            raise TrainingControlError("POSITIVE_FINITE_INNER_DATA_TIME_REQUIRED")
        if any(value not in (0, 1, False, True) for value in events.tolist()):
            raise TrainingControlError("BINARY_INNER_DATA_EVENT_REQUIRED")
        return cls(subjects, matrix, events.astype(bool), times)

    def subset(self, subject_keys: Iterable[str]) -> "InnerSelectionData":
        positions = {subject: index for index, subject in enumerate(self.subject_keys)}
        requested = tuple(str(subject) for subject in subject_keys)
        try:
            indices = np.asarray([positions[subject] for subject in requested], dtype=int)
        except KeyError as exc:
            raise TrainingControlError("LOCKED_SUBJECT_MISSING_FROM_INNER_DATA") from exc
        return InnerSelectionData(
            requested,
            self.features[indices],
            self.event[indices],
            self.time_days[indices],
        )


class _HashBoundSyntheticBackend(Protocol):
    backend_id: str
    implementation_sha256: str
    test_only: bool

    def score_unit(
        self,
        *,
        candidate: Mapping[str, Any],
        inner_fold: int,
        seed: int,
        train: InnerSelectionData,
        validation: InnerSelectionData,
    ) -> float: ...


def _backend_descriptor(backend: _HashBoundSyntheticBackend | None) -> dict[str, Any]:
    if backend is None:
        return canonical_execution_backend_contract()
    descriptor = {
        "backend_id": getattr(backend, "backend_id", None),
        "implementation_sha256": getattr(backend, "implementation_sha256", None),
        "test_only": getattr(backend, "test_only", None),
    }
    return _validated_backend_contract(descriptor)


def _administratively_censored_y(data: InnerSelectionData, cutoff: float):
    times = np.minimum(data.time_days, float(cutoff))
    events = data.event & (data.time_days < float(cutoff))
    return Surv.from_arrays(event=events, time=times)


def _canonical_inner_ibs(
    *,
    candidate: dict[str, Any],
    train: InnerSelectionData,
    validation: InnerSelectionData,
) -> float:
    """Fit the locked fold-local pipeline and compute canonical inner IBS."""

    raw_upper = min(float(train.time_days.max()), float(validation.time_days.max()))
    raw_lower = max(float(train.time_days.min()), float(validation.time_days.min()))
    if raw_lower >= raw_upper:
        raise TrainingControlError("NO_COMMON_INNER_SURVIVAL_SUPPORT")
    epsilon = max((raw_upper - raw_lower) * 1e-6, 1e-6)
    cutoff = raw_upper - epsilon
    grid = np.linspace(raw_lower + epsilon, cutoff - epsilon, num=32)
    if not np.all(np.diff(grid) > 0):
        raise TrainingControlError("INVALID_INNER_IBS_TIME_GRID")
    model = build_estimator_pipeline(candidate)
    y_fit = Surv.from_arrays(event=train.event, time=train.time_days)
    model.fit(train.features, y_fit)
    functions = model.predict_survival_function(validation.features)
    estimates = np.vstack([function(grid) for function in functions])
    score = integrated_brier_score(
        _administratively_censored_y(train, cutoff),
        _administratively_censored_y(validation, cutoff),
        estimates,
        grid,
    )
    if not math.isfinite(float(score)):
        raise TrainingControlError("FINITE_INNER_IBS_REQUIRED")
    return float(score)


def _verified_payload(path: Path, digest_field: str) -> dict[str, Any]:
    payload = read_json(path)
    claimed = payload.get(digest_field)
    body = {key: value for key, value in payload.items() if key != digest_field}
    if claimed != sha256_mapping(body):
        raise TrainingControlError(f"{digest_field.upper()}_MISMATCH")
    return payload


def _write_self_hashed(path: Path, payload: dict[str, Any], digest_field: str) -> None:
    body = dict(payload)
    body[digest_field] = sha256_mapping(payload)
    atomic_write_json(path, body)


class InnerHPOController:
    """Own locked memberships, fitting, scoring, resume and winner selection."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_contract: RunContract,
        candidate_registry: dict[str, Any],
        inner_fold_ids: Iterable[int],
        allow_test_backend: bool = False,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_contract = run_contract
        self.candidates = _validate_registry(candidate_registry)
        self.inner_fold_ids = sorted(int(value) for value in inner_fold_ids)
        if self.inner_fold_ids != run_contract.payload.get("inner_fold_ids"):
            raise TrainingControlError("INNER_FOLDS_DO_NOT_MATCH_RUN_CONTRACT")
        if candidate_registry != run_contract.payload.get("candidate_registry"):
            raise TrainingControlError("CANDIDATE_REGISTRY_DOES_NOT_MATCH_RUN_CONTRACT")
        implementation = run_contract.payload.get("implementation_contract")
        if not isinstance(implementation, Mapping):
            raise TrainingControlError("IMPLEMENTATION_CONTRACT_REQUIRED")
        expected_implementation = _core_implementation_contract(
            implementation.get("execution_backend", {})
        )
        if implementation != expected_implementation:
            raise RunContractMismatch("RUN_CONTRACT_MISMATCH:live_implementation")
        if (
            implementation["execution_backend"]["test_only"] is True
            and allow_test_backend is not True
        ):
            raise TrainingControlError("TEST_BACKEND_NOT_AUTHORIZED")
        memberships = run_contract.payload.get("subject_memberships")
        if not isinstance(memberships, Mapping):
            raise TrainingControlError("SUBJECT_MEMBERSHIP_CONTRACT_REQUIRED")
        self.memberships = memberships
        self.contract_path = self.root / "run_contract.json"
        self.state_path = self.root / "inner_hpo_state.json"
        self.winner_path = self.root / "locked_winner.json"
        self._lock_or_verify_contract()

    def _lock_or_verify_contract(self) -> None:
        if self.contract_path.exists():
            payload = read_json(self.contract_path)
            if not isinstance(payload.get("payload"), dict):
                raise TrainingControlError("MALFORMED_RUN_CONTRACT")
            existing = RunContract.from_mapping(payload["payload"])
            if payload.get("contract_sha256") != existing.digest:
                raise TrainingControlError("RUN_CONTRACT_SELF_HASH_MISMATCH")
            existing.require_match(self.run_contract)
            return
        atomic_write_json(
            self.contract_path,
            {
                "payload": self.run_contract.payload,
                "contract_sha256": self.run_contract.digest,
            },
        )

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "schema_version": "2.1",
                "status": "INNER_HPO_IN_PROGRESS",
                "run_contract_sha256": self.run_contract.digest,
                "completed_units": {},
            }
        state = _verified_payload(self.state_path, "state_sha256")
        if state.get("run_contract_sha256") != self.run_contract.digest:
            raise RunContractMismatch("RUN_CONTRACT_MISMATCH:inner_hpo_state")
        if not isinstance(state.get("completed_units"), dict):
            raise TrainingControlError("MALFORMED_INNER_HPO_STATE")
        body = {key: value for key, value in state.items() if key != "state_sha256"}
        self._validated_units(body["completed_units"], require_complete=False)
        return body

    def _commit_state(self, state: dict[str, Any]) -> None:
        _write_self_hashed(self.state_path, state, "state_sha256")

    def _required_unit_keys(self) -> set[str]:
        return {
            f"{candidate['candidate_id']}::fold_{fold}"
            for candidate in self.candidates
            for fold in self.inner_fold_ids
        }

    def _membership_digest(self, fold: int, partition: str) -> str:
        subjects = self.memberships["inner_folds"][str(fold)][
            f"{partition}_subject_keys"
        ]
        return sha256_mapping({"subject_keys": subjects})

    def _validated_units(
        self, units: Mapping[str, Any], *, require_complete: bool
    ) -> None:
        required = self._required_unit_keys()
        if not set(units).issubset(required):
            raise TrainingControlError("UNKNOWN_COMMITTED_INNER_HPO_UNIT")
        if require_complete and set(units) != required:
            raise TrainingControlError("WINNER_REQUIRES_COMPLETE_COMMITTED_UNITS")
        candidate_by_id = {
            str(candidate["candidate_id"]): candidate for candidate in self.candidates
        }
        backend_sha = self.run_contract.payload["implementation_contract"][
            "execution_backend"
        ]["implementation_sha256"]
        for key, row in units.items():
            if not isinstance(row, Mapping) or row.get("status") != "COMPLETED":
                raise TrainingControlError("MALFORMED_COMMITTED_INNER_HPO_UNIT")
            candidate_id, fold_label = key.split("::fold_", maxsplit=1)
            fold = int(fold_label)
            candidate = candidate_by_id[candidate_id]
            score = row.get("integrated_brier_score")
            if (
                row.get("candidate_id") != candidate_id
                or row.get("family") != candidate["family"]
                or row.get("inner_fold") != fold
                or row.get("run_contract_sha256") != self.run_contract.digest
                or row.get("execution_backend_sha256") != backend_sha
                or row.get("train_subjects_sha256")
                != self._membership_digest(fold, "train")
                or row.get("validation_subjects_sha256")
                != self._membership_digest(fold, "validation")
                or isinstance(score, bool)
                or not isinstance(score, Real)
                or not math.isfinite(float(score))
            ):
                raise TrainingControlError("MALFORMED_COMMITTED_INNER_HPO_UNIT")

    def _summaries(self, units: Mapping[str, Any]) -> list[dict[str, Any]]:
        self._validated_units(units, require_complete=True)
        summaries: list[dict[str, Any]] = []
        for candidate in self.candidates:
            scores = [
                float(
                    units[f"{candidate['candidate_id']}::fold_{fold}"][
                        "integrated_brier_score"
                    ]
                )
                for fold in self.inner_fold_ids
            ]
            standard_error = (
                float(np.std(scores, ddof=1) / math.sqrt(len(scores)))
                if len(scores) > 1
                else 0.0
            )
            summaries.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "family": candidate["family"],
                    "complexity_rank": int(candidate["complexity_rank"]),
                    "inner_fold_ibs": scores,
                    "mean_ibs": float(np.mean(scores)),
                    "standard_error": standard_error,
                    "status": "COMPLETED",
                }
            )
        return summaries

    def _load_and_revalidate_winner(
        self, state: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not self.winner_path.exists():
            if state.get("status") == "INNER_HPO_DONE":
                raise TrainingControlError("COMPLETED_STATE_MISSING_WINNER_LOCK")
            return None
        payload = _verified_payload(self.winner_path, "winner_lock_sha256")
        if payload.get("run_contract_sha256") != self.run_contract.digest:
            raise RunContractMismatch("RUN_CONTRACT_MISMATCH:winner_lock")
        units = state.get("completed_units", {})
        self._validated_units(units, require_complete=True)
        summaries = self._summaries(units)
        expected_winner = select_inner_winner(summaries)
        body = {key: value for key, value in payload.items() if key != "winner_lock_sha256"}
        expected = {
            "schema_version": "2.1",
            "status": "WINNER_LOCKED",
            "run_contract_sha256": self.run_contract.digest,
            "selection_source": "mean_inner_ibs_one_standard_error_only",
            "required_inner_fold_count": len(self.inner_fold_ids),
            "required_candidate_count": len(self.candidates),
            "candidate_summaries": summaries,
            "winner": expected_winner,
        }
        if body != expected:
            raise TrainingControlError("WINNER_LOCK_DOES_NOT_MATCH_COMMITTED_UNITS")
        if state.get("status") not in {"INNER_HPO_IN_PROGRESS", "INNER_HPO_DONE"}:
            raise TrainingControlError("WINNER_STATE_INVALID")
        if state.get("status") == "INNER_HPO_IN_PROGRESS":
            state["status"] = "INNER_HPO_DONE"
            state["winner_lock_sha256"] = payload["winner_lock_sha256"]
            self._commit_state(state)
        elif state.get("winner_lock_sha256") != payload["winner_lock_sha256"]:
            raise TrainingControlError("WINNER_STATE_HASH_MISMATCH")
        return body

    def _validate_data(self, data: InnerSelectionData) -> None:
        expected = set(self.memberships["outer_train_subject_keys"])
        actual = set(data.subject_keys)
        if actual != expected:
            raise TrainingControlError("OUTER_TRAIN_DATASET_MEMBERSHIP_MISMATCH")
        if actual.intersection(self.memberships["outer_test_subject_keys"]):
            raise TrainingControlError("OUTER_TEST_ROW_PRESENT_DURING_INNER_HPO")

    def _validate_backend(
        self, execution_backend: _HashBoundSyntheticBackend | None
    ) -> dict[str, Any]:
        actual = _backend_descriptor(execution_backend)
        expected = self.run_contract.payload["implementation_contract"][
            "execution_backend"
        ]
        if actual != expected:
            raise TrainingControlError("EXECUTION_BACKEND_CONTRACT_MISMATCH")
        if actual["test_only"] is True and execution_backend is None:
            raise TrainingControlError("HASH_BOUND_SYNTHETIC_BACKEND_REQUIRED")
        if actual["test_only"] is False and execution_backend is not None:
            raise TrainingControlError("INJECTED_PRODUCTION_SCORER_FORBIDDEN")
        return actual

    def run(
        self,
        *,
        data: InnerSelectionData,
        execution_backend: _HashBoundSyntheticBackend | None = None,
        interrupt_after_completed_units: int | None = None,
    ) -> dict[str, Any]:
        self._validate_data(data)
        backend = self._validate_backend(execution_backend)
        state = self._read_state()
        locked = self._load_and_revalidate_winner(state)
        if locked is not None:
            return locked
        units = state["completed_units"]
        for candidate in self.candidates:
            for fold in self.inner_fold_ids:
                key = f"{candidate['candidate_id']}::fold_{fold}"
                if key in units:
                    continue
                membership = self.memberships["inner_folds"][str(fold)]
                train = data.subset(membership["train_subject_keys"])
                validation = data.subset(membership["validation_subject_keys"])
                seed = int(self.run_contract.payload["seed"]) + fold
                if execution_backend is None:
                    score = _canonical_inner_ibs(
                        candidate=candidate,
                        train=train,
                        validation=validation,
                    )
                else:
                    score = execution_backend.score_unit(
                        candidate=candidate,
                        inner_fold=fold,
                        seed=seed,
                        train=train,
                        validation=validation,
                    )
                if (
                    isinstance(score, bool)
                    or not isinstance(score, Real)
                    or not math.isfinite(float(score))
                ):
                    raise TrainingControlError("FINITE_INNER_IBS_REQUIRED")
                units[key] = {
                    "candidate_id": candidate["candidate_id"],
                    "family": candidate["family"],
                    "inner_fold": fold,
                    "integrated_brier_score": float(score),
                    "status": "COMPLETED",
                    "run_contract_sha256": self.run_contract.digest,
                    "execution_backend_sha256": backend["implementation_sha256"],
                    "train_subjects_sha256": self._membership_digest(fold, "train"),
                    "validation_subjects_sha256": self._membership_digest(
                        fold, "validation"
                    ),
                }
                state["completed_units"] = units
                self._commit_state(state)
                if (
                    interrupt_after_completed_units is not None
                    and len(units) >= int(interrupt_after_completed_units)
                ):
                    raise SyntheticTrainingInterruption("SYNTHETIC_INTERRUPT_AFTER_UNIT")
        summaries = self._summaries(units)
        winner = select_inner_winner(summaries)
        locked = {
            "schema_version": "2.1",
            "status": "WINNER_LOCKED",
            "run_contract_sha256": self.run_contract.digest,
            "selection_source": "mean_inner_ibs_one_standard_error_only",
            "required_inner_fold_count": len(self.inner_fold_ids),
            "required_candidate_count": len(self.candidates),
            "candidate_summaries": summaries,
            "winner": winner,
        }
        _write_self_hashed(self.winner_path, locked, "winner_lock_sha256")
        winner_payload = _verified_payload(self.winner_path, "winner_lock_sha256")
        state.update(
            {
                "status": "INNER_HPO_DONE",
                "winner_lock_sha256": winner_payload["winner_lock_sha256"],
            }
        )
        self._commit_state(state)
        return locked
