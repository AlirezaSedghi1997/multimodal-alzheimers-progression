from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from src.method_y_revival.contracts import RunContract, sha256_mapping


OUTER_TRAIN = tuple(f"subject_{index:02d}" for index in range(6))
OUTER_TEST = ("outer_test_00", "outer_test_01")
INNER_MEMBERSHIPS = {
    "0": {
        "train_subject_keys": list(OUTER_TRAIN[2:]),
        "validation_subject_keys": list(OUTER_TRAIN[:2]),
    },
    "1": {
        "train_subject_keys": list(OUTER_TRAIN[:2] + OUTER_TRAIN[4:]),
        "validation_subject_keys": list(OUTER_TRAIN[2:4]),
    },
    "2": {
        "train_subject_keys": list(OUTER_TRAIN[:4]),
        "validation_subject_keys": list(OUTER_TRAIN[4:]),
    },
}
SYNTHETIC_BACKEND_CONTRACT = {
    "backend_id": "synthetic_hash_bound_inner_ibs_v1",
    "implementation_sha256": "d" * 64,
    "test_only": True,
}


class BoundSyntheticBackend:
    backend_id = SYNTHETIC_BACKEND_CONTRACT["backend_id"]
    implementation_sha256 = SYNTHETIC_BACKEND_CONTRACT["implementation_sha256"]
    test_only = True

    def __init__(self, *, fail_if_called: bool = False) -> None:
        self.calls: list[tuple[str, int, tuple[str, ...], tuple[str, ...]]] = []
        self.fail_if_called = fail_if_called

    def score_unit(self, *, candidate, inner_fold, seed, train, validation):
        if self.fail_if_called:
            raise AssertionError("a committed unit was executed again")
        self.calls.append(
            (
                candidate["candidate_id"],
                inner_fold,
                tuple(train.subject_keys),
                tuple(validation.subject_keys),
            )
        )
        assert seed == 31 + inner_fold
        values = {
            "coxnet": [0.207, 0.207, 0.207],
            "gbsa": [0.180, 0.200, 0.220],
            "rsf": [0.250, 0.250, 0.250],
        }
        return values[candidate["family"]][inner_fold]


def _contract(
    seed: int = 31,
    *,
    backend_contract: dict | None = None,
) -> tuple[dict, RunContract]:
    from src.method_y_revival.model_family_adapters import (
        build_matched_candidate_registry,
    )
    from src.method_y_revival.training_control import build_training_run_contract

    registry = build_matched_candidate_registry(seed=seed, budget_per_family=1)
    contract = build_training_run_contract(
        branch="amyloid_source_regional",
        outer_fold=0,
        seed=seed,
        endpoint_sha256="a" * 64,
        feature_sha256="b" * 64,
        split_digest="c" * 64,
        inner_fold_ids=[0, 1, 2],
        candidate_registry=registry,
        outer_train_subject_keys=OUTER_TRAIN,
        outer_test_subject_keys=OUTER_TEST,
        inner_fold_memberships=INNER_MEMBERSHIPS,
        execution_backend_contract=backend_contract or SYNTHETIC_BACKEND_CONTRACT,
    )
    return registry, contract


def _data(*, include_outer_test: bool = False):
    from src.method_y_revival.training_control import InnerSelectionData

    subjects = list(OUTER_TRAIN)
    if include_outer_test:
        subjects.append(OUTER_TEST[0])
    count = len(subjects)
    return InnerSelectionData.from_arrays(
        subject_keys=subjects,
        features=np.column_stack(
            [
                np.arange(count, dtype=float),
                np.linspace(0.1, 0.9, count),
            ]
        ),
        event=np.asarray([(index % 2) == 0 for index in range(count)]),
        time_days=np.asarray([100.0 + 50.0 * index for index in range(count)]),
    )


def _controller(tmp_path, *, contract=None, registry=None):
    from src.method_y_revival.training_control import InnerHPOController

    if contract is None or registry is None:
        registry, contract = _contract()
    return InnerHPOController(
        tmp_path,
        run_contract=contract,
        candidate_registry=registry,
        inner_fold_ids=[0, 1, 2],
        allow_test_backend=True,
    )


def test_inner_hpo_api_rejects_unbound_scorer_and_cannot_lock_an_early_winner(tmp_path):
    from src.method_y_revival.training_control import SyntheticTrainingInterruption

    controller = _controller(tmp_path / "interrupted")
    assert "outer" not in inspect.signature(controller.run).parameters
    assert "score_inner_fold" not in inspect.signature(controller.run).parameters
    with pytest.raises(TypeError):
        controller.run(data=_data(), score_inner_fold=lambda *_: 0.0)
    with pytest.raises(SyntheticTrainingInterruption, match="SYNTHETIC_INTERRUPT_AFTER_UNIT"):
        controller.run(
            data=_data(),
            execution_backend=BoundSyntheticBackend(),
            interrupt_after_completed_units=2,
        )

    assert controller.state_path.is_file()
    assert not controller.winner_path.exists()


def test_controller_extracts_only_locked_inner_memberships_and_rejects_outer_rows(tmp_path):
    from src.method_y_revival.training_control import TrainingControlError

    backend = BoundSyntheticBackend()
    result = _controller(tmp_path / "valid").run(
        data=_data(), execution_backend=backend
    )

    assert result["status"] == "WINNER_LOCKED"
    assert len(backend.calls) == 9
    for _, fold, train_subjects, validation_subjects in backend.calls:
        expected = INNER_MEMBERSHIPS[str(fold)]
        assert set(train_subjects) == set(expected["train_subject_keys"])
        assert set(validation_subjects) == set(expected["validation_subject_keys"])
        assert set(train_subjects).isdisjoint(validation_subjects)
        assert set(train_subjects + validation_subjects).isdisjoint(OUTER_TEST)

    forbidden_backend = BoundSyntheticBackend()
    with pytest.raises(TrainingControlError, match="OUTER_TRAIN_DATASET_MEMBERSHIP_MISMATCH"):
        _controller(tmp_path / "outer-row").run(
            data=_data(include_outer_test=True),
            execution_backend=forbidden_backend,
        )
    assert forbidden_backend.calls == []


def test_hash_bound_synthetic_backend_rejects_an_implementation_substitution(tmp_path):
    from src.method_y_revival.training_control import TrainingControlError

    backend = BoundSyntheticBackend()
    backend.implementation_sha256 = "e" * 64
    with pytest.raises(TrainingControlError, match="EXECUTION_BACKEND_CONTRACT_MISMATCH"):
        _controller(tmp_path).run(data=_data(), execution_backend=backend)
    assert backend.calls == []


def test_synthetic_backend_contract_requires_explicit_test_interface(tmp_path):
    from src.method_y_revival.training_control import (
        InnerHPOController,
        TrainingControlError,
    )

    registry, contract = _contract()
    with pytest.raises(TrainingControlError, match="TEST_BACKEND_NOT_AUTHORIZED"):
        InnerHPOController(
            tmp_path,
            run_contract=contract,
            candidate_registry=registry,
            inner_fold_ids=[0, 1, 2],
        )


def test_interrupted_inner_hpo_resumes_without_repeating_units_and_matches_uninterrupted(tmp_path):
    from src.method_y_revival.training_control import SyntheticTrainingInterruption

    interrupted_root = tmp_path / "interrupted"
    calls_backend = BoundSyntheticBackend()
    with pytest.raises(SyntheticTrainingInterruption):
        _controller(interrupted_root).run(
            data=_data(),
            execution_backend=calls_backend,
            interrupt_after_completed_units=4,
        )
    resumed = _controller(interrupted_root).run(
        data=_data(), execution_backend=calls_backend
    )

    uninterrupted = _controller(tmp_path / "uninterrupted").run(
        data=_data(), execution_backend=BoundSyntheticBackend()
    )

    called_units = [(candidate, fold) for candidate, fold, _, _ in calls_backend.calls]
    assert len(called_units) == 9
    assert len(set(called_units)) == 9
    assert resumed["candidate_summaries"] == uninterrupted["candidate_summaries"]
    assert resumed["winner"] == uninterrupted["winner"]
    assert resumed["winner"]["candidate_id"] == "coxnet_00"
    assert resumed["selection_source"] == "mean_inner_ibs_one_standard_error_only"


def test_preexisting_winner_without_complete_committed_units_is_rejected(tmp_path):
    from src.method_y_revival.training_control import TrainingControlError

    registry, contract = _contract()
    controller = _controller(tmp_path / "forged-winner", contract=contract, registry=registry)
    forged = {
        "schema_version": "2.0",
        "status": "WINNER_LOCKED",
        "run_contract_sha256": contract.digest,
        "selection_source": "mean_inner_ibs_one_standard_error_only",
        "required_inner_fold_count": 3,
        "required_candidate_count": 3,
        "candidate_summaries": [],
        "winner": {"candidate_id": "coxnet_00", "mean_ibs": 0.0},
    }
    forged["winner_lock_sha256"] = sha256_mapping(forged)
    controller.winner_path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(
        TrainingControlError,
        match="WINNER_REQUIRES_COMPLETE_COMMITTED_UNITS",
    ):
        controller.run(data=_data(), execution_backend=BoundSyntheticBackend())


def test_completed_winner_is_recomputed_from_state_and_tampering_fails_closed(tmp_path):
    from src.method_y_revival.training_control import TrainingControlError

    root = tmp_path / "winner-revalidation"
    controller = _controller(root)
    controller.run(data=_data(), execution_backend=BoundSyntheticBackend())
    winner = json.loads(controller.winner_path.read_text(encoding="utf-8"))
    winner["winner"]["candidate_id"] = "rsf_00"
    winner["winner_lock_sha256"] = sha256_mapping(
        {key: value for key, value in winner.items() if key != "winner_lock_sha256"}
    )
    controller.winner_path.write_text(json.dumps(winner), encoding="utf-8")

    with pytest.raises(TrainingControlError, match="WINNER_LOCK_DOES_NOT_MATCH_COMMITTED_UNITS"):
        _controller(root).run(
            data=_data(), execution_backend=BoundSyntheticBackend(fail_if_called=True)
        )


def test_run_contract_binds_code_runtime_metric_model_preprocessing_and_dependencies():
    _, contract = _contract()
    implementation = contract.payload.get("implementation_contract")

    assert isinstance(implementation, dict)
    assert set(implementation) == {
        "code_sha256",
        "dependency_sha256",
        "metric_sha256",
        "model_factory_sha256",
        "preprocessing_sha256",
        "python_runtime_sha256",
        "execution_backend",
    }
    for key in (
        "code_sha256",
        "dependency_sha256",
        "metric_sha256",
        "model_factory_sha256",
        "preprocessing_sha256",
        "python_runtime_sha256",
    ):
        assert len(implementation[key]) == 64
    assert implementation["execution_backend"] == SYNTHETIC_BACKEND_CONTRACT


def test_canonical_controller_builds_fold_local_estimators_and_computes_finite_ibs(tmp_path):
    from src.method_y_revival.model_family_adapters import (
        build_matched_candidate_registry,
    )
    from src.method_y_revival.training_control import (
        InnerHPOController,
        InnerSelectionData,
        build_training_run_contract,
        canonical_execution_backend_contract,
    )

    subjects = tuple(f"canonical_{index:02d}" for index in range(18))
    memberships = {}
    for fold in range(3):
        validation = list(subjects[fold * 6 : (fold + 1) * 6])
        memberships[str(fold)] = {
            "train_subject_keys": [
                subject for subject in subjects if subject not in validation
            ],
            "validation_subject_keys": validation,
        }
    registry = build_matched_candidate_registry(seed=31, budget_per_family=1)
    backend_contract = canonical_execution_backend_contract()
    contract = build_training_run_contract(
        branch="amyloid_source_regional",
        outer_fold=0,
        seed=31,
        endpoint_sha256="a" * 64,
        feature_sha256="b" * 64,
        split_digest="c" * 64,
        inner_fold_ids=[0, 1, 2],
        candidate_registry=registry,
        outer_train_subject_keys=subjects,
        outer_test_subject_keys=["canonical_outer_test_00"],
        inner_fold_memberships=memberships,
        execution_backend_contract=backend_contract,
    )
    rng = np.random.default_rng(31)
    data = InnerSelectionData.from_arrays(
        subject_keys=subjects,
        features=rng.normal(size=(18, 4)),
        event=np.asarray([(index % 2) == 0 for index in range(18)]),
        time_days=np.tile(np.asarray([100, 200, 300, 400, 500, 600]), 3),
    )

    result = InnerHPOController(
        tmp_path,
        run_contract=contract,
        candidate_registry=registry,
        inner_fold_ids=[0, 1, 2],
    ).run(data=data)

    assert result["status"] == "WINNER_LOCKED"
    state = json.loads((tmp_path / "inner_hpo_state.json").read_text("utf-8"))
    assert len(state["completed_units"]) == 9
    assert all(
        np.isfinite(row["integrated_brier_score"])
        and row["execution_backend_sha256"]
        == backend_contract["implementation_sha256"]
        for row in state["completed_units"].values()
    )


def test_resume_with_changed_training_implementation_contract_fails_closed(tmp_path):
    from src.method_y_revival.contracts import RunContractMismatch
    from src.method_y_revival.training_control import SyntheticTrainingInterruption

    registry, contract = _contract()
    root = tmp_path / "changed-implementation"
    with pytest.raises(SyntheticTrainingInterruption):
        _controller(root, contract=contract, registry=registry).run(
            data=_data(),
            execution_backend=BoundSyntheticBackend(),
            interrupt_after_completed_units=1,
        )

    mutated_payload = json.loads(json.dumps(contract.payload))
    mutated_payload["implementation_contract"]["code_sha256"] = "f" * 64
    changed_contract = RunContract.from_mapping(mutated_payload)
    with pytest.raises(RunContractMismatch, match="RUN_CONTRACT_MISMATCH"):
        _controller(root, contract=changed_contract, registry=registry)


def test_resume_with_changed_seed_fails_closed(tmp_path):
    from src.method_y_revival.contracts import RunContractMismatch
    from src.method_y_revival.training_control import SyntheticTrainingInterruption

    registry, contract = _contract(seed=31)
    root = tmp_path / "same-output"
    with pytest.raises(SyntheticTrainingInterruption):
        _controller(root, contract=contract, registry=registry).run(
            data=_data(),
            execution_backend=BoundSyntheticBackend(),
            interrupt_after_completed_units=1,
        )

    changed_registry, changed_contract = _contract(seed=32)
    with pytest.raises(RunContractMismatch, match="RUN_CONTRACT_MISMATCH"):
        _controller(root, contract=changed_contract, registry=changed_registry)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda registry: registry["preprocessing_policy"].update(
                {"scaling": "fit_on_all_data"}
            ),
            "FOLD_LOCAL_PREPROCESSING_REQUIRED",
        ),
        (
            lambda registry: registry["families"]["coxnet"][0].update(
                {"family": "rsf"}
            ),
            "CANDIDATE_FAMILY_MISMATCH",
        ),
        (
            lambda registry: registry["families"]["rsf"][0].update(
                {"hpo_budget_units": 2}
            ),
            "EQUAL_HPO_BUDGET_UNITS_REQUIRED",
        ),
    ],
)
def test_training_contract_rejects_unmatched_registry_mutations(mutation, error):
    from src.method_y_revival.model_family_adapters import (
        build_matched_candidate_registry,
    )
    from src.method_y_revival.training_control import (
        TrainingControlError,
        build_training_run_contract,
    )

    registry = build_matched_candidate_registry(seed=31, budget_per_family=1)
    mutation(registry)
    with pytest.raises(TrainingControlError, match=error):
        build_training_run_contract(
            branch="amyloid_source_regional",
            outer_fold=0,
            seed=31,
            endpoint_sha256="a" * 64,
            feature_sha256="b" * 64,
            split_digest="c" * 64,
            inner_fold_ids=[0, 1, 2],
            candidate_registry=registry,
            outer_train_subject_keys=OUTER_TRAIN,
            outer_test_subject_keys=OUTER_TEST,
            inner_fold_memberships=INNER_MEMBERSHIPS,
            execution_backend_contract=SYNTHETIC_BACKEND_CONTRACT,
        )


@pytest.mark.parametrize(
    ("outer_fold", "inner_folds", "error"),
    [
        (5, [0, 1, 2], "EXACT_FIVE_OUTER_FOLDS_REQUIRED"),
        (0, [0, 1], "EXACT_THREE_INNER_FOLDS_REQUIRED"),
    ],
)
def test_training_contract_rejects_wrong_nested_cv_shape(outer_fold, inner_folds, error):
    from src.method_y_revival.model_family_adapters import (
        build_matched_candidate_registry,
    )
    from src.method_y_revival.training_control import (
        TrainingControlError,
        build_training_run_contract,
    )

    registry = build_matched_candidate_registry(seed=31, budget_per_family=1)
    with pytest.raises(TrainingControlError, match=error):
        build_training_run_contract(
            branch="amyloid_source_regional",
            outer_fold=outer_fold,
            seed=31,
            endpoint_sha256="a" * 64,
            feature_sha256="b" * 64,
            split_digest="c" * 64,
            inner_fold_ids=inner_folds,
            candidate_registry=registry,
            outer_train_subject_keys=OUTER_TRAIN,
            outer_test_subject_keys=OUTER_TEST,
            inner_fold_memberships=INNER_MEMBERSHIPS,
            execution_backend_contract=SYNTHETIC_BACKEND_CONTRACT,
        )
