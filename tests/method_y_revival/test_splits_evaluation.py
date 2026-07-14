from __future__ import annotations

import json

import numpy as np
import pytest


def _split_rows():
    rows = []
    treatments = ["placebo", "treatment"]
    for subject_index in range(60):
        subject = f"sub-{subject_index:03d}"
        event = subject_index % 2
        treatment = treatments[(subject_index // 2) % 2]
        for session in ("baseline", "followup"):
            rows.append(
                {
                    "subject_key": subject,
                    "session_key": f"{subject}-{session}",
                    "event": event,
                    "treatment_arm": treatment,
                }
            )
    return rows


def test_nested_splits_keep_every_session_of_a_subject_together():
    from src.method_y_revival.splits import build_nested_subject_splits

    contract = build_nested_subject_splits(_split_rows(), outer_folds=5, inner_folds=3, seed=20260714)

    assert len(contract["outer_folds"]) == 5
    all_outer_test = []
    for outer in contract["outer_folds"]:
        train = set(outer["train_subjects"])
        test = set(outer["test_subjects"])
        assert train.isdisjoint(test)
        all_outer_test.extend(test)
        for inner in outer["inner_folds"]:
            assert set(inner["train_subjects"]).isdisjoint(inner["validation_subjects"])
            assert set(inner["validation_subjects"]).issubset(train)
    assert sorted(all_outer_test) == sorted({row["subject_key"] for row in _split_rows()})


def test_split_digest_is_reproducible_and_seed_sensitive():
    from src.method_y_revival.splits import build_nested_subject_splits

    first = build_nested_subject_splits(_split_rows(), outer_folds=5, inner_folds=3, seed=1)
    repeated = build_nested_subject_splits(list(reversed(_split_rows())), outer_folds=5, inner_folds=3, seed=1)
    changed = build_nested_subject_splits(_split_rows(), outer_folds=5, inner_folds=3, seed=2)

    assert first["split_digest"] == repeated["split_digest"]
    assert first["split_digest"] != changed["split_digest"]


def test_survival_horizons_outside_common_support_are_not_estimable():
    from src.method_y_revival.evaluation import valid_survival_horizons

    report = valid_survival_horizons(
        train_times=np.array([100.0, 600.0, 1200.0, 1700.0]),
        test_times=np.array([200.0] * 10 + [800.0] * 20 + [1300.0] * 20),
        requested_horizons=[547.875, 1095.75, 1640.0],
        min_at_risk=20,
    )

    assert report[547.875]["status"] == "ESTIMABLE"
    assert report[1095.75]["status"] == "ESTIMABLE"
    assert report[1640.0]["status"] == "NOT_ESTIMABLE"


def test_incomplete_outer_cv_cannot_be_aggregated_as_complete():
    from src.method_y_revival.evaluation import EvaluationContractError, require_complete_outer_cv

    with pytest.raises(EvaluationContractError, match="FAILED_INCOMPLETE_OUTER_CV"):
        require_complete_outer_cv([{"fold": index, "status": "COMPLETED"} for index in range(4)], expected_folds=5)


def test_committed_outer_result_is_reused_without_calling_test_again(tmp_path):
    from src.method_y_revival.evaluation import OuterEvaluationStore

    calls = {"count": 0}

    def evaluator():
        calls["count"] += 1
        return {"uno_c_index": 0.61}

    store = OuterEvaluationStore(tmp_path, fold=0, run_contract_digest="a" * 64)
    first = store.evaluate_once(checkpoint_digest="b" * 64, evaluator=evaluator)
    second = store.evaluate_once(checkpoint_digest="b" * 64, evaluator=evaluator)

    assert first == second
    assert calls["count"] == 1
    assert json.loads(store.state_path.read_text(encoding="utf-8"))["state"] == "OUTER_EVAL_COMMITTED"


def test_one_standard_error_rule_selects_the_simplest_eligible_candidate():
    from src.method_y_revival.survival_models import select_inner_winner

    winner = select_inner_winner(
        [
            {"candidate_id": "complex", "mean_ibs": 0.200, "standard_error": 0.010, "complexity_rank": 3},
            {"candidate_id": "simple", "mean_ibs": 0.207, "standard_error": 0.010, "complexity_rank": 1},
            {"candidate_id": "bad", "mean_ibs": 0.230, "standard_error": 0.002, "complexity_rank": 0},
        ]
    )

    assert winner["candidate_id"] == "simple"
