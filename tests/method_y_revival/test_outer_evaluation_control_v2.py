from __future__ import annotations

import json

import pytest


def test_outer_result_commit_recovers_after_interruption_without_second_access(tmp_path):
    from src.method_y_revival.outer_evaluation_control import (
        OuterEvaluationInterruption,
        OuterFoldController,
    )

    calls = {"count": 0}

    def evaluate_locked_outer_test():
        calls["count"] += 1
        return {"integrated_brier_score": 0.19, "uno_c_index": 0.62}

    controller = OuterFoldController(
        tmp_path,
        fold=0,
        run_contract_sha256="a" * 64,
    )
    controller.lock_checkpoint("b" * 64)
    with pytest.raises(OuterEvaluationInterruption, match="AFTER_OUTER_RESULT_COMMIT"):
        controller.evaluate_once(
            evaluate_locked_outer_test,
            interrupt_after_result_commit=True,
        )

    assert calls["count"] == 1
    assert controller.result_path.is_file()
    recovered = OuterFoldController(
        tmp_path,
        fold=0,
        run_contract_sha256="a" * 64,
    ).evaluate_once(evaluate_locked_outer_test)

    assert recovered == {"integrated_brier_score": 0.19, "uno_c_index": 0.62}
    assert calls["count"] == 1
    assert json.loads(controller.state_path.read_text(encoding="utf-8"))["state"] == (
        "OUTER_EVAL_COMMITTED"
    )
    controller.finalize()
    assert json.loads(controller.state_path.read_text(encoding="utf-8"))["state"] == "FINALIZED"
    assert controller.evaluate_once(evaluate_locked_outer_test) == recovered
    assert calls["count"] == 1


def test_outer_evaluation_requires_checkpoint_lock_and_contract_match(tmp_path):
    from src.method_y_revival.contracts import RunContractMismatch
    from src.method_y_revival.outer_evaluation_control import (
        OuterEvaluationControlError,
        OuterFoldController,
    )

    controller = OuterFoldController(
        tmp_path,
        fold=1,
        run_contract_sha256="c" * 64,
    )
    with pytest.raises(OuterEvaluationControlError, match="CHECKPOINT_LOCK_REQUIRED"):
        controller.evaluate_once(lambda: {"integrated_brier_score": 0.2})
    controller.lock_checkpoint("d" * 64)

    with pytest.raises(RunContractMismatch, match="RUN_CONTRACT_MISMATCH"):
        OuterFoldController(
            tmp_path,
            fold=1,
            run_contract_sha256="e" * 64,
        )


def test_failed_or_missing_outer_fold_makes_aggregate_incomplete(tmp_path):
    from src.method_y_revival.outer_evaluation_control import (
        OuterFoldController,
        summarize_outer_evaluation,
    )

    first = OuterFoldController(tmp_path, fold=0, run_contract_sha256="f" * 64)
    first.lock_checkpoint("1" * 64)
    first.evaluate_once(lambda: {"integrated_brier_score": 0.2})

    failed = OuterFoldController(tmp_path, fold=1, run_contract_sha256="f" * 64)
    failed.lock_checkpoint("2" * 64)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        failed.evaluate_once(lambda: (_ for _ in ()).throw(RuntimeError("synthetic failure")))

    summary = summarize_outer_evaluation(
        tmp_path,
        expected_folds=3,
        run_contract_sha256="f" * 64,
    )

    assert summary["status"] == "FAILED_INCOMPLETE_OUTER_CV"
    assert summary["outer_folds_completed"] == 1
    assert summary["metrics_aggregated"] is False
    assert "mean_integrated_brier_score" not in summary


def test_ambiguous_interruption_before_result_commit_never_reaccesses_outer_test(tmp_path):
    from src.method_y_revival.outer_evaluation_control import (
        OuterEvaluationControlError,
        OuterFoldController,
    )

    calls = {"count": 0}

    class FatalSyntheticInterruption(BaseException):
        pass

    def interrupted_access():
        calls["count"] += 1
        raise FatalSyntheticInterruption

    controller = OuterFoldController(
        tmp_path,
        fold=2,
        run_contract_sha256="3" * 64,
    )
    controller.lock_checkpoint("4" * 64)
    with pytest.raises(FatalSyntheticInterruption):
        controller.evaluate_once(interrupted_access)

    with pytest.raises(
        OuterEvaluationControlError,
        match="AMBIGUOUS_OUTER_ACCESS_WITHOUT_COMMITTED_RESULT",
    ):
        OuterFoldController(
            tmp_path,
            fold=2,
            run_contract_sha256="3" * 64,
        ).evaluate_once(interrupted_access)
    assert calls["count"] == 1
