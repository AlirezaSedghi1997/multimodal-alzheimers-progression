from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch


TRAIN_SCRIPT = Path(__file__).parents[1] / "scripts" / "train_3d_mri.py"


def test_seed_everything_repeats_torch_stream():
    from src.utils.reproducibility import seed_everything

    seed_everything(19)
    expected = torch.rand(5)
    seed_everything(19)

    assert torch.equal(torch.rand(5), expected)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False


def test_rng_state_round_trip_restores_python_numpy_and_torch():
    from src.utils.reproducibility import capture_rng_state, restore_rng_state, seed_everything

    seed_everything(73)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    random.random()
    np.random.random()
    torch.rand(4)
    restore_rng_state(state)

    actual = (random.random(), float(np.random.random()), torch.rand(4))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_rng_state_is_safe_for_weights_only_torch_load(tmp_path: Path):
    from src.utils.reproducibility import capture_rng_state

    path = tmp_path / "rng.pt"
    torch.save({"rng_state": capture_rng_state()}, path)
    loaded = torch.load(path, weights_only=True)

    assert isinstance(loaded["rng_state"]["numpy"][1], list)


def test_canonical_digest_is_order_independent_and_sensitive():
    from src.utils.reproducibility import canonical_digest

    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})
    assert canonical_digest({"a": 1}) != canonical_digest({"a": 2})


def test_source_tree_digest_changes_when_training_code_changes(tmp_path: Path):
    from src.utils.reproducibility import source_tree_digest

    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("x = 1\n", encoding="utf-8")
    second.write_text("y = 2\n", encoding="utf-8")
    before = source_tree_digest([second, first], base=tmp_path)
    assert before == source_tree_digest([first, second], base=tmp_path)
    second.write_text("y = 3\n", encoding="utf-8")
    assert source_tree_digest([first, second], base=tmp_path) != before


def test_run_contract_mismatch_fails_closed():
    from src.training.method_x_resume import assert_resume_contract, build_run_contract

    contract = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=40,
        seed=13,
        parameters={"lr": 1e-4, "scheduler": "cosine"},
        loader={"batch_size": 1, "num_workers": 0},
    )
    assert_resume_contract({"run_contract": contract, "run_contract_digest": contract["digest"]}, contract)

    changed = json.loads(json.dumps(contract))
    changed["parameters"]["lr"] = 5e-5
    with pytest.raises(RuntimeError, match="RUN_CONTRACT_MISMATCH"):
        assert_resume_contract(
            {"run_contract": contract, "run_contract_digest": contract["digest"]},
            changed,
        )


def test_source_only_contract_migration_does_not_allow_training_parameter_drift():
    from src.training.method_x_resume import build_run_contract, source_contract_migration_allowed

    old = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=40,
        seed=13,
        parameters={"lr": 1e-4, "source_contract_digest": "c" * 64},
        loader={"batch_size": 1, "num_workers": 0},
    )
    new = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=40,
        seed=13,
        parameters={"lr": 1e-4, "source_contract_digest": "d" * 64},
        loader={"batch_size": 1, "num_workers": 0},
    )
    assert source_contract_migration_allowed(
        {"run_contract": old, "run_contract_digest": old["digest"]}, new
    )
    drifted = json.loads(json.dumps(new))
    drifted["parameters"]["lr"] = 5e-5
    assert not source_contract_migration_allowed(
        {"run_contract": old, "run_contract_digest": old["digest"]}, drifted
    )


def test_assert_resume_contract_accepts_unique_source_history_ancestor():
    from src.training.method_x_resume import (
        assert_resume_contract,
        build_run_contract,
        source_history_binds_digest,
    )

    old = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=40,
        seed=13,
        parameters={"lr": 1e-4, "source_contract_digest": "c" * 64},
        loader={"batch_size": 1, "num_workers": 0},
    )
    mid = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=40,
        seed=13,
        parameters={"lr": 1e-4, "source_contract_digest": "d" * 64},
        loader={"batch_size": 1, "num_workers": 0},
    )
    new = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=40,
        seed=13,
        parameters={"lr": 1e-4, "source_contract_digest": "e" * 64},
        loader={"batch_size": 1, "num_workers": 0},
    )
    payload = {"run_contract": old, "run_contract_digest": old["digest"]}
    history = [
        {
            "from_run_contract_digest": old["digest"],
            "to_run_contract_digest": mid["digest"],
        },
        {
            "from_run_contract_digest": mid["digest"],
            "to_run_contract_digest": new["digest"],
        },
    ]
    assert source_history_binds_digest(
        history=history, from_digest=old["digest"], to_digest=new["digest"]
    )
    with pytest.raises(RuntimeError, match="RUN_CONTRACT_MISMATCH"):
        assert_resume_contract(payload, new)
    assert_resume_contract(payload, new, source_history=history)


def test_assert_resume_contract_rejects_scientific_drift_with_source_history():
    from src.training.method_x_resume import assert_resume_contract, build_run_contract

    old = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=40,
        seed=13,
        parameters={"lr": 1e-4, "source_contract_digest": "c" * 64},
        loader={"batch_size": 1, "num_workers": 0},
    )
    drifted = build_run_contract(
        cache_sha256="a" * 64,
        split_digest="b" * 64,
        folds=5,
        epochs=40,
        seed=13,
        parameters={"lr": 5e-5, "source_contract_digest": "d" * 64},
        loader={"batch_size": 1, "num_workers": 0},
    )
    history = [
        {
            "from_run_contract_digest": old["digest"],
            "to_run_contract_digest": drifted["digest"],
        }
    ]
    with pytest.raises(RuntimeError, match="RUN_CONTRACT_MISMATCH"):
        assert_resume_contract(
            {"run_contract": old, "run_contract_digest": old["digest"]},
            drifted,
            source_history=history,
        )


def test_source_history_binds_digest_rejects_forks_and_cycles():
    from src.training.method_x_resume import source_history_binds_digest

    a = "a" * 64
    b = "b" * 64
    c = "c" * 64
    assert source_history_binds_digest(history=[], from_digest=a, to_digest=a)
    assert not source_history_binds_digest(
        history=[
            {"from_run_contract_digest": a, "to_run_contract_digest": b},
            {"from_run_contract_digest": a, "to_run_contract_digest": c},
        ],
        from_digest=a,
        to_digest=b,
    )
    assert not source_history_binds_digest(
        history=[
            {"from_run_contract_digest": a, "to_run_contract_digest": b},
            {"from_run_contract_digest": b, "to_run_contract_digest": a},
        ],
        from_digest=a,
        to_digest=c,
    )


def test_fold_state_machine_is_atomic_and_rejects_illegal_transition(tmp_path: Path):
    from src.training.method_x_resume import read_fold_state, transition_fold_state

    path = tmp_path / "fold_0_state.json"
    transition_fold_state(path, fold=0, target="TRAINING", contract_digest="c" * 64)
    transition_fold_state(path, fold=0, target="CHECKPOINT_LOCKED", contract_digest="c" * 64)
    transition_fold_state(path, fold=0, target="OUTER_EVAL_IN_PROGRESS", contract_digest="c" * 64)
    state = transition_fold_state(
        path,
        fold=0,
        target="OUTER_EVAL_DONE",
        contract_digest="c" * 64,
        details={"locked_result": "fold_0_outer_eval.json"},
    )
    assert state["stage"] == "OUTER_EVAL_DONE"
    assert read_fold_state(path) == state

    with pytest.raises(RuntimeError, match="ILLEGAL_FOLD_STATE_TRANSITION"):
        transition_fold_state(path, fold=0, target="TRAINING", contract_digest="c" * 64)


def test_fold_state_machine_rejects_cross_fold_state_reuse(tmp_path: Path):
    from src.training.method_x_resume import transition_fold_state

    path = tmp_path / "fold_state.json"
    transition_fold_state(path, fold=0, target="TRAINING", contract_digest="c" * 64)
    with pytest.raises(RuntimeError, match="FOLD_STATE_FOLD_MISMATCH"):
        transition_fold_state(path, fold=1, target="TRAINING", contract_digest="c" * 64)


def test_ambiguous_outer_evaluation_recovery_fails_closed():
    from src.training.method_x_resume import outer_evaluation_resume_action

    with pytest.raises(RuntimeError, match="OUTER_EVALUATION_NOT_CHECKPOINT_LOCKED"):
        outer_evaluation_resume_action(None, locked_result_exists=False)
    with pytest.raises(RuntimeError, match="OUTER_EVALUATION_NOT_CHECKPOINT_LOCKED"):
        outer_evaluation_resume_action("TRAINING", locked_result_exists=False)
    assert outer_evaluation_resume_action("CHECKPOINT_LOCKED", locked_result_exists=False) == "RUN"
    assert outer_evaluation_resume_action("OUTER_EVAL_DONE", locked_result_exists=True) == "REUSE"
    with pytest.raises(RuntimeError, match="AMBIGUOUS_OUTER_EVALUATION_STATE"):
        outer_evaluation_resume_action("OUTER_EVAL_IN_PROGRESS", locked_result_exists=False)
    assert (
        outer_evaluation_resume_action("OUTER_EVAL_IN_PROGRESS", locked_result_exists=True)
        == "REUSE_COMMITTED_RECOVER_STATE"
    )


def test_finalized_fold_reuses_committed_outer_payload_without_test_access():
    from src.training.method_x_resume import build_locked_outer_result, reuse_committed_outer_result

    contract_digest = "c" * 64
    split_digest = "d" * 64
    locked = build_locked_outer_result(
        fold=2,
        run_contract_digest=contract_digest,
        split_membership_digest=split_digest,
        y_true=[0, 1],
        y_prob=[0.2, 0.8],
        test_logits=[-1.0, 1.0],
    )
    checkpoint = {
        "fold": 2,
        "run_contract_digest": contract_digest,
        "split_membership_digest": split_digest,
        "locked_outer_test_evaluation": {
            "evaluation_count": 1,
            "metrics": {"fold": 2, "roc_auc": 1.0},
            "y_true": [0, 1],
            "y_prob": [0.2, 0.8],
            "test_logits": [-1.0, 1.0],
            "calibrated_prob": [0.25, 0.75],
            "split_membership_digest": split_digest,
            "result_digest": locked["result_digest"],
        },
    }

    reused = reuse_committed_outer_result(
        checkpoint,
        locked,
        expected_fold=2,
        expected_contract_digest=contract_digest,
    )

    assert reused["metrics"]["roc_auc"] == 1.0
    assert reused["y_true"] == [0, 1]
    assert reused["calibrated_prob"] == [0.25, 0.75]

    locked["evaluation_count"] = 2
    with pytest.raises(RuntimeError, match="COMMITTED_OUTER_EVALUATION_INVALID"):
        reuse_committed_outer_result(
            checkpoint,
            locked,
            expected_fold=2,
            expected_contract_digest=contract_digest,
        )
    locked = build_locked_outer_result(
        fold=2,
        run_contract_digest=contract_digest,
        split_membership_digest=split_digest,
        y_true=[0, 1],
        y_prob=[0.2, 0.8],
        test_logits=[-1.0, 1.0],
    )
    checkpoint["locked_outer_test_evaluation"]["y_prob"] = [0.3, 0.7]
    with pytest.raises(RuntimeError, match="COMMITTED_OUTER_EVALUATION_INVALID"):
        reuse_committed_outer_result(
            checkpoint,
            locked,
            expected_fold=2,
            expected_contract_digest=contract_digest,
        )


def test_fabricated_historical_compatibility_mapping_is_rejected():
    """Caller-created JSON must never become authority for a digest mismatch."""

    from src.training.method_x_resume import (
        build_locked_outer_result,
        outer_prediction_identity_digest,
        reuse_committed_outer_result,
    )

    contract_digest = "c" * 64
    split_digest = "d" * 64
    locked = build_locked_outer_result(
        fold=0,
        run_contract_digest=contract_digest,
        split_membership_digest=split_digest,
        y_true=[0, 1],
        y_prob=[0.2, 0.8],
        test_logits=[-1.0, 1.0],
    )
    checkpoint = {
        "backbone": "resnet18_3d",
        "fold": 0,
        "run_contract_digest": contract_digest,
        "split_membership_digest": split_digest,
        "locked_outer_test_evaluation": {
            "evaluation_count": 1,
            "metrics": {"fold": 0, "roc_auc": 1.0},
            "y_true": [0, 1],
            "y_prob": [0.2, 0.8],
            "test_logits": [-1.0, 1.0],
            "calibrated_prob": [0.25, 0.75],
            "split_membership_digest": split_digest,
            "result_digest": "e" * 64,
        },
    }
    evidence = {
        "schema_version": 1,
        "artifact": "method_x_historical_outer_result_compatibility_entry",
        "scope": "EXACT_FIVE_FROZEN_ARTIFACTS_ONLY",
        "backbone": "resnet18_3d",
        "fold": 0,
        "checkpoint_sha256": "a" * 64,
        "outer_file_sha256": "b" * 64,
        "run_contract_digest": contract_digest,
        "split_membership_digest": split_digest,
        "evaluation_count": 1,
        "embedded_result_digest": "e" * 64,
        "external_result_digest": locked["result_digest"],
        "prediction_identity_digest": outer_prediction_identity_digest(locked),
        "outer_loader_invocation_count": 0,
    }

    with pytest.raises(RuntimeError, match="COMMITTED_OUTER_EVALUATION_INVALID"):
        reuse_committed_outer_result(
            checkpoint,
            locked,
            expected_fold=0,
            expected_contract_digest=contract_digest,
            checkpoint_sha256="a" * 64,
            outer_file_sha256="b" * 64,
            compatibility_evidence=evidence,
        )


def test_pinned_historical_receipt_exposes_exactly_the_five_frozen_rows():
    from src.training.method_x_outer_compatibility import (
        HISTORICAL_COMPATIBILITY_RECEIPT,
        load_historical_compatibility_entry,
    )

    expected = {
        ("resnet18_3d", 0),
        ("resnet18_3d", 1),
        ("resnet34_3d", 0),
        ("resnet34_3d", 1),
        ("resnet34_3d", 2),
    }
    observed = set()
    for backbone, fold in expected:
        authority = load_historical_compatibility_entry(
            HISTORICAL_COMPATIBILITY_RECEIPT,
            backbone=backbone,
            fold=fold,
        )
        observed.add((authority.evidence["backbone"], authority.evidence["fold"]))
    assert observed == expected

    with pytest.raises(RuntimeError, match="HISTORICAL_OUTER_COMPATIBILITY_NOT_AUTHORIZED"):
        load_historical_compatibility_entry(
            HISTORICAL_COMPATIBILITY_RECEIPT,
            backbone="densenet121_3d",
            fold=0,
        )


def test_outer_digest_mismatch_stays_blocked_without_compatibility_evidence():
    from src.training.method_x_resume import build_locked_outer_result, reuse_committed_outer_result

    locked = build_locked_outer_result(
        fold=1,
        run_contract_digest="c" * 64,
        split_membership_digest="d" * 64,
        y_true=[0, 1],
        y_prob=[0.3, 0.7],
        test_logits=[-0.5, 0.5],
    )
    checkpoint = {
        "backbone": "resnet18_3d",
        "fold": 1,
        "run_contract_digest": "c" * 64,
        "split_membership_digest": "d" * 64,
        "locked_outer_test_evaluation": {
            "evaluation_count": 1,
            "metrics": {"fold": 1, "roc_auc": 1.0},
            "y_true": locked["y_true"],
            "y_prob": locked["y_prob"],
            "test_logits": locked["test_logits"],
            "split_membership_digest": locked["split_membership_digest"],
            "result_digest": "e" * 64,
        },
    }
    with pytest.raises(RuntimeError, match="COMMITTED_OUTER_EVALUATION_INVALID"):
        reuse_committed_outer_result(
            checkpoint,
            locked,
            expected_fold=1,
            expected_contract_digest="c" * 64,
        )


def test_locked_outer_result_rejects_empty_or_nonfinite_content():
    from src.training.method_x_resume import (
        build_locked_outer_result,
        validate_locked_outer_result,
    )

    common = {"fold": 0, "run_contract_digest": "a" * 64, "split_membership_digest": "b" * 64}
    empty = build_locked_outer_result(**common, y_true=[], y_prob=[], test_logits=[])
    with pytest.raises(RuntimeError, match="COMMITTED_OUTER_EVALUATION_INVALID"):
        validate_locked_outer_result(
            empty, expected_fold=0, expected_contract_digest="a" * 64
        )
    invalid = build_locked_outer_result(
        **common, y_true=[0, 1], y_prob=[0.2, float("nan")], test_logits=[-1.0, 1.0]
    )
    with pytest.raises(RuntimeError, match="COMMITTED_OUTER_EVALUATION_INVALID"):
        validate_locked_outer_result(
            invalid, expected_fold=0, expected_contract_digest="a" * 64
        )

def test_exact_resume_loader_policy_rejects_ambiguous_monai_workers():
    from src.training.method_x_resume import validate_exact_resume_loader_contract

    validate_exact_resume_loader_contract(
        required=True, monai_augment=True, num_workers=0, persistent_workers=False
    )
    with pytest.raises(RuntimeError, match="EXACT_RESUME_MONAI_REQUIRES_NUM_WORKERS_ZERO"):
        validate_exact_resume_loader_contract(
            required=True, monai_augment=True, num_workers=2, persistent_workers=False
        )
    with pytest.raises(RuntimeError, match="EXACT_RESUME_PERSISTENT_WORKERS_UNSUPPORTED"):
        validate_exact_resume_loader_contract(
            required=True, monai_augment=False, num_workers=2, persistent_workers=True
        )


def test_train_cli_integrates_exact_resume_contract_and_deterministic_loader():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")

    assert '"--cache_sha256"' in source
    assert '"--require_exact_resume_contract"' in source
    assert "generator=train_generator" in source
    assert "worker_init_fn=seed_worker" in source
    assert '"rng_state": capture_rng_state()' in source
    assert '"loader_generator_state": train_generator.get_state()' in source
    assert '"run_contract_digest": run_contract["digest"]' in source
    assert '"split_membership_digest": fold_membership_digest' in source


def test_train_restores_rng_and_loader_after_optimizer_state_before_next_epoch():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    optimizer_load = source.index('optimizer.load_state_dict(progress["optimizer_state_dict"])')
    rng_restore = source.index('restore_rng_state(progress["rng_state"])', optimizer_load)
    loader_restore = source.index(
        'train_generator.set_state(progress["loader_generator_state"])', rng_restore
    )
    epoch_loop = source.index("for epoch in range(start_epoch, epochs_per_fold)", loader_restore)

    assert optimizer_load < rng_restore < loader_restore < epoch_loop


def test_train_exact_resume_uses_fold_state_machine_around_outer_loader():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    locked = source.index('target="CHECKPOINT_LOCKED"')
    in_progress = source.index('target="OUTER_EVAL_IN_PROGRESS"', locked)
    outer_loader = source.index("test_loader = DataLoader(", in_progress)
    done = source.index('target="OUTER_EVAL_DONE"', outer_loader)

    assert locked < in_progress < outer_loader < done
    # FINALIZED is owned outside the trainer (reconciler/finalizer), so it
    # must not appear as a trainer transition. The trainer-owned segment
    # above plus the legal machine below is the full ordering guarantee.
    from src.training.method_x_resume import FOLD_STAGES, _ALLOWED_TRANSITIONS

    assert "FINALIZED" in FOLD_STAGES
    assert "FINALIZED" in _ALLOWED_TRANSITIONS["OUTER_EVAL_DONE"]


def test_train_finalized_exact_resume_reuses_committed_outer_result():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    completed_branch = source.index("if completed_epochs >= epochs_per_fold")
    reuse = source.index("reuse_committed_outer_result(", completed_branch)
    legacy_re_evaluation = source.index("evaluate_saved_fold(", reuse)

    assert reuse < legacy_re_evaluation


def test_terminal_early_stopped_progress_cannot_run_an_extra_epoch():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    terminal_restore = source.index(
        'fold_terminal = bool(progress.get("fold_terminal", False)) if progress is not None else False'
    )
    epoch_loop = source.index("for epoch in range(start_epoch, epochs_per_fold):", terminal_restore)
    terminal_break = source.index("if fold_terminal:\n                    break", epoch_loop)
    run_epoch = source.index("train_loss = run_epoch(", epoch_loop)
    early_reason = source.index('"EARLY_STOPPING"', run_epoch)

    assert terminal_restore < epoch_loop < terminal_break < run_epoch < early_reason


def test_tiny_interruption_resume_matches_uninterrupted_weights_and_curve():
    from torch.utils.data import DataLoader, TensorDataset

    from src.utils.reproducibility import (
        capture_rng_state,
        restore_rng_state,
        seeded_torch_generator,
        seed_everything,
    )

    dataset = TensorDataset(torch.arange(32, dtype=torch.float32).reshape(16, 2) / 10.0)

    def run(checkpoint=None, *, stop_after=None):
        seed_everything(41)
        generator = seeded_torch_generator(41)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Dropout(0.25), torch.nn.Linear(4, 1)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        curve = []
        start_epoch = 0
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            curve = list(checkpoint["curve"])
            start_epoch = int(checkpoint["epoch"])
            restore_rng_state(checkpoint["rng_state"])
            generator.set_state(checkpoint["loader_generator_state"])
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)
        saved = None
        for epoch in range(start_epoch, 3):
            losses = []
            model.train()
            for (x,) in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model(x).square().mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            curve.append(sum(losses) / len(losses))
            saved = {
                "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "curve": list(curve),
                "epoch": epoch + 1,
                "rng_state": capture_rng_state(),
                "loader_generator_state": generator.get_state(),
            }
            if stop_after == epoch + 1:
                break
        return model.state_dict(), curve, saved

    expected_state, expected_curve, _ = run()
    _, _, interrupted = run(stop_after=2)
    actual_state, actual_curve, _ = run(interrupted)

    assert actual_curve == expected_curve
    assert all(torch.equal(actual_state[key], value) for key, value in expected_state.items())


def test_resume_payload_keeps_cpu_rng_and_loader_states_on_cpu(tmp_path):
    from scripts.train_3d_mri import load_resume_payload

    checkpoint = tmp_path / "progress.pt"
    torch.save(
        {
            "rng_state": {"torch_cpu": torch.get_rng_state()},
            "loader_generator_state": torch.Generator().get_state(),
        },
        checkpoint,
    )

    payload = load_resume_payload(checkpoint, torch.device("cuda"))

    assert payload["rng_state"]["torch_cpu"].device.type == "cpu"
    assert payload["loader_generator_state"].device.type == "cpu"
