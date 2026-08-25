from __future__ import annotations

import numpy as np
import pytest


def _dataset():
    from src.method_y_revival.image_training_runtime_v3 import MemorySurvivalDataset

    rng = np.random.default_rng(17)
    tabular = rng.normal(size=(24, 4)).astype(np.float32)
    event = np.asarray([0, 1] * 12, dtype=np.float32)
    time = np.linspace(200.0, 1500.0, 24, dtype=np.float32)
    return MemorySurvivalDataset(tabular=tabular, event=event, time_days=time)


def _model():
    from src.method_y_revival.image_survival_v3 import TabularSurvivalNet

    return TabularSurvivalNet(
        tabular_dim=4,
        hazard_bins=5,
        tabular_hidden_dim=8,
        dropout=0.0,
    )


def test_interrupted_training_resumes_at_next_epoch_with_identical_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from src.method_y_revival.image_training_runtime_v3 import (
        PlannedTrainingInterruption,
        fit_survival_model,
        seed_everything,
    )

    edges = [0.0, 365.0, 730.0, 1095.0, 1460.0, 1640.0]
    data = _dataset()
    common = dict(
        train_dataset=data,
        validation_dataset=data,
        bin_edges=edges,
        run_contract_digest="a" * 64,
        seed=123,
        max_epochs=3,
        patience=10,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=6,
        gradient_accumulation_steps=1,
        device="cpu",
        amp=False,
        fixed_epochs=3,
    )

    seed_everything(123)
    uninterrupted = fit_survival_model(
        model=_model(), output_dir=tmp_path / "uninterrupted", **common
    )

    seed_everything(123)
    with pytest.raises(PlannedTrainingInterruption):
        fit_survival_model(
            model=_model(),
            output_dir=tmp_path / "resumed",
            interrupt_after_epoch=1,
            **common,
        )
    seed_everything(123)
    resumed = fit_survival_model(model=_model(), output_dir=tmp_path / "resumed", **common)

    assert resumed["resumed_from_epoch"] == 1
    assert resumed["start_epoch"] == 2
    assert uninterrupted["history"] == resumed["history"]
    first = torch.load(uninterrupted["locked_checkpoint_path"], map_location="cpu", weights_only=True)
    second = torch.load(resumed["locked_checkpoint_path"], map_location="cpu", weights_only=True)
    assert first["model_state"].keys() == second["model_state"].keys()
    for key in first["model_state"]:
        assert torch.equal(first["model_state"][key], second["model_state"][key])


def test_resume_contract_mismatch_fails_closed(tmp_path):
    from src.method_y_revival.image_training_runtime_v3 import (
        TrainingRuntimeV3Error,
        fit_survival_model,
        seed_everything,
    )

    data = _dataset()
    seed_everything(1)
    fit_survival_model(
        model=_model(),
        train_dataset=data,
        validation_dataset=data,
        bin_edges=[0, 365, 730, 1095, 1460, 1640],
        output_dir=tmp_path / "run",
        run_contract_digest="a" * 64,
        seed=1,
        max_epochs=1,
        patience=2,
        learning_rate=1e-3,
        weight_decay=0,
        batch_size=6,
        gradient_accumulation_steps=1,
        device="cpu",
        amp=False,
        fixed_epochs=1,
    )
    with pytest.raises(TrainingRuntimeV3Error, match="TRAINING_CHECKPOINT_CONTRACT_MISMATCH"):
        fit_survival_model(
            model=_model(),
            train_dataset=data,
            validation_dataset=data,
            bin_edges=[0, 365, 730, 1095, 1460, 1640],
            output_dir=tmp_path / "run",
            run_contract_digest="b" * 64,
            seed=1,
            max_epochs=1,
            patience=2,
            learning_rate=1e-3,
            weight_decay=0,
            batch_size=6,
            gradient_accumulation_steps=1,
            device="cpu",
            amp=False,
            fixed_epochs=1,
        )


def test_resume_recovers_terminal_epoch_before_locked_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from src.method_y_revival.image_training_runtime_v3 import fit_survival_model, seed_everything

    data = _dataset()
    edges = [0, 365, 730, 1095, 1460, 1640]
    common = dict(
        train_dataset=data,
        validation_dataset=data,
        bin_edges=edges,
        run_contract_digest="c" * 64,
        seed=7,
        max_epochs=2,
        patience=10,
        learning_rate=1e-3,
        weight_decay=0,
        batch_size=6,
        gradient_accumulation_steps=1,
        device="cpu",
        amp=False,
        fixed_epochs=2,
    )
    seed_everything(7)
    uninterrupted = fit_survival_model(model=_model(), output_dir=tmp_path / "full", **common)
    # Simulate crash between terminal checkpoint and locked artifact
    import pathlib

    locked = pathlib.Path(tmp_path / "full" / "locked_checkpoint.pt")
    summary = pathlib.Path(tmp_path / "full" / "training_summary.json")
    # Remove locked/summary to simulate crash after epoch_checkpoint but before finalization
    locked.unlink(missing_ok=True)
    summary.unlink(missing_ok=True)
    seed_everything(7)
    recovered = fit_survival_model(model=_model(), output_dir=tmp_path / "full", **common)
    assert recovered["status"] == "CHECKPOINT_LOCKED"
    assert recovered["epoch_completed"] == uninterrupted["epoch_completed"]
    first = torch.load(uninterrupted["locked_checkpoint_path"], map_location="cpu", weights_only=True)
    second = torch.load(recovered["locked_checkpoint_path"], map_location="cpu", weights_only=True)
    for k in first["model_state"]:
        assert torch.equal(first["model_state"][k], second["model_state"][k])


def test_resume_recovers_locked_checkpoint_before_summary(tmp_path):
    torch = pytest.importorskip("torch")
    from src.method_y_revival.image_training_runtime_v3 import fit_survival_model, seed_everything
    import pathlib

    data = _dataset()
    edges = [0, 365, 730, 1095, 1460, 1640]
    common = dict(
        train_dataset=data,
        validation_dataset=data,
        bin_edges=edges,
        run_contract_digest="d" * 64,
        seed=8,
        max_epochs=2,
        patience=10,
        learning_rate=1e-3,
        weight_decay=0,
        batch_size=6,
        gradient_accumulation_steps=1,
        device="cpu",
        amp=False,
        fixed_epochs=2,
    )
    seed_everything(8)
    uninterrupted = fit_survival_model(model=_model(), output_dir=tmp_path / "full2", **common)
    summary = pathlib.Path(tmp_path / "full2" / "training_summary.json")
    summary.unlink(missing_ok=True)
    seed_everything(8)
    recovered = fit_survival_model(model=_model(), output_dir=tmp_path / "full2", **common)
    assert recovered["status"] == "CHECKPOINT_LOCKED"


def test_recovered_locked_weights_equal_uninterrupted_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from src.method_y_revival.image_training_runtime_v3 import fit_survival_model, seed_everything

    data = _dataset()
    edges = [0, 365, 730, 1095, 1460, 1640]
    common = dict(
        train_dataset=data,
        validation_dataset=data,
        bin_edges=edges,
        run_contract_digest="e" * 64,
        seed=9,
        max_epochs=2,
        patience=10,
        learning_rate=1e-3,
        weight_decay=0,
        batch_size=6,
        gradient_accumulation_steps=1,
        device="cpu",
        amp=False,
        fixed_epochs=2,
    )
    seed_everything(9)
    uninterrupted = fit_survival_model(model=_model(), output_dir=tmp_path / "a", **common)
    # Crash after terminal but before locked
    import pathlib

    pathlib.Path(tmp_path / "a" / "locked_checkpoint.pt").unlink(missing_ok=True)
    pathlib.Path(tmp_path / "a" / "training_summary.json").unlink(missing_ok=True)
    seed_everything(9)
    recovered = fit_survival_model(model=_model(), output_dir=tmp_path / "a", **common)
    assert recovered["history"] == uninterrupted["history"]


def test_terminal_fold_is_not_retrained_after_recovery(tmp_path):
    from src.method_y_revival.image_training_runtime_v3 import fit_survival_model, seed_everything

    data = _dataset()
    edges = [0, 365, 730, 1095, 1460, 1640]
    common = dict(
        train_dataset=data,
        validation_dataset=data,
        bin_edges=edges,
        run_contract_digest="f" * 64,
        seed=10,
        max_epochs=1,
        patience=10,
        learning_rate=1e-3,
        weight_decay=0,
        batch_size=6,
        gradient_accumulation_steps=1,
        device="cpu",
        amp=False,
        fixed_epochs=1,
    )
    seed_everything(10)
    first = fit_survival_model(model=_model(), output_dir=tmp_path / "t", **common)
    # Complete run already locked; second run must not retrain
    seed_everything(10)
    second = fit_survival_model(model=_model(), output_dir=tmp_path / "t", **common)
    assert second["resumed_from_epoch"] == 1
    assert second["start_epoch"] == 2
    assert second["epoch_completed"] == first["epoch_completed"]


def test_checkpoint_contract_mismatch_still_fails_closed(tmp_path):
    from src.method_y_revival.image_training_runtime_v3 import TrainingRuntimeV3Error, fit_survival_model, seed_everything

    data = _dataset()
    seed_everything(11)
    fit_survival_model(
        model=_model(),
        train_dataset=data,
        validation_dataset=data,
        bin_edges=[0, 365, 730, 1095, 1460, 1640],
        output_dir=tmp_path / "x",
        run_contract_digest="a" * 64,
        seed=11,
        max_epochs=1,
        patience=10,
        learning_rate=1e-3,
        weight_decay=0,
        batch_size=6,
        gradient_accumulation_steps=1,
        device="cpu",
        amp=False,
        fixed_epochs=1,
    )
    with pytest.raises(TrainingRuntimeV3Error, match="TRAINING_CHECKPOINT_CONTRACT_MISMATCH"):
        fit_survival_model(
            model=_model(),
            train_dataset=data,
            validation_dataset=data,
            bin_edges=[0, 365, 730, 1095, 1460, 1640],
            output_dir=tmp_path / "x",
            run_contract_digest="b" * 64,
            seed=11,
            max_epochs=1,
            patience=10,
            learning_rate=1e-3,
            weight_decay=0,
            batch_size=6,
            gradient_accumulation_steps=1,
            device="cpu",
            amp=False,
            fixed_epochs=1,
        )

