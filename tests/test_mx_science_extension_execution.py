"""Fail-closed and retry fixtures for the one-command scientific add-on."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.mx_fasttrack_v2.run import sha
from scripts.mx_science_extension_v1 import execution, outer, supervise_extension


def _write(path: Path, value: dict | str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value), encoding="utf-8")
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_bytes(value)


def _fake_arm(tmp_path: Path, *, name: str = "hybrid_baseline_seed29") -> tuple[list[str], Path]:
    directory = tmp_path / execution.OUTPUT / "inner_tasks" / name
    command = ["python", "trainer.py", "--out_dir", str(directory),
               "--seed", "29", "--hybrid_variant", "baseline",
               "--input_mode", "mri_plus_tabular", "--phase", "final"]
    contract = {"digest": "d" * 64, "seed": 29, "folds": 5, "epochs": 100,
                "parameters": {"hybrid_variant": "baseline",
                               "input_mode": "mri_plus_tabular", "phase": "final",
                               "sanity_random_labels": False,
                               "hpo_selection_only": True}}
    _write(directory / "run_contract.json", {"run_contract_digest": "d" * 64,
                                              "run_contract": contract})
    _write(directory / "three_d_metrics.json", {
        "aggregate": {"status": "COMPLETED", "training_status": "COMPLETED",
                      "hpo_selection_only": True, "outer_test_access_count": 0,
                      "completed_folds": 5, "epochs_per_fold": 20},
        "folds": [{"fold": i, "outer_test_evaluation_count": 0,
                   "hpo_selection_only": True} for i in range(5)]})
    for fold in range(5):
        cp = directory / f"method_x_3d_fold_{fold}.pt"
        _write(cp, f"checkpoint-{fold}".encode())
        _write(directory / f"method_x_3d_fold_{fold}_state.json", {
            "stage": "FINALIZED", "contract_digest": "d" * 64,
            "details": {"checkpoint_sha256": sha(cp), "outer_test_access_count": 0}})
    return command, directory


def test_completed_inner_arm_receipt_requires_five_hashed_folds(tmp_path: Path) -> None:
    command, directory = _fake_arm(tmp_path)
    row = execution.arm_evidence(tmp_path, "hybrid_baseline_seed29", command)
    assert row["status"] == "COMPLETED_INNER_ONLY"
    assert len(row["checkpoint_sha256"]) == 5
    (directory / "method_x_3d_fold_2.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="FOLD_STATE_INVALID"):
        execution.arm_evidence(tmp_path, "hybrid_baseline_seed29", command)


def test_inner_arm_rejects_outer_access_and_mode_drift(tmp_path: Path) -> None:
    command, directory = _fake_arm(tmp_path)
    metrics_path = directory / "three_d_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["aggregate"]["outer_test_access_count"] = 1
    _write(metrics_path, metrics)
    with pytest.raises(RuntimeError, match="CONTRACT_OR_METRICS_INVALID"):
        execution.arm_evidence(tmp_path, "hybrid_baseline_seed29", command)


def test_private_outer_context_write_once_and_order_bound(tmp_path: Path) -> None:
    split = tmp_path / "splits/fold_0_test.csv"
    _write(split, "patient_hash,sample_hash,ab_label,abs_pet_t1_gap_days\n"
                  "pt_a,s_a,1,90\npt_b,s_b,0,91\n")
    _write(tmp_path / outer.OUTPUT / "reports/day1_patient_provenance_audit.json",
           {"folds": [{"outer_qc_n": 2}]})
    expected = outer._fold_context(tmp_path, 0, {"s_a": 0, "s_b": 1}, {0, 1}, "l" * 64)
    assert [row["patient_hash"] for row in expected["rows"]] == ["pt_a", "pt_b"]
    assert outer._fold_context(tmp_path, 0, {"s_a": 0, "s_b": 1}, {0, 1}, "l" * 64) == expected
    with pytest.raises(RuntimeError, match="IMMUTABLE_FILE_CHANGED"):
        outer._fold_context(tmp_path, 0, {"s_a": 0, "s_b": 1}, {0, 1}, "m" * 64)


def test_supervisor_stage_retries_same_action_after_failure(tmp_path: Path,
                                                             monkeypatch) -> None:
    monkeypatch.setattr(supervise_extension, "ROOT", tmp_path)
    monkeypatch.setattr(supervise_extension, "OUTPUT", Path("out"))
    monkeypatch.setattr(supervise_extension, "foreign_method_x_writers", lambda root: [])
    supervisor = supervise_extension.Supervisor(tmp_path)
    with pytest.raises(RuntimeError, match="interrupted"):
        supervisor.stage("six_inner_arms", lambda: (_ for _ in ()).throw(RuntimeError("interrupted")))
    assert json.loads(supervisor.journal_path.read_text())["steps"]["six_inner_arms"]["status"] == "PAUSED_OR_FAILED_RESUMABLE"
    supervisor = supervise_extension.Supervisor(tmp_path)
    row = supervisor.stage("six_inner_arms", lambda: {"status": "SIX_ARMS_COMPLETED_INNER_ONLY"})
    assert row["status"] == "SIX_ARMS_COMPLETED_INNER_ONLY"
    assert json.loads(supervisor.journal_path.read_text())["steps"]["six_inner_arms"]["attempts"] == 2


def test_parent_frozen_drift_never_launches_new_parent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(supervise_extension, "ROOT", tmp_path)
    monkeypatch.setattr(supervise_extension, "OUTPUT", Path("out"))
    monkeypatch.setattr(supervise_extension, "PARENT_OUTPUT", Path("parent"))
    _write(tmp_path / "parent/reports/fasttrack_thesis_traceability.json", {})
    monkeypatch.setattr(supervise_extension, "ensure_v242_amendment",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("drift")))
    monkeypatch.setattr(supervise_extension, "run_logged",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")))
    with pytest.raises(RuntimeError, match="FROZEN_BUT_NOT_VERIFIABLE"):
        supervise_extension.Supervisor(tmp_path).parent()


def test_completed_private_model_receipt_reuses_without_cuda(tmp_path: Path) -> None:
    context_path = tmp_path / outer.OUTPUT / "_private_resume/outer_v1/fold_0/context.json"
    _write(context_path, {"rows": [{"patient_hash": "p", "sample_hash": "s",
                                   "label": 1, "gap_days": 10.0}]})
    cp = tmp_path / "checkpoint.pt"
    _write(cp, b"fixed checkpoint")
    name = "hybrid_baseline_seed29"
    receipt_path = context_path.parent / f"{name}.json"
    envelope = {"schema_version": 1, "fold": 0, "seed": 29,
                "variant": "baseline", "input_mode": "mri_plus_tabular",
                "outer_lock_sha256": "l" * 64,
                "fold_context_sha256": sha(context_path),
                "checkpoint_sha256": sha(cp),
                "contract_digest_as_stored": "d" * 64,
                "arm": name, "probabilities": [0.25]}
    _write(receipt_path, envelope)
    context = {"rows": [{"patient_hash": "p", "sample_hash": "s",
                         "label": 1, "gap_days": 10.0}]}
    assert outer._model_probabilities(
        tmp_path, fold=0, seed=29, variant="baseline", mode="mri_plus_tabular",
        checkpoint=cp, checkpoint_sha=sha(cp), contract_digest="d" * 64,
        context=context, sample_to_index={"s": 0}, lock_sha="l" * 64,
        receipt_name=name) == [0.25]
    envelope["probabilities"] = [float("nan")]
    _write(receipt_path, envelope)
    with pytest.raises(RuntimeError, match="PROBABILITIES_INVALID"):
        outer._model_probabilities(
            tmp_path, fold=0, seed=29, variant="baseline", mode="mri_plus_tabular",
            checkpoint=cp, checkpoint_sha=sha(cp), contract_digest="d" * 64,
            context=context, sample_to_index={"s": 0}, lock_sha="l" * 64,
            receipt_name=name)


def test_random_label_adapter_preserves_fit_prevalence_only(monkeypatch) -> None:
    from scripts import train_3d_mri as trainer
    from scripts.mx_science_extension_v1 import random_label_train
    original = trainer.H5MethodXDataset.__init__
    observed = {}

    def fake_trainer_main():
        fit = trainer.H5MethodXDataset(Path("unused.h5"), list(range(20)),
                                       [0] * 10 + [1] * 10,
                                       sanity_random_labels=True, seed=29)
        validation = trainer.H5MethodXDataset(Path("unused.h5"), [20, 21],
                                              [0, 1], sanity_random_labels=False, seed=29)
        observed["fit"] = fit.labels
        observed["validation"] = validation.labels
        return 0

    monkeypatch.setattr(trainer, "main", fake_trainer_main)
    monkeypatch.setattr("sys.argv", ["random_label_train.py", "--sanity_random_labels",
                                    "--hpo_selection_only", "--require_exact_resume_contract",
                                    "--require_gpu", "--phase", "dev", "--input_mode",
                                    "mri_plus_tabular", "--hybrid_variant", "baseline"])
    assert random_label_train.main() == 0
    assert sorted(observed["fit"]) == [0] * 10 + [1] * 10
    assert observed["fit"] != [0] * 10 + [1] * 10
    assert observed["validation"] == [0, 1]
    assert trainer.H5MethodXDataset.__init__ is original


def test_outer_panel_assembles_fold_local_choices_without_seed_replication(
        tmp_path: Path, monkeypatch) -> None:
    from scripts import train_3d_mri as trainer

    lock_path = tmp_path / outer.OUTPUT / "reports" / outer.OUTER_LOCK
    _write(lock_path, {"status": "LOCKED_BEFORE_EXTENSION_OUTER_ACCESS"})
    choices = [{"seed": seed, "fold": fold,
                "selected_variant": "plus_dense_skip" if fold % 2 == 0 else "baseline"}
               for seed in (13, 29, 47) for fold in range(5)]
    _write(tmp_path / outer.OUTPUT / "reports" / outer.SELECTION_NAME,
           {"choices": choices})
    controls = {}
    for kind in ("mri_only", "random_label"):
        name = outer._control_name(kind)
        directory = tmp_path / outer.OUTPUT / "inner_tasks" / name
        _write(directory / "run_contract.json", {"run_contract_digest": "d" * 64})
        hashes = []
        for fold in range(5):
            cp = directory / f"method_x_3d_fold_{fold}.pt"
            _write(cp, b"fixed" + bytes([fold]))
            hashes.append(sha(cp))
        controls[name] = {"checkpoint_sha256": hashes}
    monkeypatch.setattr(outer, "ensure_outer_lock",
                        lambda root, create=False: {"control_arms": controls})
    monkeypatch.setattr(outer, "baseline_binding", lambda *a: {})
    monkeypatch.setattr(trainer, "cache_index", lambda cache: {})
    monkeypatch.setattr(trainer, "ok_indices", lambda cache: set())
    monkeypatch.setattr(outer, "read_csv", lambda path: [{"sample_hash": "real",
                                                          "is_synthetic": "0"}])
    def fake_context(root, fold, sample_to_index, ok, lock_sha):
        n = 268 if fold == 4 else 267
        return {"rows": [{"patient_hash": f"patient-{fold}-{i}",
                          "sample_hash": f"sample-{fold}-{i}",
                          "label": i % 2, "gap_days": float(i)} for i in range(n)]}
    monkeypatch.setattr(outer, "_fold_context", fake_context)
    monkeypatch.setattr(outer, "_checkpoint_metadata",
                        lambda *a: (tmp_path / "unused.pt", "c" * 64, "d" * 64))
    def fake_model(root, *, context, variant, receipt_name, **kwargs):
        value = 0.9 if variant == "plus_dense_skip" else 0.1
        if receipt_name.startswith("mri_only_"):
            value = 0.4
        if receipt_name.startswith("random_label_"):
            value = 0.5
        return [value] * len(context["rows"])
    monkeypatch.setattr(outer, "_model_probabilities", fake_model)
    monkeypatch.setattr(outer, "_control_probabilities",
                        lambda root, fold, seed, context, lock_sha, real_samples: {
                            "tabular_outer_probabilities": [0.55] * len(context["rows"]),
                            "prevalence_outer_probabilities": [0.45] * len(context["rows"]),
                        })
    assert outer.run_outer(tmp_path)["patient_count"] == 1336
    panel = json.loads((tmp_path / outer.OUTPUT / "_private_resume/outer_v1"
                        / outer.PANEL).read_text())["rows"]
    assert len({row["patient_hash"] for row in panel}) == 1336
    assert panel[0]["predictions"]["hybrid_selected"] == {
        "13": 0.9, "29": 0.9, "47": 0.9}
    assert panel[267]["predictions"]["hybrid_selected"] == {
        "13": 0.1, "29": 0.1, "47": 0.1}
    assert len(panel) == 1336  # three seeds never triple the patient count


def test_execution_lock_blocks_late_creation_and_source_drift(tmp_path: Path,
                                                            monkeypatch) -> None:
    monkeypatch.setattr(execution, "OUTPUT", Path("out"))
    monkeypatch.setattr(execution, "execution_payload",
                        lambda root: {"status": "LOCKED_BEFORE_EXTENSION_GPU_AND_OUTER",
                                      "source_sha256": {"a.py": "a" * 64}})
    inner = tmp_path / "out/inner_tasks/partial"
    inner.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="LOCK_TOO_LATE"):
        execution.ensure_execution_lock(tmp_path, create=True)
    inner.rmdir()
    execution.ensure_execution_lock(tmp_path, create=True)
    monkeypatch.setattr(execution, "execution_payload",
                        lambda root: {"status": "LOCKED_BEFORE_EXTENSION_GPU_AND_OUTER",
                                      "source_sha256": {"a.py": "b" * 64}})
    with pytest.raises(RuntimeError, match="SOURCE_OR_PARENT_DRIFT"):
        execution.ensure_execution_lock(tmp_path, create=False)


def test_outer_lock_blocks_if_private_access_preceded_it(tmp_path: Path,
                                                        monkeypatch) -> None:
    monkeypatch.setattr(outer, "OUTPUT", Path("out"))
    monkeypatch.setattr(outer, "outer_lock_payload",
                        lambda root: {"status": "LOCKED_BEFORE_EXTENSION_OUTER_ACCESS"})
    _write(tmp_path / "out/_private_resume/outer_v1/fold_0/context.json", {})
    with pytest.raises(RuntimeError, match="LOCK_CREATED_AFTER_ACCESS"):
        outer.ensure_outer_lock(tmp_path, create=True)


def test_committed_inner_arm_reused_but_tampering_blocks(tmp_path, monkeypatch):
    import subprocess
    name = "hybrid_baseline_seed29"
    command, directory = _fake_arm(tmp_path, name=name)
    monkeypatch.setattr(execution, "make_arm_commands", lambda root: {name: command})
    monkeypatch.setattr(execution, "ensure_execution_lock", lambda *a, **kw: {
        "arm_command_sha256": {name: execution.command_digest(command)}})
    calls = []
    def run(*a, **kw):
        calls.append(name)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(execution, "run_logged", run)
    execution.run_arms(tmp_path)
    execution.run_arms(tmp_path)
    assert calls == [name]
    (directory / "method_x_3d_fold_0.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="FOLD_STATE_INVALID"):
        execution.run_arms(tmp_path)
    assert calls == [name]


def test_invalid_inference_does_not_poison_immutable_resume_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from scripts import train_3d_mri as trainer
    from src.evaluation import method_x_xai_forward as forward
    private = tmp_path / outer.OUTPUT / "_private_resume/outer_v1/fold_0"
    context = {"rows": [{"patient_hash": "p", "sample_hash": "s", "label": 1}]}
    _write(private / "context.json", context)
    cp = tmp_path / "method_x_3d_fold_0.pt"
    _write(cp, b"fixed")
    _write(cp.with_name("method_x_3d_fold_0_state.json"), {
        "details": {"split_membership_digest": "split"}})
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {
        "fold": 0, "backbone": "amuse_x_hybrid3d", "hybrid_variant": "baseline",
        "input_mode": "mri_plus_tabular", "drop_tabular_feature": "none",
        "run_contract_digest": "contract", "split_membership_digest": "split",
        "hpo_selection_only": True, "outer_test_access_count": 0,
        "outer_test_evaluation_count": 0, "tab_mean": None, "tab_std": None})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch, "device", lambda value: SimpleNamespace(type="cuda"))
    monkeypatch.setattr(trainer, "H5MethodXDataset", lambda *a, **kw: object())
    monkeypatch.setattr(torch.utils.data, "DataLoader", lambda *a, **kw: object())
    monkeypatch.setattr(forward, "build_model_from_checkpoint", lambda *a, **kw:
                        (object(), "model_state_dict"))
    probability = [float("nan")]
    monkeypatch.setattr(trainer, "predict", lambda *a, **kw: ([1], probability))
    kwargs = dict(fold=0, seed=29, variant="baseline", mode="mri_plus_tabular",
                  checkpoint=cp, checkpoint_sha=sha(cp), contract_digest="contract",
                  context=context, sample_to_index={"s": 0}, lock_sha="lock", receipt_name="model")
    with pytest.raises(RuntimeError, match="PROBABILITIES_INVALID"):
        outer._model_probabilities(tmp_path, **kwargs)
    assert not (private / "model.json").exists()
    probability[:] = [0.7]
    assert outer._model_probabilities(tmp_path, **kwargs) == [0.7]
    assert (private / "model.json").is_file()
