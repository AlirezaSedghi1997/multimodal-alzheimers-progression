from __future__ import annotations

import json
from pathlib import Path


def test_validity_orchestrator_writes_truthful_blockers_without_oof(tmp_path: Path, monkeypatch):
    import scripts.run_method_x_validity_orchestrator as orchestrator

    output = tmp_path / "battery"
    (output / "reports").mkdir(parents=True)
    (output / "battery_manifest.json").write_text(
        json.dumps(
            {
                "locked_protocol": {"resolution": 128, "folds": 5, "seed": 13},
                "models": [{"backbone": "resnet34_3d", "status": "SMOKE_PASS"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "evaluate_autonomous_supervision", lambda **kwargs: {"decision": "CONTINUE"})
    monkeypatch.setattr(orchestrator, "enforce_autonomous_supervision", lambda **kwargs: kwargs["report"])
    report = orchestrator.run_validity_orchestrator(root=tmp_path, output_root=output)
    assert report["status"] == "BLOCKED"
    assert "BASELINE_BATTERY_NOT_FINALIZED" in report["blockers"]
    assert "PAIRED_OOF_ARTIFACT_MISSING_OR_INVALID" in report["blockers"]
    assert report["training_started"] is False
    assert (output / "reports/sanity_report.json").is_file()
    assert (output / "reports/oof_comparison.json").is_file()


def _terminal_manifest() -> dict:
    common = {
        "fusion": "cross_attention",
        "task_loss": "bce",
        "learning_rate": 0.0001,
        "scheduler": "cosine",
        "dropout": 0.2,
        "batch_size": 1,
        "grad_accum": 8,
        "amp": True,
        "gradient_checkpointing": True,
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    # Mirrors the live five-model protocol: 7 structural rows, five effective
    # members (including reconciler-issued terminal training states) and two
    # append-only protocol exclusions.
    effective = [
        ("resnet10_3d", "COMPLETED_VALIDATED_REPAIRED"),
        ("resnet18_3d", "COMPLETED_VALIDATED_REPAIRED"),
        ("resnet34_3d", "COMPLETED_VALIDATED_RECONCILED"),
        ("densenet121_3d", "COMPLETED"),
        ("mednext_3d_s", "COMPLETED"),
    ]
    excluded = ["resnet50_3d", "efficientnet3d_b0"]
    names = [name for name, _ in effective] + excluded
    training_status = {name: status for name, status in effective}
    training_status.update({name: "PROTOCOL_EXCLUDED_NOT_STARTED" for name in excluded})
    row_status = {name: "COMPLETED" for name, _ in effective}
    row_status.update({name: "PROTOCOL_EXCLUDED_NOT_STARTED" for name in excluded})
    return {
        "locked_protocol": {"resolution": 128, "folds": 5, "epochs": 100, "seed": 13, "common": common},
        "locked_inputs": {"cache_sha256": "c" * 64},
        "spm_gate_snapshot": {"snapshot_sha256": "e" * 64},
        "models": [
            {"sequence": i, "backbone": name, "status": row_status[name], "training": {"status": training_status[name], "output_dir": f"models/{name}"}, "output_dir": f"models/{name}"}
            for i, name in enumerate(names)
        ],
    }


def test_baseline_is_terminal_accepts_five_model_protocol_with_exclusions():
    from src.evaluation.method_x_validity_tasks import baseline_is_terminal

    assert baseline_is_terminal(_terminal_manifest()) is True


def test_baseline_is_terminal_rejects_seven_completed_without_exclusions():
    from src.evaluation.method_x_validity_tasks import baseline_is_terminal

    manifest = _terminal_manifest()
    for row in manifest["models"]:
        row["status"] = "COMPLETED"
        row["training"]["status"] = "COMPLETED"
    # Excluded backbones retrained as COMPLETED violates the append-only
    # five-model amendment and must never read as terminal.
    assert baseline_is_terminal(manifest) is False


def test_baseline_is_terminal_rejects_non_terminal_training_state():
    from src.evaluation.method_x_validity_tasks import baseline_is_terminal

    manifest = _terminal_manifest()
    manifest["models"][2]["training"]["status"] = "PAUSED_RESUMABLE"
    assert baseline_is_terminal(manifest) is False


def test_mx03_task_contracts_are_prespecified_and_include_all_arms(tmp_path: Path):
    from src.evaluation.method_x_validity_tasks import (
        ALLOWED_FEATURES,
        EXECUTABLE_TASK_IDS,
        REFERENCE_BACKBONE,
        build_task_contracts,
    )

    contracts = build_task_contracts(
        root=tmp_path,
        output_root=tmp_path / "battery",
        manifest=_terminal_manifest(),
        python="python.exe",
    )
    assert len(contracts) == 4 + len(ALLOWED_FEATURES)
    assert [row["task_id"] for row in contracts] == list(EXECUTABLE_TASK_IDS)
    assert all(row["reference_backbone"] == REFERENCE_BACKBONE for row in contracts)
    assert all(row["outer_test_used_for_selection"] is False for row in contracts)
    tabular = next(row for row in contracts if row["task_id"] == "tabular_only")
    assert "--input_mode" in tabular["command"]
    assert "tabular_only" in tabular["command"]
    assert "--require_exact_resume_contract" in tabular["command"]
    assert "--spm_snapshot_sha256" in tabular["command"]


def test_private_oof_materializer_rejects_cross_model_membership_drift(tmp_path: Path):
    from src.evaluation.method_x_validity_tasks import materialize_private_oof

    output = tmp_path / "battery"
    manifest = _terminal_manifest()
    output.mkdir()
    (output / "battery_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for row_index, row in enumerate(manifest["models"]):
        model_dir = output / row["output_dir"]
        model_dir.mkdir(parents=True)
        (model_dir / "run_contract.json").write_text(
            json.dumps({"run_contract": {"spm_snapshot_sha256": "e" * 64}, "run_contract_digest": "d" * 64}),
            encoding="utf-8",
        )
        (model_dir / "three_d_metrics.json").write_text(
            json.dumps({"aggregate": {"status": "COMPLETED", "completed_folds": 5}}),
            encoding="utf-8",
        )
        for fold in range(5):
            (model_dir / f"method_x_3d_fold_{fold}_outer_eval.json").write_text(
                json.dumps(
                    {
                        "evaluation_count": 1,
                        "raw_identifiers_stored": False,
                        "y_true": [0, 1],
                        "y_prob": [0.2, 0.8],
                        "split_membership_digest": ("a" if row_index == 0 else "b") * 64,
                    }
                ),
                encoding="utf-8",
            )
    report = materialize_private_oof(root=tmp_path, output_root=output, manifest=manifest)
    assert report["status"] == "BLOCKED"
    assert any("PAIRED_MEMBERSHIP_MISMATCH" in value for value in report["blockers"])


def test_run_mode_executes_prespecified_tasks_under_autosupervision(tmp_path: Path, monkeypatch):
    import scripts.run_method_x_validity_orchestrator as orchestrator

    output = tmp_path / "battery"
    (output / "reports").mkdir(parents=True)
    (output / "battery_manifest.json").write_text(
        json.dumps(_terminal_manifest()), encoding="utf-8"
    )
    private = output / "private"
    private.mkdir()
    contracts = []
    for index, task_id in enumerate(("random_labels", "random_noise")):
        contracts.append(
            {
                "task_id": task_id,
                "kind": "sanity",
                "artifact": str(private / ("sanity_random_labels.json" if index == 0 else "sanity_random_noise.json")),
                "output_dir": str(output / "mx03_runs" / task_id),
                "contract_sha256": str(index + 1) * 64,
                "command": ["python.exe", task_id],
            }
        )
    monkeypatch.setattr(orchestrator, "build_task_contracts", lambda **kwargs: contracts)
    monkeypatch.setattr(orchestrator, "evaluate_autonomous_supervision", lambda **kwargs: {"decision": "CONTINUE"})
    monkeypatch.setattr(orchestrator, "enforce_autonomous_supervision", lambda **kwargs: kwargs["report"])

    def fake_oof(**kwargs):
        payload = {
            "status": "FINALIZED",
            "membership_digest": "a" * 64,
            "models": {"a": {}, "b": {}},
            "y_true": [0, 1],
            "blockers": [],
            "subject_level_vectors_exported": False,
        }
        (private / "oof_predictions.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_diagnostics(**kwargs):
        (private / "calibration_summary.json").write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
        (private / "cohort_probe_summary.json").write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
        return {"status": "COMPLETED", "blockers": []}

    executions = []

    def fake_process(command, **kwargs):
        executions.append(command)
        kwargs["on_start"](123 + len(executions))
        return {"returncode": 0, "decision": "COMPLETE", "duration_seconds": 0.1}

    monkeypatch.setattr(orchestrator, "materialize_private_oof", fake_oof)
    monkeypatch.setattr(orchestrator, "materialize_baseline_diagnostics", fake_diagnostics)
    monkeypatch.setattr(orchestrator, "run_supervised_process", fake_process)
    monkeypatch.setattr(
        orchestrator,
        "read_task_metrics",
        lambda **kwargs: {
            "status": "COMPLETED",
            "aggregate": {"status": "COMPLETED", "folds": 5, "roc_auc": 0.5},
            "metrics_sha256": "f" * 64,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "run_method_x_validity_suite",
        lambda **kwargs: {"status": "COMPLETED", "blockers": [], "pairwise_bootstrap": {}},
    )
    promotion_calls = []
    monkeypatch.setattr(
        orchestrator,
        "materialize_promotion_report",
        lambda **kwargs: promotion_calls.append(kwargs) or {
            "status": "COMPLETED",
            "reporting_backbone": "resnet34_3d",
            "promotion_occurred": False,
            "blockers": [],
        },
    )

    report = orchestrator.run_validity_orchestrator(
        root=tmp_path, output_root=output, mode="run", resamples=10
    )

    assert len(executions) == 2
    assert report["status"] == "COMPLETED"
    assert report["training_started"] is True
    assert report["autonomous_supervision"]["human_approval_required"] is False
    assert report["promotion_report"]["status"] == "COMPLETED"
    assert len(promotion_calls) == 1
    state = json.loads((private / "mx03_task_state.json").read_text(encoding="utf-8"))
    assert {row["status"] for row in state["tasks"].values()} == {"COMPLETED"}
