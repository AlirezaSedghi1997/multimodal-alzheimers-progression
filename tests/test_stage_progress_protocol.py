from pathlib import Path

from scripts.run_project_stage import STAGE_ORDER, artifact_is_current


def test_progress_file_has_privacy_contract():
    text = Path(".agent/PROGRESS.yaml").read_text(encoding="utf-8")
    assert "pet_as_input_forbidden: true" in text
    assert "raw_data_read_only: true" in text


def test_tau_runs_before_final_packaging():
    assert STAGE_ORDER.index("mri-volume-cache") < STAGE_ORDER.index("train-3d")
    assert STAGE_ORDER.index("train-3d") < STAGE_ORDER.index("final-analysis")
    assert STAGE_ORDER.index("tau") < STAGE_ORDER.index("final-analysis")
    assert STAGE_ORDER.index("final-analysis") < STAGE_ORDER.index("public-bundle")


def test_tau_skip_artifact_is_stale_when_tau_target_available(tmp_path, monkeypatch):
    artifact = tmp_path / "tau_regression_metrics.json"
    artifact.write_text('{"schema_version": 2, "mode": "SKIP_WITH_JUSTIFICATION"}', encoding="utf-8")
    data_root = tmp_path / "data_root"
    tau_path = data_root / "Clinical" / "External Data" / "imaging_SUVR_tau.csv"
    tau_path.parent.mkdir(parents=True)
    tau_path.write_text("header\n", encoding="utf-8")

    monkeypatch.setattr("scripts.run_project_stage.STAGE_ARTIFACTS", {"tau": artifact})
    assert artifact_is_current("tau", str(data_root)) is False


def test_tau_skip_artifact_is_stale_when_tau_target_is_discovered(tmp_path, monkeypatch):
    artifact = tmp_path / "tau_regression_metrics.json"
    artifact.write_text('{"schema_version": 2, "mode": "SKIP_WITH_JUSTIFICATION"}', encoding="utf-8")
    data_root = tmp_path / "data_root"
    tau_path = data_root / "unexpected" / "nested" / "imaging_SUVR_tau.csv"
    tau_path.parent.mkdir(parents=True)
    tau_path.write_text("header\n", encoding="utf-8")

    monkeypatch.setattr("scripts.run_project_stage.STAGE_ARTIFACTS", {"tau": artifact})
    assert artifact_is_current("tau", str(data_root)) is False
