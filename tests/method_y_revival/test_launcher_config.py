from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def test_revival_config_separates_tracers_targets_and_claims():
    config = yaml.safe_load(Path("configs/method_y_revival_v1.yaml").read_text(encoding="utf-8"))

    assert config["paths"]["project_root"] == "H:/New project"
    assert config["paths"]["raw_root"] == "H:/A4 Learn"
    assert config["branches"]["amyloid"]["tracer"] == "FBP"
    assert config["branches"]["tau"]["tracer"] == "FTP"
    assert config["branches"]["amyloid"]["forbidden_primary_targets"] == ["amyloid_pos", "centiloid_from_same_pet"]
    assert config["validation"]["outer_folds"] == 5
    assert config["validation"]["inner_folds"] == 3
    assert config["external_validation"]["allow_refit"] is False
    assert config["scanner_sensitivity"]["minimum_events_per_manufacturer"] == 10


def test_powershell_launcher_is_preflight_only_and_has_no_runtime_limit():
    text = Path("scripts/run_method_y_revival.ps1").read_text(encoding="utf-8")

    assert "run_method_y_revival.py" in text
    assert "--preflight-only" in text
    assert "--train-tau-regional" not in text
    assert "--train-amyloid-regional" not in text
    assert "--train-paired-at-regional" not in text
    assert "--build-tau-endpoint" not in text
    assert "--build-amyloid-endpoint" not in text
    assert "--build-paired-at-endpoint" not in text
    assert "--build-scanner-audit" not in text
    assert "--run-scanner-sensitivity" not in text
    assert "max_runtime" not in text.casefold()
    assert "no_resume" not in text.casefold()
    assert "--output-root" not in text or "method_y_revival_v1" in text


def test_legacy_train_flags_fail_closed_without_v2_task4_committed(tmp_path):
    from scripts.run_method_y_revival import main

    with pytest.raises(SystemExit, match="LEGACY_V1_TRAIN_REQUIRES_V2_TASK4_COMMITTED"):
        main(
            [
                "--config",
                "configs/method_y_revival_v1.yaml",
                "--output-root",
                str(tmp_path / "method_y_revival_v1"),
                "--train-amyloid-regional",
            ]
        )
    assert not (tmp_path / "method_y_revival_v1" / "provenance" / "preflight_report.json").exists()


def test_preflight_cli_writes_truthful_pending_and_blocked_artifacts(tmp_path):
    from scripts.run_method_y_revival import main

    output = tmp_path / "method_y_revival_v1"
    exit_code = main(
        [
            "--config",
            "configs/method_y_revival_v1.yaml",
            "--output-root",
            str(output),
            "--preflight-only",
        ]
    )

    assert exit_code == 0
    report = json.loads((output / "provenance" / "preflight_report.json").read_text(encoding="utf-8"))
    assert report["regional_path_status"] == "READY_FOR_MANIFEST_CONSTRUCTION"
    assert report["branch_source_readiness"]["regional_sources_ready"] is True
    assert report["branch_source_readiness"]["external_sources_ready"] is False
    assert report["image_path_status"] == "BLOCKED_REFERENCE_TEMPLATE_NOT_LOCKED"
    assert report["external_validation_status"] == "EXTERNAL_PENDING"
    assert report["visual_qc_status"] == "AGENT_MULTI_PASS_QC_PENDING"
    assert "HUMAN" not in json.dumps(report).upper()
    assert report["raw_root_mutated"] is False

    second_exit_code = main(
        [
            "--config",
            "configs/method_y_revival_v1.yaml",
            "--output-root",
            str(output),
            "--preflight-only",
        ]
    )
    assert second_exit_code == 0
