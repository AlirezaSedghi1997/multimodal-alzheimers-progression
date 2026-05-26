import json
from pathlib import Path


def test_clinical_ad_feasibility_stage_is_registered_as_prefinal_public_stage():
    from scripts.run_project_stage import DATA_ROOT_STAGES, STAGE_ARTIFACTS, STAGE_COMMANDS, STAGE_ORDER

    stage = "clinical-ad-feasibility"
    assert stage in STAGE_ORDER
    assert stage in STAGE_COMMANDS
    assert stage in STAGE_ARTIFACTS
    assert stage in DATA_ROOT_STAGES
    assert STAGE_ORDER.index("tau") < STAGE_ORDER.index(stage)
    assert STAGE_ORDER.index(stage) < STAGE_ORDER.index("final-analysis")
    assert STAGE_ORDER.index(stage) < STAGE_ORDER.index("public-bundle")
    assert STAGE_COMMANDS[stage] == ["scripts/clinical_ad_target_feasibility.py"]
    assert STAGE_ARTIFACTS[stage] == Path("results/tables/clinical_ad_target_feasibility.json")


def test_clinical_ad_feasibility_artifact_current_requires_blocked_clinical_claim(tmp_path, monkeypatch):
    from scripts import run_project_stage

    artifact = tmp_path / "clinical_ad_target_feasibility.json"
    artifact.write_text(
        json.dumps(
            {
                "status": "FEASIBILITY_ONLY_NOT_TRAINED",
                "training_started": False,
                "clinical_alzheimer_output_allowed_now": False,
                "subject_level_outputs_exported": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(run_project_stage.STAGE_ARTIFACTS, "clinical-ad-feasibility", artifact)
    assert run_project_stage.artifact_is_current("clinical-ad-feasibility") is True

    artifact.write_text(
        json.dumps(
            {
                "status": "FEASIBILITY_ONLY_NOT_TRAINED",
                "training_started": False,
                "clinical_alzheimer_output_allowed_now": True,
                "subject_level_outputs_exported": False,
            }
        ),
        encoding="utf-8",
    )
    assert run_project_stage.artifact_is_current("clinical-ad-feasibility") is False
