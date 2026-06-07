from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_makefile_patch_contains_method_y_stages():
    text = (ROOT / "patches" / "MAKEFILE_METHOD_Y_PHASE2_TARGETS.mk").read_text(encoding="utf-8")
    for target in ["method-y-gap-filter", "method-y-pet-manifest", "method-y-pet-dry", "method-y-readiness"]:
        assert target in text


def test_agent_playbook_exists_and_mentions_dry_run():
    text = (ROOT / "docs" / "agent_method_y_pet_phase2_playbook.md").read_text(encoding="utf-8")
    assert "--dry_run" in text
    assert "Read compact summaries first" in text
