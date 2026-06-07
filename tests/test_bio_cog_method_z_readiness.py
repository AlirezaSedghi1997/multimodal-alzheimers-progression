from __future__ import annotations

import json
from pathlib import Path


def test_bio_cog_method_z_readiness_exists():
    path = Path("results/tables/bio_cog_method_z_readiness.json")
    assert path.exists(), "Run scripts/audit_bio_cog_method_z_readiness.py"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["branch"] == "bio_cog_method_z"
    assert obj["supports_method_x_strict_claim"] is False
    assert obj["pet_as_method_x_input"] is False
    assert obj["rows"] > 0


def test_thesis_report_mentions_phase2_mvp():
    path = Path("results/final_results/thesis/thesis_grade_report.md")
    assert path.exists(), "Run scripts/build_thesis_grade_report.py"
    text = path.read_text(encoding="utf-8")
    assert "Phase 2 Minimum MVP" in text
    assert "Method X-Cog/SF internal stress tests" in text
    assert "Bio-Cog / Method Z" in text
