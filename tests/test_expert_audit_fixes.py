from __future__ import annotations

import json
from pathlib import Path


def test_thesis_report_has_centiloid_and_coupling_values():
    path = Path("results/final_results/thesis/thesis_grade_report.md")
    assert path.exists(), "Run scripts/build_thesis_grade_report.py"
    text = path.read_text(encoding="utf-8")
    assert "before SF Cramér's V: `1.0000`" in text or "before SF Cramér's V: `1.0`" in text
    assert "after SF Cramér's V: `0.8996`" in text
    assert "pooled enrolled best Spearman" in text
    assert "0.443" in text or "0.441" in text
    assert "Incremental value vs validity" in text
    assert "DESIGN_READY_EXECUTION_GATED" in text or "execution blockers" in text.lower()


def test_incremental_value_has_validity_flags():
    path = Path("results/tables/phase1_incremental_value_table.json")
    assert path.exists()
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj.get("schema_version", 1) >= 2
    assert obj.get("cohort_probe_nonlinear_auc_roi_wmh_reads") is not None
    assert obj.get("validity_safe_headline_step")
    flags = {row["step"]: row.get("validity_flag") for row in obj.get("rows", [])}
    assert flags.get("fixed_tabular_plus_roi_wmh_reads") == "DESCRIPTIVE_ONLY_COHORT_ARTIFACT"


def test_method_y_readiness_split_blockers():
    path = Path("results/tables/method_y_training_readiness.json")
    assert path.exists()
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert "execution_blockers" in obj
    assert obj.get("pet_preprocess_progress") is not None


def test_bio_cog_summary_has_no_subject_level_vectors():
    path = Path("results/final_results/bio_cog_method_z/bio_cog_method_z_summary.json")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    assert "y_prob" not in text
    assert "y_true" not in text
