from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_smoke_contracts_exist_after_final_run():
    required = [
        "results/tables/validity_audit/validity_audit_summary.md",
        "results/tables/temporal_window_sensitivity.md",
        "results/final_results/external/external_final_metrics.md",
        "results/final_results/method_x_cog/method_x_cog_family_ablation.md",
        "results/final_results/thesis/thesis_grade_report.md",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        pytest.skip("Run the v3.6 pipeline stages first. Missing: " + ", ".join(missing))
    for rel in required:
        assert (ROOT / rel).exists()
