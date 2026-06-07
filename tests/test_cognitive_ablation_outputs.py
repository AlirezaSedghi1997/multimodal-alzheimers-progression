import json
from pathlib import Path

from scripts.run_cognitive_ablation import MODEL_VARIANTS


def test_required_method_x_cog_ablation_variants_exist():
    required = {
        "fixed_tabular_only",
        "cognitive_core_only",
        "fixed_tabular_plus_cognitive_core",
        "fixed_tabular_plus_all_cognitive",
        "mri_feature_surrogate_plus_cognitive",
    }
    assert required.issubset(set(MODEL_VARIANTS))


def test_method_x_cog_summary_if_generated_is_aggregate_safe():
    path = Path("results/final_results/method_x_cog/method_x_cog_summary.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["uses_pet_as_input"] is False
    assert obj["uses_raw_identifiers"] is False
    assert obj["method_x_claim_unchanged"] is True
    assert obj["method_x_cog_exploratory"] is True
