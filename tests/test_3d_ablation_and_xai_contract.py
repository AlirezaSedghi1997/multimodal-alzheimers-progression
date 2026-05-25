from pathlib import Path


def test_3d_ablation_summary_is_aggregate_and_privacy_safe():
    from scripts.run_3d_ablation import summarize

    summary = summarize(
        [
            {
                "aggregate": {
                    "ablation_variant": "concat",
                    "mode": "FULL_3D_TRAINING",
                    "folds": 5,
                    "cache_resolution": 96,
                    "roc_auc": 0.7,
                    "pr_auc": 0.8,
                    "brier": 0.2,
                    "external_like": {"roc_auc": 0.71},
                    "uses_pet_as_input": False,
                    "uses_raw_identifiers": False,
                    "final_claim_allowed": True,
                }
            }
        ]
    )
    assert summary["status"] == "COMPLETED"
    assert summary["contains_pet_input"] is False
    assert summary["contains_raw_identifiers"] is False
    assert summary["public_output_granularity"] == "aggregate_variant_level"


def test_xai_blocked_report_is_privacy_safe(tmp_path, monkeypatch):
    from scripts.run_xai_summary import write_blocked

    monkeypatch.chdir(tmp_path)
    write_blocked(Path("results/final_results/xai"), "missing_cache")
    text = Path("results/final_results/xai/xai_summary.json").read_text(encoding="utf-8")
    assert '"contains_pet_input": false' in text
    assert '"contains_raw_identifiers": false' in text
    assert '"contains_images": false' in text


def test_xai_tabular_reader_accepts_unordered_h5_indices():
    import pytest

    torch = pytest.importorskip("torch")
    from scripts.run_xai_summary import read_tabular_rows

    class FakeDataset:
        def __getitem__(self, index):
            if isinstance(index, list):
                raise TypeError("Indexing elements must be in increasing order")
            return [float(index), float(index + 1)]

    rows = read_tabular_rows({"tabular": FakeDataset()}, [2, 0], torch.device("cpu"))
    assert rows.tolist() == [[2.0, 3.0], [0.0, 1.0]]
