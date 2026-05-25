import pytest

from src.evaluation.stats_tests import delong_roc_auc_guarded


def test_delong_guard_blocks_cv():
    with pytest.raises(ValueError):
        delong_roc_auc_guarded(
            [0, 1],
            [0.2, 0.8],
            [0.1, 0.9],
            {"split_type": "cv_fold", "external_is_truly_independent": False, "name": "fold_0"},
        )


def test_delong_guard_allows_true_external_binary():
    out = delong_roc_auc_guarded(
        [0, 1],
        [0.2, 0.8],
        [0.1, 0.9],
        {"split_type": "external_independent_binary", "external_is_truly_independent": True, "name": "external"},
    )
    assert out["status"] == "ALLOWED"

