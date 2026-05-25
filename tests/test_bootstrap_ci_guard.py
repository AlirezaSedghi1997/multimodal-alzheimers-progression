from src.evaluation.bootstrap_ci import guarded_bootstrap_ci


def _metric(y, s):
    return sum(s) / len(s)


def test_guarded_bootstrap_handles_single_class():
    out = guarded_bootstrap_ci([1, 1, 1], [0.7, 0.8, 0.9], _metric, n_boot=10)
    assert out["status"] == "SINGLE_CLASS"


def test_guarded_bootstrap_handles_binary():
    out = guarded_bootstrap_ci([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], _metric, n_boot=10)
    assert out["status"] == "OK"

