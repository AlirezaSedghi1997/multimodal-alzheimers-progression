"""Only the five proof-bound historical checkpoint views may be projected."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.mx_fasttrack_v2_4_2 import checkpoint_projection as projection
from scripts.mx_fasttrack_v2_4_2.supervise_v242 import Supervisor as V242Supervisor
from scripts.mx_fasttrack_v2_4.supervise_v24 import Supervisor as V24Supervisor


def test_projection_is_caller_scoped_and_restores_torch_load(tmp_path, monkeypatch):
    import torch

    checkpoint = tmp_path / "method_x_3d_fold_0.pt"
    checkpoint.write_bytes(b"unchanged checkpoint fixture")
    old, new = "a" * 64, "b" * 64
    fake_payload = {"fold": 0, "run_contract_digest": old,
                    "hpo_selection_only": True, "outer_test_access_count": 0,
                    "outer_test_evaluation_count": 0}
    original = torch.load
    monkeypatch.setattr(torch, "load", lambda path, *a, **k: dict(fake_payload))
    stub = torch.load
    monkeypatch.setattr(projection, "baseline_binding", lambda root, out: {
        "old_digest": old, "new_digest": new, "paths": {checkpoint.resolve(): 0}})
    with projection.evaluator_checkpoint_projection(tmp_path, tmp_path):
        # Non-evaluator callers, including migration verifiers, see old bytes.
        assert torch.load(checkpoint)["run_contract_digest"] == old
        monkeypatch.setattr(projection, "ALLOWED_CALLERS", {Path(__file__).name})
        viewed = torch.load(checkpoint)
        assert viewed["run_contract_digest"] == new
        assert fake_payload["run_contract_digest"] == old
    assert torch.load is stub
    assert checkpoint.read_bytes() == b"unchanged checkpoint fixture"
    monkeypatch.setattr(torch, "load", original)


def test_projection_rejects_wrong_historical_identity(tmp_path, monkeypatch):
    import torch

    checkpoint = tmp_path / "method_x_3d_fold_0.pt"
    checkpoint.write_bytes(b"fixture")
    old, new = "a" * 64, "b" * 64
    monkeypatch.setattr(projection, "baseline_binding", lambda root, out: {
        "old_digest": old, "new_digest": new, "paths": {checkpoint.resolve(): 0}})
    monkeypatch.setattr(projection, "ALLOWED_CALLERS", {Path(__file__).name})
    monkeypatch.setattr(torch, "load", lambda path, *a, **k: {
        "fold": 0, "run_contract_digest": "c" * 64,
        "hpo_selection_only": True, "outer_test_access_count": 0,
        "outer_test_evaluation_count": 0})
    with projection.evaluator_checkpoint_projection(tmp_path, tmp_path):
        with pytest.raises(RuntimeError, match="HISTORICAL_CHECKPOINT_IDENTITY_INVALID"):
            torch.load(checkpoint)


def test_v242_supervisor_uses_new_journal_and_wrapper(tmp_path, monkeypatch):
    captured = []

    def fake_init(self, root, output_relative, package_root):
        self.root = root
        self.output_relative = output_relative
        self.out = root / output_relative
        self.reports = self.out / "reports"
        self.reports.mkdir(parents=True)

    def fake_run_step(self, name, command, *, allowed=None):
        captured.append((name, command))
        return 0

    monkeypatch.setattr(V24Supervisor, "__init__", fake_init)
    monkeypatch.setattr(V24Supervisor, "run_step", fake_run_step)
    supervisor = V242Supervisor(tmp_path, Path("results/versioned"), tmp_path)
    assert supervisor.state_path.name == "fasttrack_v242_supervisor_state.json"
    supervisor.run_step("mx5_outer_hybrid_pair", ["old-command"])
    assert captured[0][1][1].endswith("entry_v242.py")
    assert captured[0][1][2] == "hybrid-pair"


def test_real_historical_migration_binds_all_five_folds():
    root = projection.Path(__file__).resolve().parents[1]
    out = root / "results/final_results/method_x_fasttrack_v2_20260927_mx05_gate_import"
    if not out.is_dir():
        pytest.skip("Authoritative Method X artifact tree not present")
    binding = projection.baseline_binding(root, out)
    assert set(binding["paths"].values()) == set(range(5))
    assert binding["old_digest"] != binding["new_digest"]
