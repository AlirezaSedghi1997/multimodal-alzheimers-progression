"""Adversarial checks for the versioned MX-05 import control plane."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.mx_fasttrack_v2.run import sha
from scripts.mx_fasttrack_v2_4.import_resume import (
    ALLOWED_PARENT_DRIFT, PROTOCOL, _copy_exact, _seed_mx04_task_state,
)
from scripts.mx_fasttrack_v2_4.supervise_v24 import CONFIG, outer_command


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v24_protocol_has_a_dedicated_unchanged_scientific_config():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    assert config["protocol_id"] == PROTOCOL
    assert (config["seed"], config["folds"], config["locked_epochs_field"],
            config["effective_epoch_cap_per_fold"]) == (13, 5, 100, 20)
    assert config["outer_test_used_for_selection"] is False
    assert config["external_validation"] is False
    assert ALLOWED_PARENT_DRIFT == {
        "scripts/train_3d_mri.py", "src/training/method_x_resume.py",
        "scripts/mx_fasttrack_v2/run.py", "scripts/mx_fasttrack_v2/integrity.py",
    }


def test_import_copy_is_idempotent_and_refuses_destination_tampering(tmp_path):
    source = tmp_path / "source.pt"
    target = tmp_path / "target.pt"
    source.write_bytes(b"immutable model bytes")
    assert _copy_exact(source, target) == sha(source)
    assert _copy_exact(source, target) == sha(source)
    target.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="V24_IMPORT_DESTINATION_DRIFT"):
        _copy_exact(source, target)


def test_crash_after_receipt_can_seed_task_state_once(tmp_path):
    out = tmp_path / "out"
    _json(out / "reports/fasttrack_preregistration.json", {"protocol_id": PROTOCOL})
    receipt = out / "reports/mx05_resume_import.v1.json"
    _json(receipt, {"status": "IMPORTED_UNDER_HASH_PROOF"})
    old = {"tasks": {
        "mx04_scratch": {"status": "COMPLETED", "command_sha256": "old"},
        "mx04_head_last": {"status": "COMPLETED", "command_sha256": "old"},
    }}
    commands = {"mx04_scratch": "new-s", "mx04_head_last": "new-h"}
    for task in commands:
        _json(out / "inner_tasks" / task / "three_d_metrics.json", {"aggregate": {}})
    _seed_mx04_task_state(out, old, commands, receipt)
    state_path = out / "reports/fasttrack_task_state.json"
    first = sha(state_path)
    _seed_mx04_task_state(out, old, commands, receipt)
    assert sha(state_path) == first
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["tasks"]["mx04_scratch"]["command_sha256"] == "new-s"
    state["tasks"]["mx04_scratch"]["command_sha256"] = "tampered"
    _json(state_path, state)
    with pytest.raises(RuntimeError, match="V24_IMPORT_TASK_STATE_DRIFT"):
        _seed_mx04_task_state(out, old, commands, receipt)


def test_outer_wrapper_requires_an_explicit_task_only_for_locked_evaluation():
    root = Path(__file__).resolve().parents[1]
    output = Path("results/final_results/method_x_fasttrack_v2_20260927_mx05_gate_import")
    command = outer_command(root, output, "locked", task="selected")
    assert command[-4:] == ["--scope", "outer", "--task", "selected"]
    with pytest.raises(RuntimeError, match="V24_OUTER_TASK_INVALID"):
        outer_command(root, output, "locked", task=None)
    with pytest.raises(RuntimeError, match="V24_PAIR_TASK_FORBIDDEN"):
        outer_command(root, output, "hybrid-pair", task="selected")
