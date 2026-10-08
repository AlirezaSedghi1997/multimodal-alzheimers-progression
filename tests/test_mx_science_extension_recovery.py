"""Real CLI failure reproduction and fail-closed extension continuation."""
from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from scripts.mx_fasttrack_v2_4_4 import supervise_v244 as historical
from scripts.mx_science_extension_v1 import supervise_extension as supervisor
from scripts.mx_science_extension_v1 import outer, process_log


def test_historical_console_failure_occurs_after_work(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(historical, "ROOT", tmp_path)
    monkeypatch.setattr(historical, "foreign_method_x_writers", lambda root: [])
    monkeypatch.setattr(historical, "coordinator_lock", lambda path: nullcontext())
    monkeypatch.setattr(historical, "ensure_v244_amendment", lambda *a, **kw: {})
    monkeypatch.setattr(historical, "_patch_parent_checkers", lambda: None)
    class Finished:
        def __init__(self, *args):
            pass
        def mx7(self):
            calls.append("finished")
    monkeypatch.setattr(historical, "Supervisor", Finished)
    monkeypatch.setattr(sys, "argv", ["supervise_v244.py", "all"])
    # Historical source is immutable: reproduce its exact bug, repair at caller.
    with pytest.raises(TypeError) as exc:
        historical.main()
    assert str(exc.value) == supervisor.V244_CONSOLE_ERROR
    assert calls == ["finished"]


@pytest.mark.parametrize("mode", ["console", "unrelated", "incomplete"])
def test_parent_recovers_only_console_failure_with_terminal_proof(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "OUTPUT", Path("extension"))
    monkeypatch.setattr(supervisor, "PARENT_OUTPUT", Path("parent"))
    monkeypatch.setattr(supervisor, "ensure_v242_amendment", lambda *a, **kw: {})
    monkeypatch.setattr(supervisor, "ensure_v244_amendment", lambda *a, **kw: {})
    monkeypatch.setattr(supervisor, "verify_frozen_scope", lambda *a, **kw: {})
    calls = []
    terminal = {"status": "VERIFIED_PARENT_MX7_TERMINAL", "freeze_sha256": "a" * 64}
    def verify(root):
        calls.append("check")
        if len(calls) == 1 or mode == "incomplete":
            raise RuntimeError("NOT_TERMINAL")
        return terminal
    monkeypatch.setattr(supervisor, "require_parent_mx7_terminal", verify)
    reason = supervisor.V244_CONSOLE_ERROR if mode != "unrelated" else "CHECKPOINT_CHANGED"
    monkeypatch.setattr(supervisor, "run_logged", lambda *a, **kw:
                        subprocess.CompletedProcess([], 3, stdout=json.dumps(
                            {"status": "BLOCKED", "reason": reason})))
    instance = supervisor.Supervisor(tmp_path)
    receipt = instance.reports / "parent_v244_console_recovery.v1.json"
    if mode == "console":
        assert instance.parent() == terminal
        assert json.loads(receipt.read_text())["locked_parent_source_modified"] is False
    else:
        with pytest.raises(RuntimeError):
            instance.parent()
        assert not receipt.exists()


def test_verified_parent_never_relaunches(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "OUTPUT", Path("extension"))
    monkeypatch.setattr(supervisor, "ensure_v242_amendment", lambda *a, **kw: {})
    monkeypatch.setattr(supervisor, "require_parent_mx7_terminal",
                        lambda root: {"status": "VERIFIED_PARENT_MX7_TERMINAL"})
    monkeypatch.setattr(supervisor, "run_logged", lambda *a, **kw: pytest.fail("relaunch"))
    assert supervisor.Supervisor(tmp_path).parent()["status"] == "VERIFIED_PARENT_MX7_TERMINAL"


def test_full_sequence_retries_interrupted_step_and_stops_at_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "OUTPUT", Path("extension"))
    monkeypatch.setattr(supervisor, "foreign_method_x_writers", lambda root: [])
    for name in ("check_inputs", "verify_frozen_scope", "xai_audit_payload",
                 "preflight_arms", "ensure_execution_lock"):
        monkeypatch.setattr(supervisor, name, lambda *a, **kw: {})
    monkeypatch.setattr(supervisor, "sha", lambda p: "a" * 64)
    monkeypatch.setattr(supervisor.Supervisor, "parent", lambda self: {"status": "VERIFIED"})
    invocations = []
    def arms(root):
        invocations.append("arms")
        if len(invocations) == 1:
            raise RuntimeError("INTERRUPTED")
        return {"status": "DONE"}
    monkeypatch.setattr(supervisor, "run_arms", arms)
    for name in ("lock_selection", "ensure_outer_lock", "run_outer", "run_analysis",
                 "run_xai_audit", "run_report"):
        monkeypatch.setattr(supervisor, name, lambda *a, **kw: {"status": "DONE"})
    instance = supervisor.Supervisor(tmp_path)
    assert instance.all(stop_after_lock=True)["status"].startswith("READY_TO_RESUME")
    assert invocations == []
    with pytest.raises(RuntimeError, match="INTERRUPTED"):
        instance.all()
    assert "inner_selection" not in instance.journal["steps"]
    resumed = supervisor.Supervisor(tmp_path)
    resumed.all()
    assert all(resumed.journal["steps"][name]["status"] == "COMPLETED"
               for name in supervisor.STAGES)
    assert resumed.journal["steps"]["six_inner_arms"]["attempts"] == 2


def test_child_stdout_stderr_failure_and_retry_are_preserved(tmp_path, capsys):
    log = tmp_path / "child.log"
    command = [sys.executable, "-c", "import sys; print('fold 2'); print('broken', file=sys.stderr); sys.exit(3)"]
    result = process_log.run_logged(command, cwd=tmp_path, log_path=log)
    assert result.returncode == 3
    assert "fold 2" in result.stdout and "broken" in result.stdout
    result = process_log.run_logged([sys.executable, "-c", "print('resumed')"],
                                    cwd=tmp_path, log_path=log)
    assert result.returncode == 0
    saved = log.read_text(encoding="utf-8")
    assert all(value in saved for value in ("fold 2", "broken", "resumed"))
    assert "resumed" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [[float("nan")], [float("inf")], [-0.1], [1.1], [True], [], None])
def test_bad_probabilities_rejected_before_commit(bad):
    with pytest.raises(RuntimeError, match="PROBABILITIES_INVALID"):
        outer._validate_probabilities(bad, 1, "test")
