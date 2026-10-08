"""Post-outer repair must preserve old evidence and authenticate the XAI child."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.mx_fasttrack_v2.run import sha
from scripts.mx_fasttrack_v2_4.supervise_v24 import Supervisor as V24Supervisor
from scripts.mx_fasttrack_v2_4_2.supervise_v242 import Supervisor as V242Supervisor
from scripts.mx_fasttrack_v2_4_3 import entry_v243, supervise_v243 as repair


def _write(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value,
                    encoding="utf-8")


def _parent_tree(tmp_path: Path, monkeypatch) -> Path:
    out = tmp_path / "out"
    reports = out / "reports"
    steps = {name: {"status": "COMPLETED", "returncode": 0}
             for name in repair.REQUIRED_COMPLETED}
    steps["mx6_v22_xai_crossfit"] = {
        "status": "PAUSED_OR_FAILED_RESUMABLE", "returncode": 3,
        "command_sha256": "a" * 64,
    }
    _write(reports / repair.V242_STATE, {"steps": steps})
    _write(reports / repair.V242_AMENDMENT, {"status": "LOCKED_BEFORE_FIRST_OUTER_ACCESS"})
    for name in (*repair.OUTER_REPORTS, "fasttrack_inner_lock.json", "xai_v22_context.json"):
        _write(reports / name, {"fixed": name})
    monkeypatch.setattr(repair, "ensure_v242_amendment", lambda *a, **k: {})
    monkeypatch.setattr(repair, "SOURCE_FILES", ())
    return out


def test_postouter_amendment_binds_old_journal_and_outer_bytes(tmp_path, monkeypatch):
    out = _parent_tree(tmp_path, monkeypatch)
    expected = repair.ensure_v243_amendment(tmp_path, out, create=True)
    assert expected["v242_failed_journal_sha256"] == sha(out / "reports" / repair.V242_STATE)
    assert expected["selection_or_training_changed"] is False
    assert repair.ensure_v243_amendment(tmp_path, out, create=False) == expected
    _write(out / "reports" / repair.OUTER_REPORTS[0], {"fixed": "changed"})
    with pytest.raises(RuntimeError, match="AMENDMENT_OR_BOUND_SOURCE_DRIFT"):
        repair.ensure_v243_amendment(tmp_path, out, create=False)


def test_new_journal_preserves_completed_steps_and_failed_command_provenance(tmp_path, monkeypatch):
    out = _parent_tree(tmp_path, monkeypatch)

    def fake_init(self, root, output_relative, package_root):
        self.root = root
        self.output_relative = output_relative
        self.out = root / output_relative
        self.reports = self.out / "reports"
        self.state_path = self.reports / repair.V242_STATE
        self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(V242Supervisor, "__init__", fake_init)
    supervisor = repair.Supervisor(tmp_path, Path("out"), tmp_path)
    assert supervisor.state_path == out / "reports" / repair.V243_STATE
    assert "mx6_v22_xai_crossfit" not in supervisor.state["steps"]
    assert supervisor.state["migration"]["failed_xai_step"]["command_sha256"] == "a" * 64
    assert all(supervisor.state["steps"][name]["status"] == "COMPLETED"
               for name in repair.REQUIRED_COMPLETED)
    assert json.loads((out / "reports" / repair.V242_STATE).read_text())["steps"]\
        ["mx6_v22_xai_crossfit"]["status"] == "PAUSED_OR_FAILED_RESUMABLE"

    captured = []
    monkeypatch.setattr(V24Supervisor, "run_step",
                        lambda self, name, command, **kwargs: captured.append(command) or 0)
    supervisor.run_step("mx6_v22_xai_crossfit", ["old-xai"])
    assert captured[0][1].endswith("entry_v243.py")
    assert captured[0][2] == "xai-run"


@pytest.mark.parametrize("launcher_hop", [False, True])
def test_xai_guard_accepts_only_live_v243_ancestry(tmp_path, monkeypatch, launcher_hop):
    coordinator = {"ProcessId": 100, "ParentProcessId": 1,
                   "CommandLine": r"python H:\New project\scripts\mx_fasttrack_v2_4_3\supervise_v243.py all"}
    launcher = {"ProcessId": 200, "ParentProcessId": 100,
                "CommandLine": r"python H:\New project\scripts\mx_fasttrack_v2_4_3\entry_v243.py xai-run"}
    monkeypatch.setenv(entry_v243.INTERNAL_PARENT_ENV, "100")
    monkeypatch.setattr(entry_v243.os, "getppid", lambda: 200 if launcher_hop else 100)
    monkeypatch.setattr(entry_v243, "windows_python_processes",
                        lambda **kwargs: [coordinator, launcher])
    entry_v243.require_v243_coordinator(tmp_path)
    launcher["ParentProcessId"] = 999
    if launcher_hop:
        with pytest.raises(RuntimeError, match="ACTIVE_COORDINATOR_REQUIRED"):
            entry_v243.require_v243_coordinator(tmp_path)
    coordinator["CommandLine"] = "python unrelated.py"
    with pytest.raises(RuntimeError, match="ACTIVE_COORDINATOR_REQUIRED"):
        entry_v243.require_v243_coordinator(tmp_path)


def test_amendment_rejects_missing_outer_or_frozen_parent(tmp_path, monkeypatch):
    out = _parent_tree(tmp_path, monkeypatch)
    (out / "reports" / repair.OUTER_REPORTS[0]).unlink()
    with pytest.raises(RuntimeError, match="PARENT_RECEIPT_MISSING"):
        repair.ensure_v243_amendment(tmp_path, out, create=True)
    _write(out / "reports" / repair.OUTER_REPORTS[0], {})
    journal = json.loads((out / "reports" / repair.V242_STATE).read_text())
    journal["steps"]["mx7_freeze_scoped_report"] = {"status": "COMPLETED"}
    _write(out / "reports" / repair.V242_STATE, journal)
    with pytest.raises(RuntimeError, match="PARENT_JOURNAL_NOT_AT_XAI_FAILURE"):
        repair.ensure_v243_amendment(tmp_path, out, create=True)
