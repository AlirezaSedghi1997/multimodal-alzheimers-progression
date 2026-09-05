"""Unit tests for the MX-03 execution watchdog (pure logic, no processes)."""
from __future__ import annotations


def test_classify_safe_pause_relaunches():
    from scripts.mx03_execution_watchdog import classify_last_outcome

    state = {"tasks": {"random_labels": {"status": "PAUSED_RESUMABLE", "execution": {"returncode": 75}}}}
    out = classify_last_outcome(state, {"blockers": ["TASK_PAUSED_RESUMABLE:random_labels"]})
    assert out["class"] == "SAFE_PAUSE" and out["relaunch"] is True


def test_classify_hard_block_never_relaunches():
    from scripts.mx03_execution_watchdog import classify_last_outcome

    state = {"tasks": {"random_labels": {"status": "BLOCKED",
                                          "execution": {"returncode": 2}}}}
    out = classify_last_outcome(state, {"blockers": ["TASK_EXECUTION_BLOCKED:random_labels"]})
    assert out["class"] == "HARD_BLOCK" and out["relaunch"] is False


def test_classify_contract_mismatch_is_hard():
    from scripts.mx03_execution_watchdog import classify_last_outcome

    out = classify_last_outcome({}, {"blockers": ["RUN_CONTRACT_MISMATCH:abc"]})
    assert out["class"] == "HARD_BLOCK" and out["relaunch"] is False


def test_classify_unknown_never_relaunches():
    from scripts.mx03_execution_watchdog import classify_last_outcome

    out = classify_last_outcome({}, {"blockers": ["SOMETHING_UNSEEN"]})
    assert out == {"class": "UNKNOWN", "task": None, "relaunch": False}


def test_classify_running_waits():
    from scripts.mx03_execution_watchdog import classify_last_outcome

    state = {"tasks": {"random_labels": {"status": "RUNNING", "pid": 12345}}}
    out = classify_last_outcome(state, {"blockers": []})
    assert out["class"] == "RUNNING" and out["relaunch"] is False


def test_command_safety_rejects_no_resume():
    from scripts.mx03_execution_watchdog import command_is_safe

    assert command_is_safe(["python", "x.py", "--no_resume"]) == (False, "FORBIDDEN_FLAG_--no_resume")
    assert command_is_safe(["python", "x.py", "--mode", "run"])[0] is True
