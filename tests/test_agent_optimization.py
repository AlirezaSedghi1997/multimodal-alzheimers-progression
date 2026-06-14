import json
import tempfile
from pathlib import Path

import pytest


def test_agent_token_tracker_basic():
    from src.utils.agent_optimization import AgentTokenTracker

    tracker = AgentTokenTracker()
    tracker.record_call("pytest -q", 100, category="testing")
    tracker.record_call("git status", 50, category="git")
    summary = tracker.get_summary()
    assert summary["total_calls"] == 2
    assert summary["total_tokens"] == 150
    assert "testing" in summary["category_stats"]
    assert "git" in summary["category_stats"]


def test_agent_token_tracker_persistence():
    from src.utils.agent_optimization import AgentTokenTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_file = Path(tmpdir) / "tracking.json"
        tracker = AgentTokenTracker(tracking_file=tracking_file)
        tracker.record_call("pytest -q", 100, category="testing")
        assert tracking_file.exists()
        tracker2 = AgentTokenTracker(tracking_file=tracking_file)
        assert tracker2.total_tokens == 100
        assert len(tracker2.calls) == 1


def test_should_spawn_subagent():
    from src.utils.agent_optimization import AgentTokenTracker

    tracker = AgentTokenTracker()
    assert tracker.should_spawn_subagent(0.5, 200) is True
    assert tracker.should_spawn_subagent(0.1, 200) is False
    assert tracker.should_spawn_subagent(0.5, 50) is False


def test_compact_pytest():
    from src.utils.agent_optimization import CompactOutputFormatter

    short_output = "line1\nline2\nline3"
    assert CompactOutputFormatter.compact_pytest(short_output) == short_output
    long_output = "\n".join([f"line{i}" for i in range(100)])
    compact = CompactOutputFormatter.compact_pytest(long_output, max_lines=10)
    assert len(compact.split("\n")) < 20


def test_compact_json():
    from src.utils.agent_optimization import CompactOutputFormatter

    small_data = {"a": 1, "b": 2}
    result = CompactOutputFormatter.compact_json(small_data)
    assert "a" in result
    large_data = {f"key_{i}": i for i in range(30)}
    result = CompactOutputFormatter.compact_json(large_data, max_keys=10)
    assert "_remaining_keys" in result


def test_compact_metrics():
    from src.utils.agent_optimization import CompactOutputFormatter

    metrics = {
        "roc_auc": 0.85,
        "pr_auc": 0.90,
        "brier": 0.15,
        "accuracy": 0.88,
        "irrelevant_key": "value",
    }
    compact = CompactOutputFormatter.compact_metrics(metrics)
    assert "roc_auc" in compact
    assert "irrelevant_key" not in compact
