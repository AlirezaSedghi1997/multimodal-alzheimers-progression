from __future__ import annotations

import json
from pathlib import Path


def test_phase2_mvp_audit_exists():
    path = Path("results/tables/phase2_mvp/phase2_mvp_status.json")
    assert path.exists(), "Run scripts/run_phase2_mvp_audit.py"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] in {"MVP_PARTIAL", "MVP_COMPLETE"}
    assert "method_x_strict_tabular" in obj["models"]
