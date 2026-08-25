from __future__ import annotations

import json
from pathlib import Path

import pytest


RC = "a" * 64


def _state(root):
    from src.method_y_revival.image_launcher_state_v3 import LauncherStateV3

    return LauncherStateV3(output_root=root, run_contract_sha256=RC)


def test_stage_order_enforced(tmp_path):
    st = _state(tmp_path)
    with pytest.raises(Exception, match="STAGE_ORDER"):
        st.set_stage("GPU_SMOKE_PASSED")  # skips four required predecessors
    st.set_stage("PREFLIGHT_DONE")
    st.set_stage("DUAL_GATE_CONTRACT_VERIFIED")
    st.set_stage("READINESS_LOCKED")
    st.set_stage("PATTERN_CACHES_COMPLETE")
    st.set_stage("GPU_SMOKE_PASSED")


def test_same_stage_rewrite_is_idempotent(tmp_path):
    st = _state(tmp_path)
    st.set_stage("INIT")
    st.set_stage("INIT")
    assert json.loads((tmp_path / "orchestration" / "launcher_state.json").read_text(encoding="utf-8"))["stage"] == "INIT"


def test_tamper_detected_on_reload(tmp_path):
    st = _state(tmp_path)
    st.set_stage("INIT")
    p = tmp_path / "orchestration" / "launcher_state.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["stage"] = "FINALIZED"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="STATE_SHA256_MISMATCH|RUN_CONTRACT_MISMATCH"):
        _state(tmp_path)


def test_contract_mismatch_fails_closed(tmp_path):
    st = _state(tmp_path)
    st.set_stage("INIT")
    with pytest.raises(Exception, match="RUN_CONTRACT_MISMATCH"):
        from src.method_y_revival.image_launcher_state_v3 import LauncherStateV3

        LauncherStateV3(output_root=tmp_path, run_contract_sha256="b" * 64)


def test_finalized_requires_all_previous_stages(tmp_path):
    from src.method_y_revival.image_launcher_state_v3 import STAGES, LauncherStateV3

    st = _state(tmp_path)
    for s in STAGES[:-2]:  # everything except CONSISTENCY_PRIVACY_PASS and FINALIZED
        st.set_stage(s)
    with pytest.raises(Exception, match="FINALIZE_BLOCKED"):
        st.set_stage("FINALIZED")
    st.set_stage("CONSISTENCY_PRIVACY_PASS")
    st.set_stage("FINALIZED")
