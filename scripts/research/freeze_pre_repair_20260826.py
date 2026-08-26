"""One-shot P0 freeze: capture pre-repair state before any reconciliation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY = ROOT / "results" / "method_x_registry" / "brainiac_r1"
OUT = REGISTRY / "BRAINIAC_PREPROCESSING_PRE_REPAIR_FREEZE_20260826.json"


def _sha(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False}
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _pid_live(pid: int):
    try:
        import psutil
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    try:
        subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, timeout=10, check=False)
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"if(Get-Process -Id {pid} -ErrorAction SilentlyContinue){{'LIVE'}}else{{'DEAD'}}"],
            capture_output=True, text=True, timeout=15, check=False)
        return "LIVE" in (out.stdout or "")
    except Exception:
        return None


def main() -> int:
    body = {
        "schema_version": 1,
        "artifact": "BRAINIAC_PREPROCESSING_PRE_REPAIR_FREEZE_20260826",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "branch": subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "method_x_source_digest": None,
        "battery_manifest_sha256": _sha(ROOT / "results" / "final_results" / "method_x_model_battery_v3_depth_and_diversity_20260729" / "battery_manifest.json"),
        "policy_preprocessing_v2": _sha(ROOT / "governance" / "brainiac_preprocessing_r1_primary_v2_policy.json"),
        "config_yaml": _sha(ROOT / "configs" / "research" / "brainiac_preprocessing_r1_primary_v2.yaml"),
        "runner_script": _sha(ROOT / "scripts" / "research" / "prepare_brainiac_raw_t1_v2.py"),
        "v2_module": _sha(ROOT / "src" / "research" / "brainiac_preprocessing_v2.py"),
        "v2_schema_module": _sha(ROOT / "src" / "research" / "brainiac_preprocessing_v2_schema.py"),
        "calibration_receipts_n": 0,
        "calibration_policy_binding_sample": [],
        "calibration_dead_locks": [],
        "primary_receipts_n": 0,
        "equivalence_artifact": _sha(REGISTRY / "brainiac_preprocessing_v2_equivalence.json"),
        "superseded_moves_planned": [
            {"from": "data/derived/method_x_brainiac/r1_primary_v2_calibration",
             "to": "data/derived/method_x_brainiac/superseded/calibration_pre_pin_policy_f5c69a",
             "reason": "POLICY_GENERATION_MISMATCH_F5C69A_VS_140ACC + DEAD_PID_LOCKS"},
            {"from": "data/derived/method_x_brainiac/r1_primary_v2_smoke/eq_sequential",
             "to": "data/derived/method_x_brainiac/superseded/equivalence_partial_20260825/eq_sequential",
             "reason": "EQUIVALENCE_INCOMPLETE_ABORTED_RUN"},
            {"from": "data/derived/method_x_brainiac/r1_primary_v2_smoke/eq_parallel",
             "to": "data/derived/method_x_brainiac/superseded/equivalence_partial_20260825/eq_parallel",
             "reason": "EQUIVALENCE_INCOMPLETE_ABORTED_RUN"},
            {"from": "results/method_x_registry/brainiac_r1/brainiac_preprocessing_v2_equivalence.json",
             "to": "data/derived/method_x_brainiac/superseded/equivalence_partial_20260825/",
             "reason": "SUPERSEDED_BY_GATE_V2"},
        ],
        "dead_pids_to_prove": [19216, 1360],
    }
    try:
        from scripts.train_3d_mri import method_x_training_source_contract
        body["method_x_source_digest"] = method_x_training_source_contract()[0]
    except Exception as exc:
        body["method_x_source_digest_error"] = type(exc).__name__

    cal_root = ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2_calibration"
    receipts_dir = cal_root / "receipts"
    if receipts_dir.is_dir():
        recs = sorted(receipts_dir.glob("*.json"))
        body["calibration_receipts_n"] = len(recs)
        for r in recs:
            try:
                payload = json.loads(r.read_text(encoding="utf-8"))
                body["calibration_policy_binding_sample"].append({
                    "sample_hash_prefix": str(payload.get("sample_hash", ""))[:10],
                    "policy_sha256_prefix": str(payload.get("policy_sha256", ""))[:12],
                })
            except Exception:
                pass
    journals = cal_root / "journals"
    if journals.is_dir():
        for lock in journals.rglob(".sample.lock"):
            try:
                lp = json.loads(lock.read_text(encoding="utf-8"))
                pid = int(lp.get("pid", -1))
                live = _pid_live(pid)
                entry = {"path_suffix": str(lock.relative_to(cal_root)), "pid": pid, "live": live}
                if not live:
                    entry["action"] = "MOVE_WITH_PARENT_AFTER_DEAD_PROOF"
                body["calibration_dead_locks"].append(entry)
            except Exception:
                pass
    prim = ROOT / "data" / "derived" / "method_x_brainiac" / "r1_primary_v2" / "receipts"
    if prim.is_dir():
        body["primary_receipts_n"] = len(list(prim.glob("*.json")))
    shared_guard = ROOT / "results" / "final_results" / "method_x_model_battery_v3_depth_and_diversity_20260729" / ".method_x_reconciliation.lock"
    body["shared_guard"] = _sha(shared_guard)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(OUT))
    print(json.dumps({"status": "FROZEN", "artifact": str(OUT),
                      "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
