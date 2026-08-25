from __future__ import annotations

from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, read_json
from .contracts import sha256_mapping


class LauncherStateV3Error(RuntimeError):
    """Raised when the Plan 013 launcher stage machine is violated."""


STAGES = (
    "INIT",
    "PREFLIGHT_DONE",
    "DUAL_GATE_CONTRACT_VERIFIED",
    "READINESS_LOCKED",
    "PATTERN_CACHES_COMPLETE",
    "GPU_SMOKE_PASSED",
    "AMYLOID_INNER_HPO",
    "AMYLOID_MODELS_LOCKED",
    "AMYLOID_OUTER_COMMITTED",
    "TAU_INNER_HPO",
    "TAU_MODELS_LOCKED",
    "TAU_OUTER_COMMITTED",
    "ROBUSTNESS_COMPLETE",
    "REPORTS_COMPLETE",
    "CONSISTENCY_PRIVACY_PASS",
    "FINALIZED",
)


class LauncherStateV3:
    """Self-hashed, contract-bound stage machine for the unified launcher."""

    def __init__(self, *, output_root: str | Path, run_contract_sha256: str) -> None:
        if len(str(run_contract_sha256)) != 64:
            raise LauncherStateV3Error("RUN_CONTRACT_SHA256_REQUIRED")
        self.output_root = Path(output_root)
        self.run_contract_sha256 = str(run_contract_sha256)
        self.path = self.output_root / "orchestration" / "launcher_state.json"
        self.payload: dict[str, Any] = {"stage": "INIT", "facts": {}}
        if self.path.exists():
            stored = read_json(self.path)
            body = {k: v for k, v in stored.items() if k != "state_sha256"}
            if stored.get("state_sha256") != sha256_mapping(body):
                raise LauncherStateV3Error("STATE_SHA256_MISMATCH")
            if body.get("run_contract_sha256") != self.run_contract_sha256:
                raise LauncherStateV3Error("RUN_CONTRACT_MISMATCH:launcher_state")
            self.payload = body
        else:
            self._commit()

    def _commit(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "schema_version": "3.0",
            "run_contract_sha256": self.run_contract_sha256,
            "stage": self.payload.get("stage", "INIT"),
            "facts": self.payload.get("facts", {}),
        }
        atomic_write_json(self.path, {**body, "state_sha256": sha256_mapping(body)})

    @property
    def stage(self) -> str:
        return str(self.payload.get("stage", "INIT"))

    def set_stage(self, stage: str, **facts: Any) -> None:
        if stage not in STAGES:
            raise LauncherStateV3Error(f"UNKNOWN_STAGE:{stage}")
        current = STAGES.index(self.stage)
        target = STAGES.index(stage)
        if stage == "FINALIZED":
            missing = [s for s in STAGES[1:-1] if s not in (self.payload.get("facts", {}).get("completed", []))]
            if missing:
                raise LauncherStateV3Error(f"FINALIZE_BLOCKED:{','.join(missing)}")
        elif target < current:
            raise LauncherStateV3Error(f"STAGE_ORDER:cannot_move_backwards:{self.stage}->{stage}")
        if target > current + 1 and stage != "INIT":
            raise LauncherStateV3Error(f"STAGE_ORDER:skipped:{self.stage}->{stage}")
        self.payload["stage"] = stage
        merged = dict(self.payload.get("facts", {}))
        merged.update(facts)
        completed = set(merged.get("completed", []))
        completed.add(stage)
        merged["completed"] = sorted(completed)
        self.payload["facts"] = merged
        self._commit()

    def set_fact(self, key: str, value: Any) -> None:
        merged = dict(self.payload.get("facts", {}))
        merged[key] = value
        self.payload["facts"] = merged
        self._commit()

    def fact(self, key: str, default: Any = None) -> Any:
        return self.payload.get("facts", {}).get(key, default)
