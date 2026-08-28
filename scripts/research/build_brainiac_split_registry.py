"""Build the private CAFT split registry (Plan 004 / B3).

--mode preflight : run the contradiction gate read-only; no writes
--mode build      : gate first, then materialize the private registry and the
                    public receipt (append-only; existing artifacts are
                    validated byte-for-byte and never overwritten)

On contract inconsistency the script writes the standard executor blocker
and exits 3 without producing any split artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import brainiac_split_registry as sr

BLOCKER_DIR = (
    ROOT / "results" / "method_x_registry" / "brainiac_r1" / "blockers"
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _write_blocker(detail: str, evidence: list[str]) -> int:
    import time

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    payload = {
        "schema_version": 1,
        "artifact": "BRAINIAC_EXECUTOR_BLOCKER",
        "plan_id": "004",
        "status": "BLOCKED",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason_code": "BRAINIAC_SPLIT_CONTRACT_INCONSISTENT",
        "evidence": [f"{detail}"] + evidence[:10],
        "writes_before_block": [],
        "source_digest_unchanged": True,
        "battery_artifacts_unchanged": True,
        "recommended_next_action": (
            "Operator resolves the split-authority inconsistency (counts above, no IDs); "
            "do not improvise fold or role changes"
        ),
    }
    _atomic_write_json(BLOCKER_DIR / f"BLOCKED_PLAN_004_{stamp}.json", payload)
    return 3


def preflight() -> dict[str, Any]:
    facts = sr.load_split_authority()
    validated = sr.contradiction_gate(facts)
    return {
        "status": "PASS",
        "real_rows": validated["real_rows"],
        "dev_rows": validated["dev_rows"],
        "sealed_rows": validated["sealed_rows"],
        "minority_value": validated["minority_value"],
        "fold_role_counts": {
            fold: {role: len(rows) for role, rows in row["roles"].items()}
            for fold, row in validated["folds"].items()
        },
    }


def build() -> dict[str, Any]:
    facts = sr.load_split_authority()
    validated = sr.contradiction_gate(facts)
    registry = sr.build_registry(validated)
    receipt = sr.build_public_receipt(registry)
    if sr.PRIVATE_REGISTRY_PATH.exists():
        existing = json.loads(sr.PRIVATE_REGISTRY_PATH.read_text(encoding="utf-8"))
        if existing.get("integrity_sha256") != registry["integrity_sha256"]:
            raise RuntimeError("BRAINIAC_SPLIT_REGISTRY_DRIFT_REFUSES_OVERWRITE")
        result = "REUSED"
    else:
        _atomic_write_json(sr.PRIVATE_REGISTRY_PATH, registry)
        result = "MATERIALIZED"
    if sr.PUBLIC_RECEIPT_PATH.exists():
        existing_receipt = json.loads(sr.PUBLIC_RECEIPT_PATH.read_text(encoding="utf-8"))
        if existing_receipt.get("integrity_sha256") != receipt["integrity_sha256"]:
            raise RuntimeError("BRAINIAC_SPLIT_PUBLIC_RECEIPT_DRIFT_REFUSES_OVERWRITE")
    else:
        _atomic_write_json(sr.PUBLIC_RECEIPT_PATH, receipt)
    sr.validate_registry()
    sr.validate_public_receipt()
    return {
        "status": "PASS",
        "result": result,
        "registry_integrity_sha256": registry["integrity_sha256"],
        "receipt_integrity_sha256": receipt["integrity_sha256"],
        "dev_rows": registry["dev_rows"],
        "sealed_rows": registry["sealed_rows"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "build"), default="preflight")
    args = parser.parse_args(argv)
    try:
        result = preflight() if args.mode == "preflight" else build()
    except RuntimeError as exc:
        if "SPLIT_CONTRACT_INCONSISTENT" in str(exc):
            detail = str(exc).split(":", 1)[-1]
            evidence = []
            try:
                facts = sr.load_split_authority()
                evidence = [
                    f"real_rows={len(facts['real_sample_hashes'])}",
                    f"synthetic_rows={facts['synthetic_rows']}",
                    f"duplicate_rows={facts['duplicate_rows']}",
                    f"labels_known={len(facts['labels'])}",
                ]
            except Exception:  # noqa: BLE001 - evidence is best-effort
                pass
            code = _write_blocker(detail, evidence)
            print(json.dumps({"status": "BLOCKED", "detail": detail}, indent=2))
            return code
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
