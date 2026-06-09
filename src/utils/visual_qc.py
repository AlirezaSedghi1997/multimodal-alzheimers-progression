from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.pet_spatial_qc import atomic_write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_pet_qc_payload(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_visual_review_manifest(
    pet_qc_dir: Path,
    *,
    project_root: Path | None = None,
    pass_sample_fraction: float = 0.05,
    seed: int = 42,
    ncc_warn_below: float = 0.75,
) -> dict[str, Any]:
    """Build manifest: 100% FAIL/WARN/low-NCC + random PASS fraction for human review."""
    root = project_root or Path(".")
    items: list[dict[str, Any]] = []
    pass_candidates: list[dict[str, Any]] = []

    for qc_path in sorted(pet_qc_dir.glob("*.json")):
        payload = _load_pet_qc_payload(qc_path)
        if not payload:
            continue
        sample_hash = str(payload.get("sample_hash") or qc_path.stem)
        status = str(payload.get("status") or "UNKNOWN")
        gate = payload.get("gate") or {}
        qc_after = payload.get("qc_after") or {}
        ncc = qc_after.get("ncc_pet_t1")
        signed = bool(payload.get("visual_qc_signoff"))
        tracer = str(payload.get("tracer") or "fbp").lower()
        overlay_rel = Path("results/qc/method_y_pet/overlays") / tracer / f"{sample_hash}_pet_on_t1.png"
        overlay_exists = (root / overlay_rel).exists()

        item = {
            "sample_hash": sample_hash,
            "pet_qc_json": str(qc_path.as_posix()),
            "status": status,
            "gate_status": gate.get("status"),
            "gate_failures": gate.get("failures") or [],
            "ncc_pet_t1": ncc,
            "overlay_path": overlay_rel.as_posix() if overlay_exists else None,
            "visual_qc_signoff": signed,
            "review_required_reason": None,
            "review_decision": payload.get("visual_qc_decision"),
            "signed_at": payload.get("visual_qc_signed_at"),
        }

        mandatory = False
        if status in {"QC_FAIL", "FAIL"}:
            item["review_required_reason"] = "QC_FAIL"
            mandatory = True
        elif gate.get("warnings"):
            item["review_required_reason"] = "GATE_WARN"
            mandatory = True
        elif ncc is not None and float(ncc) < ncc_warn_below:
            item["review_required_reason"] = "LOW_NCC"
            mandatory = True

        if mandatory:
            items.append(item)
        elif status == "PASS":
            pass_candidates.append(item)

    rng = random.Random(seed)
    n_pass = max(1, int(round(len(pass_candidates) * pass_sample_fraction))) if pass_candidates else 0
    for item in rng.sample(pass_candidates, k=min(n_pass, len(pass_candidates))):
        item["review_required_reason"] = "RANDOM_PASS_SAMPLE"
        items.append(item)

    pending = sum(1 for i in items if not i.get("visual_qc_signoff"))
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "pet_qc_dir": str(pet_qc_dir),
        "pass_sample_fraction": pass_sample_fraction,
        "total_review_items": len(items),
        "pending_signoff_count": pending,
        "all_signed_off": pending == 0 and len(items) > 0,
        "checklist": [
            "PET aligned to native T1 without gross misregistration",
            "No severe ghosting or truncation in field of view",
            "Foreground uptake pattern biologically plausible for amyloid tracer",
            "Reject if NCC < 0.60 or centiloid delta FAIL without re-preprocess",
        ],
        "items": items,
    }


def apply_visual_qc_signoff(
    pet_qc_path: Path,
    *,
    decision: str,
    reviewer: str = "human_reviewer",
) -> dict[str, Any]:
    payload = _load_pet_qc_payload(pet_qc_path)
    if payload is None:
        raise FileNotFoundError(str(pet_qc_path))
    if decision not in {"pass", "fail", "hold"}:
        raise ValueError("decision must be pass, fail, or hold")
    payload["visual_qc_signoff"] = decision == "pass"
    payload["visual_qc_decision"] = decision
    payload["visual_qc_signed_at"] = _utc_now()
    payload["visual_qc_reviewer"] = reviewer
    atomic_write_json(pet_qc_path, payload)
    return payload
