from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.preprocess_quality import classify_mri_numeric_qc, load_json


def load_mri_qc_status(qc_dir: Path, sample_hash: str) -> str | None:
    path = qc_dir / f"{sample_hash}.json"
    payload = load_json(path)
    if not payload:
        return None
    qc = classify_mri_numeric_qc(payload)
    return str(qc.get("qc_status") or "UNKNOWN")


def mri_cache_allowed(
    *,
    sample_hash: str,
    qc_dir: Path | None,
    require_qc_pass: bool,
) -> tuple[bool, str]:
    if not require_qc_pass or qc_dir is None:
        return True, "QC_NOT_REQUIRED"
    status = load_mri_qc_status(qc_dir, sample_hash)
    if status is None:
        return False, "MISSING_QC_JSON"
    if status == "FAIL":
        return False, "MRI_QC_FAIL"
    if status == "HOLD":
        return False, "MRI_QC_HOLD"
    return True, "MRI_QC_PASS"


def pet_cache_allowed(qc_payload: dict[str, Any], *, require_visual_signoff: bool = True) -> tuple[bool, str]:
    if qc_payload.get("status") != "PASS":
        return False, "QC_NOT_PASS"
    gate = qc_payload.get("gate") or {}
    if gate.get("status") == "FAIL":
        return False, "GATE_FAIL"
    if require_visual_signoff and qc_payload.get("visual_qc_required_for_training") and not qc_payload.get("visual_qc_signoff"):
        return False, "VISUAL_QC_UNSIGNED"
    return True, "OK"
