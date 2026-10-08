"""Artifact-bound parent MX-07 terminal gate, including post-outer v2.4.3."""
from __future__ import annotations

from pathlib import Path

from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_science_extension_v1.audit_and_lock import PARENT

STEPS = ("mx7_freeze_scoped_report", "mx7_v22_thesis_addendum")


def require_parent_mx7_terminal(root: Path) -> dict:
    reports = root / PARENT / "reports"
    v244 = reports / "fasttrack_v244_posthoc_xai_atlas_eligibility.json"
    v243 = reports / "fasttrack_v243_postouter_xai_coordinator_repair.json"
    v242 = reports / "fasttrack_v242_checkpoint_projection_before_outer.json"
    v241 = reports / "fasttrack_v241_verifier_repair_before_outer.json"
    state_name = ("fasttrack_v244_supervisor_state.json" if v244.is_file()
                  else "fasttrack_v243_supervisor_state.json" if v243.is_file()
                  else "fasttrack_v242_supervisor_state.json" if v242.is_file()
                  else "fasttrack_v241_supervisor_state.json" if v241.is_file()
                  else "fasttrack_supervisor_state.json")
    state_path = reports / state_name
    if not state_path.is_file():
        raise RuntimeError("EXTENSION_PARENT_MX7_STATE_MISSING")
    steps = read(state_path).get("steps") or {}
    required_steps = (*STEPS, "mx7_v244_control_eligibility_appendix") if v244.is_file() else STEPS
    if any((steps.get(name) or {}).get("status") != "COMPLETED"
           or (steps.get(name) or {}).get("returncode") != 0 for name in required_steps):
        raise RuntimeError("EXTENSION_PARENT_MX7_STEPS_NOT_TERMINAL")
    frozen_path = reports / "fasttrack_thesis_traceability.json"
    manifest_path = reports / "fasttrack_v22_thesis_manifest.json"
    attestation_path = reports / "fasttrack_v21_final_attestation.json"
    if not all(path.is_file() for path in (frozen_path, manifest_path, attestation_path)):
        raise RuntimeError("EXTENSION_PARENT_MX7_FREEZE_ARTIFACT_MISSING")
    frozen, manifest, attestation = map(read, (frozen_path, manifest_path, attestation_path))
    digest = sha(frozen_path)
    if (frozen.get("immutable") is not True
        or manifest.get("parent_freeze_sha256") != digest
        or attestation.get("parent_freeze_sha256") != digest):
        raise RuntimeError("EXTENSION_PARENT_MX7_FREEZE_BINDING_INVALID")
    if v242.is_file():
        amendment = read(v242)
        if amendment.get("status") != "LOCKED_BEFORE_FIRST_OUTER_ACCESS":
            raise RuntimeError("EXTENSION_PARENT_V242_AMENDMENT_INVALID")
    elif v241.is_file():
        amendment = read(v241)
        if amendment.get("status") != "LOCKED_BEFORE_FIRST_OUTER_ACCESS":
            raise RuntimeError("EXTENSION_PARENT_V241_AMENDMENT_INVALID")
    if v243.is_file():
        from scripts.mx_fasttrack_v2_4_3.supervise_v243 import ensure_v243_amendment
        ensure_v243_amendment(root, root / PARENT, create=False)
    if v244.is_file():
        from scripts.mx_fasttrack_v2_4_4.supervise_v244 import APPENDIX, ensure_v244_amendment
        ensure_v244_amendment(root, root / PARENT, create=False)
        appendix_path = reports / APPENDIX
        appendix = read(appendix_path) if appendix_path.is_file() else {}
        doc = reports / "thesis_method_x_v244_control_eligibility_FA.md"
        if (not appendix_path.is_file() or not doc.is_file()
            or appendix.get("text_sha256") != sha(doc)
            or read(appendix_path).get("status") != "POST_HOC_XAI_CONTROL_ELIGIBILITY_DISCLOSED"
            or read(appendix_path).get("amendment_sha256") != sha(v244)
            or read(appendix_path).get("xai_summary_sha256") != sha(reports / "xai_v22_summary.json")
            or read(appendix_path).get("v22_thesis_manifest_sha256") != sha(manifest_path)):
            raise RuntimeError("EXTENSION_PARENT_V244_APPENDIX_INVALID")
    return {"status": "VERIFIED_PARENT_MX7_TERMINAL",
            "journal_sha256": sha(state_path), "freeze_sha256": digest,
            "thesis_manifest_sha256": sha(manifest_path),
            "attestation_sha256": sha(attestation_path),
            "v242_amendment_sha256": sha(v242) if v242.is_file() else None,
            "v243_amendment_sha256": sha(v243) if v243.is_file() else None,
            "v244_amendment_sha256": sha(v244) if v244.is_file() else None,
            "v241_amendment_sha256": sha(v241) if v241.is_file() else None}
