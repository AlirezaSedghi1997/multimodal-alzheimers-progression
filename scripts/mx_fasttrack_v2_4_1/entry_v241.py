"""Explicit in-process entry for v2.4.1 read-gate repair and locked evaluators.

No historical artifact is rewritten.  The adapter changes only the verifier's
in-memory view of an absent aggregate digest, never labels or predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2_4_1.compatible_lock import check_lock_compatible


def _patched_modules() -> None:
    from scripts.mx_fasttrack_v2 import integrity, run
    from scripts.mx_fasttrack_v2_1 import supervise_v21
    from scripts.mx_fasttrack_v2_2 import xai_runner
    from scripts.mx_fasttrack_v2_4.outer_guard_runner import gpu_idle_except_self

    integrity.check_lock = check_lock_compatible
    supervise_v21.check_lock = check_lock_compatible
    xai_runner.check_lock = check_lock_compatible
    run.gpu_idle = gpu_idle_except_self


def _invoke_main(module, arguments: list[str]) -> int:
    previous = sys.argv
    try:
        sys.argv = [getattr(module, "__file__", module.__name__), *arguments]
        return int(module.main())
    finally:
        sys.argv = previous


def freeze_guard_compatible(root: Path, out: Path, output_relative: Path) -> int:
    """v2.1 freeze sequence with its report verifier called in this process."""
    from scripts.mx_fasttrack_v2 import freeze_report
    from scripts.mx_fasttrack_v2_1 import supervise_v21 as v21
    from scripts.mx_fasttrack_v2.run import read, sha

    reason = "OPERATOR_CHOICE"  # v2.2 runs its own five-fold XAI, omitting v2.1 XAI.
    v21.ensure_amendment(root, out)
    reports = out / "reports"
    xai_state = reports / "xai_limited_state.json"
    xai_summary = reports / "xai_limited_summary.json"
    context_path = reports / v21.CONTEXT_NAME
    fold_binding_path = reports / v21.FOLD_BINDING_NAME
    waiver_path = reports / v21.WAIVER_NAME
    alias = reports / "xai_limited_summary_state.json"
    expected_waiver = v21.xai_omission_payload(root, out, reason)
    if not waiver_path.is_file() or read(waiver_path) != expected_waiver or alias.exists():
        raise RuntimeError("V241_V21_XAI_OMISSION_RECEIPT_INVALID")
    temporal_payload = v21.temporal_appendix_payload(out)
    freeze_report.check_lock = check_lock_compatible
    code = _invoke_main(freeze_report, ["--project-root", str(root),
                                         "--output-root", str(output_relative)])
    if code:
        return code
    final = reports / "fasttrack_thesis_traceability.json"
    if not final.is_file():
        raise RuntimeError("V241_PARENT_FREEZE_NOT_SCOPED_READY")
    frozen = v21.verify_frozen_scope(root, out, xai_skipped=True)
    if not waiver_path.is_file() or read(waiver_path) != v21.xai_omission_payload(root, out, reason):
        raise RuntimeError("V241_XAI_OMISSION_DRIFT_DURING_FREEZE")
    if v21.temporal_appendix_payload(out) != temporal_payload:
        raise RuntimeError("V241_TEMPORAL_EVIDENCE_DRIFT_DURING_FREEZE")
    temporal_appendix = reports / v21.TEMPORAL_APPENDIX_NAME
    v21.immutable_json(temporal_appendix, temporal_payload)
    attestation = {
        "schema_version": 1,
        "status": "READY_SCOPED_FASTTRACK_V21_AMENDMENT",
        "amendment_sha256": sha(reports / v21.AMENDMENT_NAME),
        "parent_freeze_sha256": sha(final),
        "xai_context_sha256": sha(context_path) if context_path.exists() else None,
        "xai_fold_binding_sha256": sha(fold_binding_path) if fold_binding_path.exists() else None,
        "xai_canonical_state_sha256": sha(xai_state) if xai_state.exists() else None,
        "xai_state_alias_sha256": sha(alias) if alias.exists() else None,
        "xai_explicit_omission_sha256": sha(waiver_path) if waiver_path.exists() else None,
        "temporal_appendix_sha256": sha(temporal_appendix),
        "xai_limited_status": frozen["status"]["MX06_limited_optional"],
        "xai_region_claim_allowed": False,
        "xai_occlusion_faithfulness_claim_allowed": False,
        "xai_occlusion_limitation": (
            "POSITIVE_CLASS_MEAN_OVER_MIXED_LABELS_NOT_A_CLASS_TARGETED_PAIRED_GATE"
        ),
        "original_MX05_MX06_MX07": "NOT_DONE_BY_FASTTRACK",
    }
    v21.immutable_json(reports / "fasttrack_v21_final_attestation.json", attestation)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("hybrid-pair", "locked", "xai-audit", "xai-run",
                                         "freeze-guard", "thesis"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=("selected", "mx04_scratch", "mx04_head_last"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root) or out == root:
        raise RuntimeError("V241_PROJECT_OR_OUTPUT_ROOT_INVALID")
    from scripts.mx_fasttrack_v2_4_1.supervise_v241 import ensure_v241_amendment
    ensure_v241_amendment(root, out, create=False)
    _patched_modules()
    if args.mode == "hybrid-pair":
        if args.task is not None:
            raise RuntimeError("V241_HYBRID_PAIR_TASK_FORBIDDEN")
        from scripts.mx_fasttrack_v2 import evaluate_hybrid_pair
        evaluate_hybrid_pair.check_lock = check_lock_compatible
        return _invoke_main(evaluate_hybrid_pair, ["--project-root", str(root),
                                                   "--output-root", str(args.output_root)])
    if args.mode == "locked":
        if args.task is None:
            raise RuntimeError("V241_LOCKED_TASK_REQUIRED")
        from scripts.mx_fasttrack_v2 import evaluate_locked
        return _invoke_main(evaluate_locked, ["--project-root", str(root),
                                              "--output-root", str(args.output_root),
                                              "--scope", "outer", "--task", args.task])
    if args.task is not None:
        raise RuntimeError("V241_TASK_ONLY_FOR_LOCKED_OUTER")
    if args.mode in {"xai-audit", "xai-run"}:
        from scripts.mx_fasttrack_v2_2 import xai_runner
        return _invoke_main(xai_runner, ["--project-root", str(root),
                                         "--output-root", str(args.output_root),
                                         "--mode", "audit" if args.mode == "xai-audit" else "run"])
    if args.mode == "freeze-guard":
        return freeze_guard_compatible(root, out, args.output_root)
    from scripts.mx_fasttrack_v2_2 import thesis_report, xai_runner
    xai_runner.check_lock = check_lock_compatible
    return _invoke_main(thesis_report, ["--project-root", str(root),
                                        "--output-root", str(args.output_root)])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
