"""Run the uncommitted fifth fold under a bound post-hoc control-eligibility amendment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.capture_preflight import windows_python_processes
from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2_1.supervise_v21 import INTERNAL_PARENT_ENV, immutable_json
from scripts.mx_fasttrack_v2_4_1.compatible_lock import check_lock_compatible
from scripts.mx_fasttrack_v2_4_2 import entry_v242
from scripts.mx_fasttrack_v2_4_2.supervise_v242 import OUTPUT
from scripts.mx_fasttrack_v2_4_4.control_eligibility import run_with_eligibility
from scripts.mx_fasttrack_v2_4_4.supervise_v244 import AMENDMENT, APPENDIX, ensure_v244_amendment


def require_v244_coordinator(root: Path) -> None:
    claimed = os.environ.get(INTERNAL_PARENT_ENV)
    if claimed is None or not claimed.isdecimal():
        raise RuntimeError("V244_XAI_ACTIVE_COORDINATOR_REQUIRED")
    rows = {int(row.get("ProcessId", -1)): row for row in windows_python_processes(cwd=root)}
    coordinator_pid = int(claimed)
    coordinator = rows.get(coordinator_pid) or {}
    command = str(coordinator.get("CommandLine") or "").lower().replace("/", "\\")
    if "\\scripts\\mx_fasttrack_v2_4_4\\supervise_v244.py" not in command:
        raise RuntimeError("V244_XAI_ACTIVE_COORDINATOR_REQUIRED")
    launcher_pid = os.getppid()
    if launcher_pid == coordinator_pid:
        return
    launcher = rows.get(launcher_pid) or {}
    launcher_command = str(launcher.get("CommandLine") or "").lower().replace("/", "\\")
    if (int(launcher.get("ParentProcessId", -1)) != coordinator_pid
        or "\\scripts\\mx_fasttrack_v2_4_4\\entry_v244.py" not in launcher_command):
        raise RuntimeError("V244_XAI_ACTIVE_COORDINATOR_REQUIRED")


def write_appendix(root: Path, out: Path) -> None:
    from scripts.mx_fasttrack_v2_2 import xai_runner
    from scripts.mx_fasttrack_v2_2.supervise_v22 import ensure_protocol

    reports = out / "reports"
    amendment = ensure_v244_amendment(root, out, create=False)
    xai_runner.check_lock = check_lock_compatible
    summary = xai_runner.verify_xai_summary(root, out, ensure_protocol(root, out))
    fifth = summary["folds"][4]
    amendment_sha = sha(reports / AMENDMENT)
    if (fifth.get("post_hoc_control_eligibility_amendment_sha256") != amendment_sha
        or fifth.get("control_eligibility_policy") != "ATLAS_ONLY_THREE_CONTROLS_WITHIN_1_5X"
        or len(fifth.get("selection_trace", [])) != 2
        or any(direction.get("top_region_id") != trace.get("assessable_top_id")
               for direction, trace in zip(fifth["crossfit_directions"],
                                            fifth["selection_trace"], strict=True))
        or any(trace.get("assessable_top_id") not in amendment["eligible_atlas_ids"]
               or (trace.get("unrestricted_top_id") != trace.get("assessable_top_id")
                   and trace.get("unrestricted_top_id") not in amendment["excluded_atlas_ids"])
               for trace in fifth["selection_trace"])
        or not (reports / "fasttrack_v22_thesis_manifest.json").is_file()):
        raise RuntimeError("V244_FIFTH_FOLD_OR_THESIS_BINDING_INVALID")
    text = (
        "# پیوست اصلاح پسینی XAI برای Method X v2.4.4\n\n"
        "در اجرای نسخهٔ ۲.۲، چهار فولد اول رسید کامل داشتند، اما در فولد پنجم ROI نخستِ "
        "رتبه‌بندی‌شده سه ناحیهٔ کنترل اطلس در بازهٔ حجمی ۱/۱٫۵ تا ۱٫۵ برابر نداشت. "
        "این محدودیت هندسی اطلس و خطای برنامه‌ریزیِ قابلیت آزمون است؛ نتیجهٔ علمیِ مثبت نیست.\n\n"
        "در اصلاح نسخه‌دارِ پس از مشاهدهٔ این توقف، فهرست نواحی واجد سه کنترل فقط از حجم "
        "اطلسِ قفل‌شده محاسبه شد؛ سپس بالاترین ناحیهٔ واجد شرایط روی چهار نمونهٔ انتخاب و "
        "روی چهار نمونهٔ مقابل سنجیده شد. چهار فولد قبلی بدون تغییر ماندند، چون ROIهای "
        "آن‌ها از پیش واجد شرط بودند. این تغییر پسینی است و نتیجهٔ XAI را فقط اکتشافی "
        "می‌گذارد؛ تأیید ناحیهٔ بیماری، اعتبارسنجی مستقل یا آزمون پیش‌ثبت‌شده محسوب نمی‌شود.\n\n"
        f"تعداد ناحیه‌های واجد شرط: {len(amendment['eligible_atlas_ids'])} از "
        f"{len(amendment['eligible_atlas_ids']) + len(amendment['excluded_atlas_ids'])}؛ "
        f"وضعیت گیت XAI: `{summary['status']}`. "
        f"SHA256 اصلاح: `{amendment_sha}`؛ "
        f"SHA256 خلاصهٔ XAI: `{sha(reports / 'xai_v22_summary.json')}`.\n"
    )
    doc = reports / "thesis_method_x_v244_control_eligibility_FA.md"
    from scripts.mx_fasttrack_v2_2.thesis_report import _text_once

    _text_once(doc, text)
    immutable_json(reports / APPENDIX, {
        "schema_version": 1,
        "status": "POST_HOC_XAI_CONTROL_ELIGIBILITY_DISCLOSED",
        "amendment_sha256": amendment_sha,
        "xai_summary_sha256": sha(reports / "xai_v22_summary.json"),
        "v22_thesis_manifest_sha256": sha(reports / "fasttrack_v22_thesis_manifest.json"),
        "old_four_fold_receipt_sha256": amendment["prior_fold_receipt_sha256"],
        "fifth_receipt_sha256": sha(reports / "xai_v22_folds/fold_4.json"),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "post_hoc_internal_exploratory_only": True,
        "external_validation": False,
    })
    print(json.dumps({"status": "V244_POST_HOC_XAI_APPENDIX_COMPLETE",
                      "xai_gate": summary["status"]}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("xai-run", "appendix"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if (root != ROOT.resolve() or args.output_root != OUTPUT
        or out == root or not out.is_relative_to(root)):
        raise RuntimeError("V244_PROJECT_OR_OUTPUT_ROOT_INVALID")
    ensure_v244_amendment(root, out, create=False)
    require_v244_coordinator(root)
    if args.mode == "appendix":
        write_appendix(root, out)
        return 0
    from scripts.mx_fasttrack_v2_2 import xai_runner

    original_run = xai_runner.run_fold_crossfit
    amendment_sha = sha(out / "reports" / AMENDMENT)

    def repaired_fold(**kwargs):
        return run_with_eligibility(original_run=original_run,
                                    amendment_sha256=amendment_sha, **kwargs)

    xai_runner.run_fold_crossfit = repaired_fold
    xai_runner.require_internal_coordinator_parent = require_v244_coordinator
    prior = sys.argv
    try:
        sys.argv = [str(root / "scripts/mx_fasttrack_v2_4_2/entry_v242.py"),
                    "xai-run", "--project-root", str(root),
                    "--output-root", str(args.output_root)]
        return entry_v242.main()
    finally:
        sys.argv = prior
        xai_runner.run_fold_crossfit = original_run


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(3)
