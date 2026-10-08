"""Artifact-bound MX-05 add-on analysis, XAI audit, and Persian MX-07 addendum."""
from __future__ import annotations

import math
from pathlib import Path

from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2_1.supervise_v21 import immutable_json
from scripts.mx_fasttrack_v2_2.thesis_report import _text_once
from scripts.mx_fasttrack_v2_4_1.compatible_lock import check_lock_compatible
from scripts.mx_science_extension_v1.analysis import analyze_panel, validate_panel
from scripts.mx_science_extension_v1.audit_and_lock import CONFIG, OUTPUT, PARENT
from scripts.mx_science_extension_v1.execution import LOCK, ensure_execution_lock
from scripts.mx_science_extension_v1.outer import (
    EXPECTED_N, OUTER_LOCK, PANEL, _private, ensure_outer_lock,
)
from scripts.mx_science_extension_v1.selection import NAME as SELECTION_NAME
from scripts.mx_science_extension_v1.terminal import require_parent_mx7_terminal

ANALYSIS = "extension_paired_analysis.v1.json"
XAI_AUDIT = "extension_xai_receipt_audit.v1.json"
MANIFEST = "extension_mx7_manifest.v1.json"
ADDENDUM = "extension_mx7_results_limitations_FA.v1.md"


def run_analysis(root: Path) -> dict:
    ensure_execution_lock(root, create=False)
    ensure_outer_lock(root, create=False)
    panel_path = _private(root) / PANEL
    private = read(panel_path)
    if (private.get("status") != "COMPLETED_PRIVATE_PATIENT_PANEL"
        or private.get("outer_lock_sha256") !=
           sha(root / OUTPUT / "reports" / OUTER_LOCK)
        or private.get("patient_count") != EXPECTED_N):
        raise RuntimeError("EXTENSION_PRIVATE_PANEL_UNBOUND")
    validate_panel(private["rows"], expected_n=EXPECTED_N)
    config = read(root / CONFIG)
    result = analyze_panel(
        private["rows"], expected_n=EXPECTED_N,
        resamples=config["analysis"]["paired_patient_bootstrap_resamples"],
        bootstrap_seed=config["analysis"]["bootstrap_seed"])
    result.update({
        "private_panel_sha256": sha(panel_path),
        "outer_lock_sha256": sha(root / OUTPUT / "reports" / OUTER_LOCK),
        "execution_preregistration_sha256": sha(root / OUTPUT / "reports" / LOCK),
        "pseudo_holdout_used": False,
        "internal_outer_historically_inspected": True,
        "multiple_secondary_comparisons_confirmatory": False,
    })
    path = root / OUTPUT / "reports" / ANALYSIS
    immutable_json(path, result)
    return {"status": result["status"], "analysis_sha256": sha(path)}


def xai_audit_payload(root: Path) -> dict:
    """Revalidate all five source receipts when available; preserve XAI gaps."""
    terminal = require_parent_mx7_terminal(root)
    parent = root / PARENT
    reports = parent / "reports"
    state_path = reports / ("fasttrack_v244_supervisor_state.json"
                            if (reports / "fasttrack_v244_posthoc_xai_atlas_eligibility.json").is_file()
                            else "fasttrack_v243_supervisor_state.json"
                            if (reports / "fasttrack_v243_postouter_xai_coordinator_repair.json").is_file()
                            else "fasttrack_v242_supervisor_state.json")
    steps = read(state_path).get("steps") or {}
    summary_path = reports / "xai_v22_summary.json"
    if summary_path.is_file():
        from scripts.mx_fasttrack_v2_2 import xai_runner
        from scripts.mx_fasttrack_v2_2.supervise_v22 import ensure_protocol

        # The v2.4.1 amendment repairs only the verifier's missing digest view.
        prior_check = xai_runner.check_lock
        xai_runner.check_lock = check_lock_compatible
        try:
            protocol = ensure_protocol(root, parent)
            summary = xai_runner.verify_xai_summary(root, parent, protocol)
        finally:
            xai_runner.check_lock = prior_check
        if terminal.get("v244_amendment_sha256") is not None:
            from scripts.mx_fasttrack_v2_4_4.supervise_v244 import ensure_v244_amendment
            amendment = ensure_v244_amendment(root, parent, create=False)
            if (summary["folds"][4].get("post_hoc_control_eligibility_amendment_sha256")
                != terminal["v244_amendment_sha256"]
                or any(sha(reports / "xai_v22_folds" / f"fold_{i}.json") != digest
                       for i, digest in amendment["prior_fold_receipt_sha256"].items())):
                raise RuntimeError("EXTENSION_XAI_V244_FOLD_PROVENANCE_INVALID")
        status = summary["status"]
        evidence = {"summary_sha256": sha(summary_path),
                    "context_sha256": summary["context_sha256"],
                    "fold_receipt_sha256": summary["fold_receipt_sha256"],
                    "n_images": summary["n_images"],
                    "atlas_spm_provenance": read(reports / "xai_v22_context.json")
                    ["atlas_spm_provenance"],
                    "checkpoint_sha256": read(reports / "xai_v22_context.json")
                    ["checkpoint_sha256"],
                    "randomization_seeds": [13, 29, 47],
                    "paired_crossfit_occlusion": True}
    else:
        if (steps.get("mx6_v22_xai_crossfit") or {}).get("status") == "COMPLETED":
            raise RuntimeError("EXTENSION_XAI_MARKED_COMPLETE_BUT_SUMMARY_MISSING")
        raise RuntimeError("EXTENSION_XAI_FIVE_FOLD_SUMMARY_NOT_DONE_OR_BLOCKED")
    return {
        "schema_version": 1, "status": status,
        "parent_freeze_sha256": terminal["freeze_sha256"],
        "parent_v242_amendment_sha256": terminal["v242_amendment_sha256"],
        "parent_v243_amendment_sha256": terminal.get("v243_amendment_sha256"),
        "parent_v244_amendment_sha256": terminal.get("v244_amendment_sha256"),
        "evidence": evidence,
        "independent_region_confirmation": False,
        "external_validation": False,
        "claim_scope": "INTERNAL_EXPLORATORY_MODEL_ATTRIBUTION_ONLY",
    }


def run_xai_audit(root: Path) -> dict:
    ensure_execution_lock(root, create=False)
    payload = xai_audit_payload(root)
    path = root / OUTPUT / "reports" / XAI_AUDIT
    immutable_json(path, payload)
    return {"status": payload["status"], "xai_audit_sha256": sha(path)}


def _number(value: object) -> str:
    return f"{value:.4f}" if type(value) in (int, float) and math.isfinite(value) else "غیرقابل‌برآورد"


def _interval(row: dict) -> str:
    interval = row.get("patient_bootstrap_ci95")
    return (f"[{_number(interval[0])}، {_number(interval[1])}]"
            if isinstance(interval, list) and len(interval) == 2 else "غیرقابل‌برآورد")


def _negative(result: dict) -> list[str]:
    issues = []
    all_rows = result["subsets"]["all"]
    for key, comparison in all_rows["comparisons"].items():
        for metric in ("auroc", "auprc", "brier"):
            row = comparison["metrics"][metric]
            effect = row["selected_minus_comparator"]
            interval = row["patient_bootstrap_ci95"]
            if effect is None:
                issues.append(f"{key}، {metric}: برآورد نشد")
            elif (metric == "brier" and effect >= 0) or (metric != "brier" and effect <= 0):
                issues.append(f"{key}، {metric}: جهت اثر به نفع مدل منتخب نیست ({_number(effect)})")
            elif interval is None or interval[0] <= 0 <= interval[1]:
                issues.append(f"{key}، {metric}: بازهٔ عدم‌قطعیت صفر را در بر می‌گیرد")
    return issues


def render_addendum(root: Path, analysis: dict, xai: dict) -> str:
    audit = read(root / OUTPUT / "reports/day1_patient_provenance_audit.json")
    lines = [
        "# پیوست نتایج و محدودیت‌ها ــ Method X / افزونهٔ MX-05 v1",
        "",
        "این پیوست به MX7 fast-track افزوده می‌شود و جایگزین گزارش اصلی آن نیست. "
        "تمام برآوردها روی outer foldهای داخلیِ تاریخاً بررسی‌شده‌اند؛ "
        "تحلیل تأییدی، آزمون بیرونی مستقل یا تأیید ناحیهٔ بیماری نیستند.",
        "",
        "## روش و جریان داده",
        "",
        "پنج فولد سطح بیمار، انتخاب Hybrid3D جداگانه در هر فولد و بذر فقط از "
        "کمینهٔ BCE در inner-validation، و بذرهای ۱۳/۲۹/۴۷. "
        "بذرهای ۲۹ و ۴۷ هر دو گونهٔ Hybrid را تکرار کردند؛ MRI-only از صفر، "
        "کنترل برچسب تصادفی فقط در fit، شیوع پایه و لجستیک فقط-جدولی اجرا شدند. "
        "گزینش C و آماده‌سازی جدولی فقط بر fit/inner انجام شد. "
        "بذرهای تکراری بیمار مستقل محسوب نشده‌اند. PET ورودی مدل نبود.",
        "",
        f"تعداد بیماران outer با QC: {analysis['all_patient_count']}؛ "
        f"قفل طراحی: `{sha(root / OUTPUT / 'reports/design_lock.json')}`؛ "
        f"قفل outer: `{analysis['outer_lock_sha256']}`.",
        "",
        "## عملکرد مطلق بازوها در کل outer داخلی",
        "",
        "برای بازوهای چندبذری، هر معیار ابتدا در هر بذر روی همان بیماران "
        "محاسبه و سپس میانگین شده است؛ این جدول تعداد بیمار را چندبرابر نمی‌کند.",
        "",
        "| بازو | بذرها | AUROC | AUPRC | Brier | intercept | slope |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, item in analysis["subsets"]["all"]["arm_metrics"].items():
        m = item["metrics"]
        lines.append(
            f"| {arm} | {','.join(str(v) for v in item['seed_set'])} | "
            f"{_number(m['auroc'])} | {_number(m['auprc'])} | {_number(m['brier'])} | "
            f"{_number(m['calibration_intercept'])} | {_number(m['calibration_slope'])} |"
        )
    lines += [
        "",
        "## نتایج جفتی در سطح بیمار",
        "",
        "علامت اثر، مدل منتخب منهای کنترل است؛ برای AUROC/AUPRC مقدار مثبت "
        "و برای Brier مقدار منفی بهتر است. CI با bootstrap جفتی بیمار است؛ "
        "مقایسه‌های ثانویه و پنجره‌های زمانی اکتشافی‌اند.",
        "",
        "| بازه PET–MRI | n (مثبت/منفی) | مقایسه | AUROC Δ [CI 95٪] | AUPRC Δ [CI 95٪] | Brier Δ [CI 95٪] | کالیبراسیون intercept/slope Δ |",
        "|---|---:|---|---|---|---|---|",
    ]
    for window in ("all", "90", "180", "365"):
        block = analysis["subsets"][window]
        counts = f"{block['n']} ({block['positive_n']}/{block['negative_n']})"
        if block["status"] == "NOT_ESTIMABLE_TWO_CLASSES_REQUIRED":
            lines.append(f"| {window} | {counts} | همه | غیرقابل‌برآورد | غیرقابل‌برآورد | غیرقابل‌برآورد | غیرقابل‌برآورد |")
            continue
        for key, item in block["comparisons"].items():
            metrics = item["metrics"]
            cells = [f"{_number(metrics[m]['selected_minus_comparator'])} {_interval(metrics[m])}"
                     for m in ("auroc", "auprc", "brier")]
            calibration = ", ".join(
                f"{m}: {_number(metrics[m]['selected_minus_comparator'])} {_interval(metrics[m])}"
                for m in ("calibration_intercept", "calibration_slope"))
            lines.append(f"| {window} | {counts} | {key} | {' | '.join(cells)} | {calibration} |")
    lines += ["", "## نتایج منفی و آزمون‌های حساسیت", ""]
    negatives = _negative(analysis)
    lines.extend(f"- {item}" for item in negatives) if negatives else lines.append(
        "- در جهت اثرهای نقطه‌ایِ اصلی مورد منفی ثبت نشد؛ این عدم قطعیت یا سوگیری را رفع نمی‌کند.")
    lines += ["", f"XAI: `{xai['status']}`؛ "
              "رسیدهای پنج فولد، تصادفی‌سازی وزن، atlas/SPM و هش checkpoint فقط در صورت وجود "
              "و صحت رسید تجمیعی پذیرفته شدند. نتیجهٔ XAI صرفاً توضیح مدل است.",
              "در فولد پنجم، پس از آشکارشدن ناکافی‌بودن کنترل‌های هم‌اندازه برای ROI اول، "
              "انتخاب ROI به نواحی واجد سه کنترل محدود شد؛ این تغییر نسخهٔ ۲.۴.۴ پسینی است "
              "و در پیوست هش‌دار مستقل ثبت شده، نه در پروتکل پیش‌ثبت‌شدهٔ اولیه.",
              "", "## جدول خطر سوگیری و گزارش‌دهی هم‌راستا با TRIPOD+AI", "",
              "| حوزه | وضعیت/شاهد | اثر بر ادعا |", "|---|---|---|",
              f"| انتخاب نمونه و جریان بیماران | {analysis['all_patient_count']} بیمار QC؛ پنج outer fold؛ شبه‌هولدآوت در این افزونه استفاده نشد | فقط داخلی |",
              "| منشأ برچسب و cohort | A4 فقط کلاس ۱، LEARN فقط کلاس ۰ | تفکیک اثر بیماری از cohort ممکن نیست |",
              f"| MRI/PET و QC | هش T1 در split موجود؛ منشأ فایل PET={audit['pet_image_provenance']}؛ site/scanner={audit['scanner_overlap']} | عدم‌قطعیت منشأ/انتقال‌پذیری |",
              f"| سابقهٔ شبه‌هولدآوت | {audit['pseudo_holdout_prior_access']} | ادعای آزمون دست‌نخورده ممنوع |",
              "| انتخاب مدل و دادهٔ ارزیابی | انتخاب هر fold فقط inner؛ outer تاریخی قبلاً دیده شده | برآوردها post-hoc اکتشافی |",
              "| دادهٔ گمشده/ویژگی‌ها | ترتیب فیزیکی audit شد؛ imputer/scaler جدولی فقط fit | خطر دادهٔ گمشده باقی است |",
              "| عدم‌قطعیت و تکرار | bootstrap جفتی بیمار؛ بذر مستقل از بیمار نیست | CI توصیفی، نه اثبات برتری |",
              "| XAI و مکان‌یابی | cross-fit در inner، atlas/SPM و randomization در رسیدهای موجود | ناحیهٔ بیماری تأیید نشده |",
              "| اعتبارسنجی بیرونی | دادهٔ بیرونی موجود نیست | تعمیم/کاربرد بالینی قابل ادعا نیست |",
              "", "در صورت دسترسی بعدی به cohort مستقلِ دارای هر دو برچسب، site/scanner و مرجع ناحیه‌ای کورسازی‌شده، "
              "پروتکل تأییدی تازه باید پیش از بازکردن داده ثبت شود. هیچ بازتحلیل همین outer foldها جای آن را نمی‌گیرد.", ""]
    return "\n".join(lines)


def run_report(root: Path) -> dict:
    terminal = require_parent_mx7_terminal(root)
    ensure_execution_lock(root, create=False)
    ensure_outer_lock(root, create=False)
    reports = root / OUTPUT / "reports"
    analysis_path, xai_path = reports / ANALYSIS, reports / XAI_AUDIT
    analysis, xai = read(analysis_path), read(xai_path)
    if (analysis.get("status") != "COMPLETED_INTERNAL_EXPLORATORY"
        or analysis.get("all_patient_count") != EXPECTED_N
        or analysis.get("private_panel_sha256") != sha(_private(root) / PANEL)
        or xai != xai_audit_payload(root)):
        raise RuntimeError("EXTENSION_MX7_INPUT_EVIDENCE_DRIFT")
    text = render_addendum(root, analysis, xai)
    text_path = reports / ADDENDUM
    _text_once(text_path, text)
    manifest = {
        "schema_version": 1, "status": "COMPLETED_POST_HOC_INTERNAL_EXPLORATORY",
        "parent_freeze_sha256": terminal["freeze_sha256"],
        "parent_thesis_manifest_sha256": terminal["thesis_manifest_sha256"],
        "execution_preregistration_sha256": sha(reports / LOCK),
        "fold_local_selection_sha256": sha(reports / SELECTION_NAME),
        "outer_lock_sha256": sha(reports / OUTER_LOCK),
        "private_panel_sha256": sha(_private(root) / PANEL),
        "paired_analysis_sha256": sha(analysis_path),
        "xai_receipt_audit_sha256": sha(xai_path),
        "persian_results_limitations_sha256": sha(text_path),
        "negative_result_count": len(_negative(analysis)),
        "independent_validation": False,
        "disease_region_confirmation": False,
        "clinical_use_claim_allowed": False,
    }
    immutable_json(reports / MANIFEST, manifest)
    return {"status": manifest["status"], "manifest_sha256": sha(reports / MANIFEST)}
