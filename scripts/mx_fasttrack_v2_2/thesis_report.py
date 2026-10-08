"""Evidence-bound Persian thesis addendum for the scoped fast-track MX7 freeze.

This does not turn the historical fold tests into independent validation, nor
does it certify the original full MX7 research program.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2_1.supervise_v21 import immutable_json


def validate_freeze(frozen: dict) -> None:
    if (frozen.get("scope") != "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
        or frozen.get("immutable") is not True
        or (frozen.get("status") or {}).get("MX07_fasttrack") != "READY_SCOPED_FASTTRACK"
        or frozen.get("external_validation") is not False
        or frozen.get("cohort_label_perfectly_confounding") is not True
        or not isinstance(frozen.get("paper_numbers"), dict)):
        raise ValueError("V22_FROZEN_PARENT_NOT_SCOPED_READY")


def _metric(row: dict, key: str) -> str:
    value = row.get(key)
    if value is None:
        return ("غیرقابل‌محاسبه" if row.get("status") == "NOT_MEASURABLE_ONE_CLASS"
                else "گزارش‌نشده")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"V22_NONFINITE_OR_NONNUMERIC_METRIC:{key}")
    return f"{value:.4f}"


def build_sections(frozen: dict, xai: dict | None, *, external_status: str,
                   pair_by_fold: list[dict] | None = None) -> dict[str, str]:
    validate_freeze(frozen)
    if not external_status.startswith(("NOT_DONE", "BLOCKED")):
        raise ValueError("V22_EXTERNAL_CLAIM_REQUIRES_SEPARATE_VALIDATED_PROTOCOL")
    if xai is not None:
        if (xai.get("status") not in {"COMPLETED_EXPLORATORY_CROSSFIT",
                                       "BLOCKED_EXPLORATORY_GATE"}
            or xai.get("n_images") != 40
            or xai.get("not_independent_testing") is not True
            or xai.get("anatomical_or_causal_region_claim_allowed") is not False
            or [row.get("fold") for row in xai.get("folds", [])] != list(range(5))
            or any(row.get("sample_count") != 8 for row in xai["folds"])):
            raise ValueError("V22_XAI_FIVE_FOLDS_OR_SCOPE_INVALID")
    numbers = frozen["paper_numbers"]
    selected = str(numbers.get("selected_inner_task", "گزارش‌نشده"))
    arms = [("MX4 scratch", numbers.get("mx04_scratch_overall")),
            ("MX4 head/last", numbers.get("mx04_head_overall")),
            ("Hybrid baseline", numbers.get("hybrid_baseline_overall")),
            ("Hybrid dense skip", numbers.get("hybrid_dense_overall"))]
    if any(not isinstance(row, dict) for _, row in arms):
        raise ValueError("V22_ALL_FOUR_ARMS_REQUIRED")
    methods = (
        "# فصل روش‌ها ــ Method X fast-track v2.2\n\n"
        "این متن فقط بازنمایی شواهد داخلیِ نسخهٔ کوتاه‌شده است؛ نسخهٔ کامل MX4–MX7 را جایگزین نمی‌کند. "
        "برچسب آمیلوئید از PET گرفته شد، اما PET ورودی مدل Method X نبود. "
        "تقسیم‌بندی پنج‌فولدی در سطح بیمار و seed=13 بود؛ هر فولد سقف ۲۰ epoch داشت. "
        "چهار بازوی ثبت‌شده شامل ResNet34 از صفر، fine-tuning سر و لایهٔ آخر ResNet34، "
        "Hybrid3D پایه و Hybrid3D با dense skip بودند. نرمال‌سازی tabular درون فولد انجام شد. "
        "انتخاب نهایی فقط با میانگین loss در inner validation انجام شد؛ بازوی fine-tuning به‌علت "
        "مواجههٔ پیشین با برچسب‌های supervised در انتخاب معتبر نبود. ارزیابی outer روی fold testهای "
        "تاریخاً دیده‌شده، توصیفی است و مقایسهٔ دو Hybrid با bootstrap جفتی در سطح بیمار گزارش می‌شود.\n\n"
        "## XAI نسخه‌دار\n\n"
        "در هر فولد نخستین هشت MRI واقعیِ inner validation با digest تقسیم و checkpoint قفل‌شده "
        "انتخاب می‌شوند. در طرح دوطرفهٔ ۴/۴، ناحیهٔ atlas از چهار تصویر انتخاب و فقط روی چهار تصویر "
        "دیگر همان جهت آزموده می‌شود؛ جهت معکوس نیز اجرا می‌شود. برای هر تصویر، کلاس هدف از پیش "
        "با پیش‌بینی تصویر دست‌نخورده تعیین و افت احتمال همان کلاس پس از پوشاندن ROI با میانهٔ "
        "سه ROI اطلسِ هم‌اندازه مقایسه می‌شود. سه randomization وزن با seedهای ۱۳، ۲۹ و ۴۷ "
        "ثبت می‌شود؛ واحد bootstrap تجمیعی فولد است. پوشاندن با صفر می‌تواند خارج از توزیع باشد. "
        "این نمونه‌ها قبلاً در انتخاب مدل دخیل بوده‌اند، بنابراین این آزمون مستقل نیست.\n")
    table = ["| بازو | ROC-AUC | AP | Brier |", "|---|---:|---:|---:|"]
    for name, row in arms:
        table.append(f"| {name} | {_metric(row, 'roc_auc')} | "
                     f"{_metric(row, 'average_precision')} | {_metric(row, 'brier')} |")
    delta = (numbers.get("hybrid_paired_difference") or {}).get("roc_auc") or {}
    if not isinstance(delta, dict):
        raise ValueError("V22_PAIRED_DELTA_MISSING")
    difference = delta.get("dense_minus_baseline")
    if type(difference) not in (float, int) or not math.isfinite(difference):
        raise ValueError("V22_PAIRED_DELTA_INVALID")
    baseline_auc = arms[2][1].get("roc_auc")
    dense_auc = arms[3][1].get("roc_auc")
    if (type(baseline_auc) in (float, int) and type(dense_auc) in (float, int)
        and not math.isclose(difference, dense_auc - baseline_auc,
                             rel_tol=1e-8, abs_tol=1e-9)):
        raise ValueError("V22_PAIRED_EFFECT_SIGN_MISMATCH")
    results = ("# فصل نتایج ــ Method X fast-track v2.2\n\n"
               "تمام چهار بازو، حتی اگر نتیجهٔ نامطلوب داشته باشند، گزارش می‌شوند. "
               "این مقادیر از fold testهای داخلیِ تاریخاً دیده‌شده‌اند، نه مجموعهٔ آزمون مستقل.\n\n"
               + "\n".join(table) + "\n\n"
               f"بازوی برگزیده با inner loss: `{selected}`. تفاوت ROC-AUC dense منهای baseline: "
               f"{difference:.4f}. ")
    interval = delta.get("patient_bootstrap_ci95")
    if (isinstance(interval, list) and len(interval) == 2
        and all(type(v) in (float, int) and math.isfinite(v) for v in interval)):
        results += f"CI 95٪ bootstrap بیمار: [{interval[0]:.4f}, {interval[1]:.4f}].\n\n"
    else:
        results += "CI 95٪: در شواهد ورودی حاضر نیست.\n\n"
    inner = numbers.get("inner_mean_best_validation_loss")
    if isinstance(inner, dict) and inner:
        results += "میانگین بهترین loss در inner validation برای همهٔ بازوها:\n\n"
        results += "| بازو | mean best validation loss |\n|---|---:|\n"
        for task, value in sorted(inner.items()):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"V22_INNER_LOSS_INVALID:{task}")
            results += f"| {task} | {value:.4f} |\n"
        results += "\n"
    if pair_by_fold is not None:
        if [row.get("fold") for row in pair_by_fold] != list(range(5)):
            raise ValueError("V22_HYBRID_PER_FOLD_INCOMPLETE")
        results += ("| فولد | n | AUC پایه | AUC dense | AP پایه | AP dense | Brier پایه | Brier dense |\n"
                    "|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in pair_by_fold:
            results += (f"| {row['fold']} | {row['n']} | "
                        f"{_metric(row['baseline'], 'roc_auc')} | "
                        f"{_metric(row['dense'], 'roc_auc')} | "
                        f"{_metric(row['baseline'], 'average_precision')} | "
                        f"{_metric(row['dense'], 'average_precision')} | "
                        f"{_metric(row['baseline'], 'brier')} | "
                        f"{_metric(row['dense'], 'brier')} |\n")
        results += "\n"
    else:
        results += "جدول per-fold در این ورودی حاضر نیست؛ از اعداد تجمیعی بازسازی نمی‌شود.\n\n"
    results += ("برای MX4، ارزیابی outer نمرهٔ تجمیعی دارد؛ معیارهای per-fold آن در این "
                "بستهٔ گزارش نهایی موجود نیست و از aggregate بازسازی نمی‌شود.\n\n")
    temporal = numbers.get("temporal_gap_counts")
    if isinstance(temporal, dict):
        results += "شمارش حساسیت پنجرهٔ PET–MRI، صرفاً توصیفی:\n\n"
        results += "| فاصلهٔ حداکثر | n | مثبت | منفی |\n|---:|---:|---:|---:|\n"
        for days in ("90", "180", "365"):
            item = temporal.get(days)
            if not isinstance(item, dict) or any(type(item.get(key)) is not int
                                                  for key in ("n", "positive", "negative")):
                raise ValueError(f"V22_TEMPORAL_COUNT_INVALID:{days}")
            results += f"| {days} روز | {item['n']} | {item['positive']} | {item['negative']} |\n"
        results += "\n"
    if xai is None:
        results += "XAI نسخهٔ ۲.۲: NOT_DONE؛ هیچ نتیجهٔ ناحیه‌ای گزارش نمی‌شود.\n\n"
    else:
        effect = xai.get("fold_mean_paired_effect")
        if type(effect) not in (float, int) or not math.isfinite(effect):
            # The caller may be rendering a minimal shape in a focused test;
            # the real writer must always provide the aggregate value.
            effect_text = "گزارش‌نشده"
        else:
            effect_text = f"{effect:.4f}"
        top_drops = [row.get("paired_mean_top_target_drop") for row in xai["folds"]]
        top_text = (f"{sum(top_drops) / 5:.4f}" if all(
            type(value) in (int, float) and math.isfinite(value) for value in top_drops)
            else "گزارش‌نشده")
        results += (f"XAI ۲.۲: {xai['status']}؛ ۴۰ تصویر در پنج فولد؛ "
                    f"میانگین فولدی افت احتمال هدف برای ROI منتخب: {top_text}؛ "
                    f"میانگین فولدی اختلاف افت هدف نسبت به کنترل: {effect_text}. "
                    "این شاخص صرفاً وفاداریِ اکتشافی به مدل است.\n\n")
        if all(isinstance(row.get("crossfit_directions"), list)
               and len(row["crossfit_directions"]) == 2 for row in xai["folds"]):
            results += ("| فولد | ROI منتخب جهت اول/دوم (atlas ID) | افت هدف | "
                        "اختلاف جفتی با کنترل | gate |\n"
                        "|---:|---|---:|---:|---|\n")
            for row in xai["folds"]:
                region_ids = "/".join(str(item["top_region_id"])
                                      for item in row["crossfit_directions"])
                results += (f"| {row['fold']} | {region_ids} | "
                            f"{_metric(row, 'paired_mean_top_target_drop')} | "
                            f"{_metric(row, 'paired_mean_effect')} | "
                            f"{row.get('gate_status', 'گزارش‌نشده')} |\n")
            results += "\n"
        interval = (xai.get("aggregate") or {}).get("fold_bootstrap_ci95")
        if (isinstance(interval, list) and len(interval) == 2
            and all(type(value) in (int, float) and math.isfinite(value) for value in interval)):
            results += (f"CI 95٪ bootstrap در سطح پنج فولد برای اختلاف جفتی: "
                        f"[{interval[0]:.4f}, {interval[1]:.4f}].\n\n")
    results += (f"وضعیت pseudo-holdout در freeze والد: "
                f"`{(frozen.get('status') or {}).get('pseudo_random_holdout', 'UNKNOWN')}`.\n\n")
    results += f"اعتبارسنجی مستقل: `{external_status}`.\n"
    discussion = (
        "# فصل بحث و محدودیت‌ها ــ Method X fast-track v2.2\n\n"
        "برچسب و cohort در دادهٔ فعلی کاملاً هم‌بسته‌اند؛ بنابراین تفکیک سیگنال بیماری "
        "از تفاوت cohort با این داده ممکن نیست و AUC درون هر cohort با یک کلاس قابل محاسبه نیست. "
        "fold testهای تاریخی نیز آزمون خارجی یا تأیید مستقل نیستند. "
        "انتخاب ناحیه در نیمهٔ جدا از تصاویرِ ارزیابیِ occlusion انجام می‌شود، ولی کل تصاویر "
        "inner-validation در انتخاب مدل دخیل بوده‌اند و masked image ممکن است خارج از توزیع باشد. "
        "نقشهٔ attribution، اثبات ناحیهٔ بیماری، علیت یا تشخیص بالینی نیست. "
        "برای اعتبارسنجی مستقل، cohort تازه با همپوشانی واقعی برچسب درون cohort، ثبت scanner/site، "
        "endpoint از پیش ثبت‌شده و مدل کاملاً قفل‌شده لازم است. ادعای مکان‌یابی بیماری علاوه بر آن "
        "به مرجع ناحیه‌ای مستقل و کورسازی‌شده نیاز دارد. تا آن زمان این ادعاها مجاز نیستند.\n")
    return {"methods": methods, "results": results, "discussion": discussion}


def _text_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"V22_IMMUTABLE_THESIS_TEXT_CHANGED:{path}")
        return
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.new")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if path.read_bytes() != encoded:
                raise RuntimeError(f"V22_IMMUTABLE_THESIS_TEXT_CHANGED:{path}")
            return
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/final_results/method_x_fasttrack_v2_20260923"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root) or out == root:
        raise RuntimeError("V22_PROJECT_OR_OUTPUT_INVALID")
    from scripts.mx_fasttrack_v2_2.xai_runner import verify_xai_summary
    from scripts.mx_fasttrack_v2_2.supervise_v22 import ensure_protocol

    protocol = ensure_protocol(root, out)
    reports = out / "reports"
    parent_path = reports / "fasttrack_thesis_traceability.json"
    attestation_path = reports / "fasttrack_v21_final_attestation.json"
    frozen = read(parent_path)
    validate_freeze(frozen)
    attestation = read(attestation_path)
    if attestation.get("parent_freeze_sha256") != sha(parent_path):
        raise RuntimeError("V22_PARENT_ATTESTATION_MISMATCH")
    inventory = frozen.get("asset_inventory")
    if not isinstance(inventory, dict):
        raise RuntimeError("V22_PARENT_INVENTORY_MISSING")
    for name, item in inventory.items():
        relative = Path(str(item.get("relative_path", "")))
        source = (root / relative).resolve()
        if (relative.is_absolute() or not source.is_relative_to(root)
            or sha(source) != item.get("sha256")):
            raise RuntimeError(f"V22_PARENT_ASSET_DRIFT:{name}")
    xai = verify_xai_summary(root, out, protocol)
    pair_path = reports / "evaluation_outer_hybrid_pair.json"
    if inventory.get("hybrid_pair", {}).get("sha256") != sha(pair_path):
        raise RuntimeError("V22_PAIRED_EVALUATION_NOT_IN_PARENT_FREEZE")
    pair = read(pair_path)
    if pair.get("status") != "COMPLETED_EXPLORATORY_PAIRED":
        raise RuntimeError("V22_PAIRED_EVALUATION_INVALID")
    per_fold = pair.get("by_fold")
    if (not isinstance(per_fold, list)
        or any(not isinstance(row, dict) for row in per_fold)
        or [row.get("fold") for row in per_fold] != list(range(5))
        or any(type(row.get("n")) is not int or row["n"] <= 0 for row in per_fold)
        or sum(row["n"] for row in per_fold) != 1336
        or pair.get("n") != 1336):
        raise RuntimeError("V22_PAIRED_PER_FOLD_COVERAGE_INVALID")
    for row in per_fold:
        for arm in ("baseline", "dense"):
            values = row.get(arm)
            if not isinstance(values, dict):
                raise RuntimeError(f"V22_PAIRED_PER_FOLD_METRIC_INVALID:{row['fold']}:{arm}")
            positive, negative = values.get("positive"), values.get("negative")
            if (type(positive) is not int or type(negative) is not int
                or min(positive, negative) < 0 or positive + negative != row["n"]):
                raise RuntimeError(f"V22_PAIRED_PER_FOLD_CLASS_COUNTS_INVALID:{row['fold']}:{arm}")
            one_class = min(positive, negative) == 0
            if ((values.get("status") == "NOT_MEASURABLE_ONE_CLASS") != one_class
                or (not one_class and values.get("status") not in
                    {"ESTIMATED", "SPARSE_CLASS_EXPLORATORY"})):
                raise RuntimeError(f"V22_PAIRED_PER_FOLD_STATUS_INVALID:{row['fold']}:{arm}")
            for metric in ("roc_auc", "average_precision", "brier"):
                value = values.get(metric)
                if one_class and metric != "brier" and value is None:
                    continue
                if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= 1):
                    raise RuntimeError(f"V22_PAIRED_PER_FOLD_METRIC_INVALID:{row['fold']}:{arm}:{metric}")
    external_status = "NOT_DONE_NO_INDEPENDENT_COHORT_OR_REGIONAL_REFERENCE"
    documents = build_sections(frozen, xai, external_status=external_status,
                               pair_by_fold=per_fold)
    documents["results"] += (
        "\n## ردگیری اعداد\n\n"
        f"SHA256 بستهٔ منجمد والد: `{sha(parent_path)}`؛ "
        f"SHA256 مقایسهٔ Hybrid: `{sha(pair_path)}`؛ "
        f"SHA256 تحلیل XAI نسخهٔ ۲.۲: `{sha(reports / 'xai_v22_summary.json')}`. "
        "نگاشت کامل فصل به artifact در manifest همین نسخه ثبت شده است.\n")
    files = {}
    for section, content in documents.items():
        path = reports / f"thesis_method_x_v22_{section}_FA.md"
        _text_once(path, content)
        files[section] = {"relative_path": path.relative_to(root).as_posix(),
                          "sha256": sha(path)}
    manifest = {"schema_version": 1, "status": "READY_SCOPED_THESIS_V22",
                "protocol_sha256": sha(reports / "fasttrack_v22_protocol.json"),
                "parent_freeze_sha256": sha(parent_path),
                "v21_attestation_sha256": sha(attestation_path),
                "xai_v22_summary_sha256": sha(reports / "xai_v22_summary.json"),
                "xai_v22_gate_status": xai["status"],
                "hybrid_pair_sha256": sha(pair_path), "documents": files,
                "section_sources": {
                    "methods": {
                        "parent_preregistration": inventory["preregistration"],
                        "parent_config": inventory["config"],
                        "parent_inner_lock": inventory["inner_lock"],
                        "v22_protocol_sha256": sha(reports / "fasttrack_v22_protocol.json"),
                        "v22_xai_context_sha256": sha(reports / "xai_v22_context.json"),
                    },
                    "results": {
                        "parent_mx04_scratch": inventory["mx04_scratch"],
                        "parent_mx04_head": inventory["mx04_head"],
                        "parent_hybrid_pair": inventory["hybrid_pair"],
                        "v22_xai_summary_sha256": sha(reports / "xai_v22_summary.json"),
                    },
                    "discussion": {
                        "parent_data_audit": inventory["data_audit"],
                        "parent_freeze_sha256": sha(parent_path),
                        "external_validation_status": external_status,
                    },
                },
                "external_validation": False,
                "disease_region_confirmation": False,
                "original_full_MX7_completed": False,
                "supersedes_parent_paper_text_for_v22_reporting": True,
                "claim_scope": "INTERNAL_EXPLORATORY_TECHNICAL_ONLY"}
    immutable_json(reports / "fasttrack_v22_thesis_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "documents": files}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
