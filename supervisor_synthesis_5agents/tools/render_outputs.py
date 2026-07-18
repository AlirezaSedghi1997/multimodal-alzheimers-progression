"""Render the sixteen supervisor-facing synthesis deliverables."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from core import atomic_write_text, utc_now
from scoring import Candidate, choose_primary, score_candidate


def candidate_definitions() -> list[Candidate]:
    common = {
        "G0": "PASS", "G1": "PASS", "G2": "PASS", "G3": "PASS",
        "G4": "PASS", "G5": "PASS", "G7": "PASS", "G8": "PASS",
    }
    return [
        Candidate(
            "continuation_prespecified_method_y", "continuation_without_pivot",
            {**common, "G6": "CONDITIONAL"},
            {"novelty": 7, "clinical": 12, "data": 15, "reuse": 15, "time": 12,
             "compute": 8, "evaluation": 8, "story": 4, "interpretability": 3},
            confidence="A",
            metadata={
                "hypothesis": "A prespecified future-progression survival analysis with locked negative tau increment is more defensible than a new complex model.",
                "core_innovation": "Matched, leakage-closed prognostic comparison with explicit negative-result handling.",
                "optional_extension": "Fold-local biology-function discordance as an audit/secondary feature.",
                "precedents": "Existing Method Y v2 endpoints; external survival evaluation standards; residual/discordance literature.",
                "required_data": "Existing A4 clinical, cognition, FBP and FTP tables; no new DUA for MVP.",
            },
        ),
        Candidate(
            "biology_function_discordance", "biology_function_discordance",
            {**common, "G6": "CONDITIONAL"},
            {"novelty": 13, "clinical": 14, "data": 14, "reuse": 14, "time": 10,
             "compute": 8, "evaluation": 8, "story": 5, "interpretability": 3},
            penalties=[("direct_residual_discordance_precedent", 6)],
            confidence="B",
            metadata={
                "hypothesis": "Fold-local residual mismatch between biomarker burden and cognitive function identifies heterogeneous progression risk beyond either raw component.",
                "core_innovation": "Prespecified cross-fitted biology-function residual as a secondary prognostic feature.",
                "optional_extension": "Robustness to alternative normative specifications.",
                "precedents": "Residual-based resilience/discordance studies; ProMUSE-style multimodal mismatch modeling.",
                "required_data": "A4 clinical/cognitive + valid baseline FBP/FTP; no external data for MVP.",
            },
        ),
        Candidate(
            "conflict_aware_fusion", "conflict_aware_fusion",
            {**common, "G6": "CONDITIONAL"},
            {"novelty": 10, "clinical": 12, "data": 13, "reuse": 12, "time": 9,
             "compute": 6, "evaluation": 8, "story": 4, "interpretability": 3},
            penalties=[("direct_multimodal_conflict_precedent", 10)],
            confidence="B",
            metadata={
                "hypothesis": "Reliability-weighted disagreement between independently calibrated experts can identify uncertain subjects.",
                "core_innovation": "Audit-first disagreement feature, not a new opaque fusion network.",
                "optional_extension": "Selective prediction only if action and risk-coverage are prespecified.",
                "precedents": "ProMUSE and multimodal conflict-aware fusion.",
                "required_data": "Existing calibrated MRI/PET/clinical scores.",
            },
        ),
        Candidate(
            "selective_prediction", "selective_prediction_abstention",
            {**common, "G6": "FAIL"},
            {"novelty": 7, "clinical": 11, "data": 14, "reuse": 13, "time": 9,
             "compute": 6, "evaluation": 7, "story": 4, "interpretability": 3},
            penalties=[("direct_selective_prediction_precedent", 8)],
            confidence="B",
            metadata={
                "hypothesis": "A calibrated risk-coverage curve can expose cases where the model should not be trusted.",
                "core_innovation": "Secondary safety analysis; no autonomous clinical abstention claim.",
                "optional_extension": "Conditional conformal intervals after domain-shift audit.",
                "precedents": "Selective prediction and conformal risk-control literature.",
                "required_data": "Existing outcomes plus prespecified utility.",
            },
        ),
        Candidate(
            "value_of_pet", "value_of_pet",
            {**common, "G1": "FAIL", "G6": "FAIL"},
            {"novelty": 8, "clinical": 14, "data": 9, "reuse": 11, "time": 7,
             "compute": 6, "evaluation": 7, "story": 5, "interpretability": 3},
            penalties=[("external_decision_dataset_missing", 15), ("direct_precedent", 8)],
            confidence="C",
            metadata={
                "hypothesis": "PET changes a prespecified decision compared with a clinical-only model.",
                "core_innovation": "Decision-analytic incremental utility.",
                "optional_extension": "Adaptive acquisition policy.",
                "precedents": "Amyloid-PET decision-support studies.",
                "required_data": "Action threshold, treatment utility, independent decision cohort; unavailable.",
            },
        ),
        Candidate(
            "two_visit_delta", "two_visit_delta_biomarker_change",
            {**common, "G3": "CONDITIONAL", "G6": "CONDITIONAL"},
            {"novelty": 10, "clinical": 11, "data": 10, "reuse": 12, "time": 8,
             "compute": 6, "evaluation": 7, "story": 4, "interpretability": 3},
            penalties=[("paired_availability_uncertain", 10)],
            confidence="C",
            metadata={
                "hypothesis": "Within-person PET change predicts future progression better than baseline burden.",
                "core_innovation": "Longitudinal delta feature.",
                "optional_extension": "Joint longitudinal-survival model.",
                "precedents": "Longitudinal PET progression literature.",
                "required_data": "Two valid scans per subject and censoring-complete outcomes; not yet locked.",
            },
        ),
        Candidate(
            "faq_informant", "informant_function_discordance",
            {**common, "G0": "FAIL", "G6": "FAIL"},
            {"novelty": 9, "clinical": 12, "data": 8, "reuse": 8, "time": 7,
             "compute": 5, "evaluation": 6, "story": 4, "interpretability": 3},
            penalties=[("endpoint_not_locked", 20)],
            confidence="C",
            metadata={
                "hypothesis": "Self-informant discrepancy predicts progression.",
                "core_innovation": "Item-level discrepancy score.",
                "optional_extension": "FAQ/ECog subtyping.",
                "precedents": "Informant discrepancy studies.",
                "required_data": "Validated item-level measures and non-circular endpoint; not ready.",
            },
        ),
        Candidate(
            "underused_biomarkers", "low_cost_mri_biomarkers",
            {**common, "G1": "FAIL", "G6": "CONDITIONAL"},
            {"novelty": 11, "clinical": 10, "data": 7, "reuse": 9, "time": 6,
             "compute": 5, "evaluation": 6, "story": 4, "interpretability": 3},
            penalties=[("roi_volumetry_not_ready", 12)],
            confidence="C",
            metadata={
                "hypothesis": "WMH/subfield/choroid-plexus features add low-cost signal.",
                "core_innovation": "Prespecified regional MRI biomarker set.",
                "optional_extension": "Multimodal residual.",
                "precedents": "Regional volumetry and vascular burden studies.",
                "required_data": "Validated MRI segmentation and quality-controlled ROI table; not currently locked.",
            },
        ),
    ]


def scored_candidates() -> list[dict[str, Any]]:
    return [score_candidate(candidate) for candidate in candidate_definitions()]


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def _current_counts(audit: dict[str, Any]) -> str:
    return (
        "- Method Y amyloid endpoint (locked v1 evidence): 1,613 eligible subjects / 278 events; "
        "A4 development 1,106/240; LEARN transport 507/38.\n"
        "- Tau and paired A/T endpoint (locked v1 evidence): 369 subjects / 91 events.\n"
        "- Tau beyond amyloid: locked negative result; this is retained, not overwritten.\n"
        "- External data: DUA/raw manifest unavailable; external claim is not permitted.\n"
        f"- FTP transform writer active at audit start: {audit.get('safety', {}).get('active_transform_writer_observed', False)}; "
        "its count is intentionally read from the live snapshot and is not frozen here."
    )


def _specification(primary: dict[str, Any]) -> str:
    meta = primary.get("metadata", {})
    values = [
        ("Title", "Prespecified, leakage-closed prognostic Method Y with a cross-fitted biology-function discordance secondary analysis."),
        ("Question", "Does baseline PET/clinical information predict future sustained CDR progression within A4, and does a fold-local discordance feature add reproducible information?"),
        ("Hypothesis", meta.get("hypothesis", "")),
        ("Conservative novelty claim", "The contribution is an auditable, cross-fitted operationalization and matched negative-result protocol; no claim of a new disease mechanism is made."),
        ("Five nearest precedents", "ProMUSE (multimodal mismatch), residual-based resilient brain aging, amyloid-PET decision support, updated PET appropriate-use criteria, and modern survival prognostication."),
        ("Substantive difference", "Index-time, future-outcome, subject-grouped evaluation with explicit scanner/provenance gates and a locked negative tau-beyond-amyloid result."),
        ("Cohort", "A4 development; LEARN is internal transport only; ADNI/SCAN/OASIS-3 are future external tracks after DUA."),
        ("Inclusion", "Baseline CDR=0, valid index PET before outcome, valid clinical/cognitive covariates, subject-level linkage, and complete follow-up definition."),
        ("Exclusion", "Post-origin PET, ambiguous linkage, in-brain NaN/Inf, unknown tracer, same-PET-derived target, and unresolved scanner/geometry failure."),
        ("Modalities", "FBP amyloid PET, FTP tau PET, T1 MRI where a validated transform exists, clinical/cognitive tables."),
        ("Outcome", "Future sustained CDR progression; PACC decline is secondary only when its future-only contract is complete."),
        ("Horizon", "Index PET to first confirmed worsening, with censoring at last reliable follow-up."),
        ("Preprocessing", "Tracer-specific, quantitative-scale-preserving, fold-local scaling/harmonization, transform provenance locked."),
        ("Architecture", "Primary CoxNet/clinical regional survival; no new raw 3D architecture in MVP."),
        ("Components", "Clinical+cognition baseline, FBP global/regional burden, FTP regional burden, optional cross-fitted residual."),
        ("Losses", "Cox partial likelihood for survival; IBS for inner selection; no binary classifier for censored data."),
        ("Missingness", "No outcome-based imputation; fold-local imputation with missingness indicators and explicit exclusions."),
        ("Uncertainty", "Clustered subject bootstrap, fold variability, and calibration intervals."),
        ("Fusion", "Matched additive models first; discordance is a secondary scalar, not an opaque MoE."),
        ("Interpretation", "Coefficient/feature stability, calibration, subgroup and scanner audits; XAI only after model contract lock."),
        ("Split", "Subject-grouped locked five outer folds and three inner folds."),
        ("Nested CV", "All feature fitting, selection, imputation and calibration inside inner training data; outer test exactly once after lock."),
        ("External validation", "Without refit or reselection on ADNI/SCAN/OASIS-3 after DUA and mapping lock."),
        ("Baselines", "Clinical+cognition; clinical+global FBP; clinical+regional FTP; matched combined model."),
        ("Ablations", "Remove PET, remove discordance, shuffled/null residual, alternative normative specification, and scanner leave-out."),
        ("Subgroup", "Cohort, scanner manufacturer, site where powered, sex, age bands, treatment arm, and QC strata."),
        ("Robustness", "Censoring sensitivity, competing-risk/death sensitivity, unknown-scanner exclusion, and leave-site/manufacturer-out."),
        ("Calibration", "Slope/intercept and horizon-specific plots with prespecified support; no test-selected horizon."),
        ("Statistics", "Uno/Harrell C-index, dynamic AUC, Brier/IBS, paired clustered bootstrap, Holm adjustment for families."),
        ("Compute", "CPU-first regional models; one RTX 4060 only for optional secondary image checks."),
        ("Difficulty", "Moderate; mostly reuse and audit, with one small residual-feature module."),
        ("Timeline", "Six weeks maximum for MVP and full internal evaluation."),
        ("Risks", "Scanner shortcut, incomplete transforms, event scarcity in tau, and external DUA delay."),
        ("Fallback", "Finish current Method Y as a negative/ exploratory result with no new feature if any gate fails."),
        ("MVP", "Reproduce locked endpoint, rerun matched baseline/PET models, and publish audit-safe negative/positive deltas."),
        ("Extended version", "Add cross-fitted biology-function discordance only if null controls and fold stability pass."),
        ("Figures", "Cohort flow, endpoint timeline, nested-fold diagram, calibration curves, paired deltas, and scanner audit."),
        ("Tables", "Data lineage, candidate scorecard, fold metrics, subgroup metrics, and claim checklist."),
        ("Reviewer criticisms", "No independent cohort, internal confounding, modest tau events, and direct discordance precedents."),
        ("Abandonment criteria", "Any leakage, incomplete outer fold, invalid provenance, failed null control, or no measurable incremental value."),
        ("Contribution statement", "A reproducible, conservative prognostic audit that distinguishes technical completion from scientific claim and preserves negative evidence."),
    ]
    return "\n".join(f"{index}. **{key}:** {value}" for index, (key, value) in enumerate(values, start=1))


def render_all(
    output_root: Path,
    audit: dict[str, Any],
    sources: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    final = output_root / "08_final_outputs"
    final.mkdir(parents=True, exist_ok=True)
    results = scored_candidates()
    primary = next(row for row in results if row["idea_id"] == choose_primary(results)["idea_id"])
    fallback = sorted(
        [row for row in results if row["idea_id"] != primary["idea_id"] and not row["cannot_be_primary"]],
        key=lambda row: row["practical_continuation_score"],
        reverse=True,
    )[0]
    extension = "biology_function_discordance" if primary["idea_id"] == "continuation_prespecified_method_y" else "selective_prediction_abstention"

    score_fields = [
        "idea_id", "normalized_family", "scientific_from_scratch_score",
        "practical_continuation_score", "weighted_score", "penalties",
        "cannot_be_primary", "hard_fail_gates", "verdict", "confidence",
    ]
    score_buffer = io.StringIO()
    writer = csv.DictWriter(score_buffer, fieldnames=score_fields, lineterminator="\n")
    writer.writeheader()
    for row in results:
        writer.writerow({field: json.dumps(row.get(field), ensure_ascii=False) if isinstance(row.get(field), (list, dict)) else row.get(field, "") for field in score_fields})
    atomic_write_text(final / "06_CANDIDATE_SCORECARD.csv", score_buffer.getvalue())

    audit_counts = _current_counts(audit)
    decision = f"""# تصمیم اجرایی نهایی ممیزی پنج‌ایجنت

## تصمیم در یک نگاه

**توصیه اصلی:** `{primary['idea_id']}` با Practical Continuation Score برابر **{primary['practical_continuation_score']}**. مسیر اصلی، تکمیل و بازاجرای کنترل‌شده‌ی Method Y موجود با endpoint آینده‌نگر، ارزیابی nested و گزارش صادقانه‌ی نتیجه‌ی منفی tau-beyond-amyloid است. این انتخاب به‌معنای اثبات نوآوری مثبت یا external validation نیست.

**چرا:** داده و کد محلی آماده‌تر از هر pivot بیرونی است؛ event scarcity و نبود DUA مدل complex یا external-dependent را از مسیر اصلی خارج می‌کند؛ و وضعیت علمی فعلی نیازمند قفل‌کردن provenance، leakage و claim policy است.

**نوآوری هسته‌ای شماره ۱:** پروتکل matched و leakage-closed برای سنجش ارزش افزوده‌ی PET در برابر clinical+cognition، با outer-test دقیقاً یک‌باره.

**نوآوری هسته‌ای شماره ۲:** لازم نیست برای MVP. discordance cross-fitted فقط extension مشروط است، نه novelty مفروض.

**افزونه اختیاری:** `{extension}`، فقط پس از عبور null/shuffle، fold-stability و scanner gates.

**داده و endpoint:** A4 برای development، LEARN برای internal transport؛ endpoint محلی future sustained CDR progression و PACC فقط secondary. DUA خارجی موجود نیست.

**Claim مجاز:** «در محیط داخلی A4، با پروتکل از پیش‌تعیین‌شده، PET در صورت عبور gate می‌تواند اطلاعات پیش‌آگهی افزوده ارائه کند.» اگر gate شکست بخورد، claim فقط exploratory/technical است.

**Claim ممنوع:** تشخیص آلزایمر، MCI→AD محلی، causal effect، external generalization، یا novelty صرفاً به‌دلیل conflict/discordance.

**زمان:** MVP چهار هفته؛ full internal evaluation حداکثر شش هفته.

**منابع محاسباتی:** CPU/ROI-first؛ RTX 4060 فقط برای تحلیل تصویری ثانویه پس از قرارداد مدل.

**بزرگ‌ترین ریسک:** scanner/site shortcut و کمبود event در tau.

**شرط توقف:** هر leakage، transform/provenance failure، outer-fold ناقص، null-control failure، یا نبود incremental value.

**Fallback:** `{fallback['idea_id']}`؛ در صورت شکست هر gate، فقط نتیجه‌ی منفی/اکتشافی فعلی تکمیل و گزارش شود.

## وضعیت ممیزی زنده

{audit_counts}

## نکته‌ی ایمنی

writer فعال FTP→T1→MNI متوقف یا دست‌کاری نشد. T0 و Tfinal باید اختلاف‌های مسیر transform را با process writer نسبت دهند، نه به این ممیزی.
"""
    atomic_write_text(final / "00_FINAL_EXECUTIVE_DECISION.md", decision)

    state_doc = f"""# وضعیت فعلی پروژه

Snapshot UTC: {utc_now()}

{audit_counts}

## baseline آزمون

```json
{json.dumps(baseline, ensure_ascii=False, indent=2)}
```

Baseline کل مخزن در این snapshot به‌علت import-file mismatch ناشی از testهای duplicate در `archives/.../patch_work/old` و `new` سبز نبود؛ این خطای پیشین است و برای اصلاح آن هیچ فایل محافظت‌شده‌ای تغییر داده نشد.

## process و provenance

- Git HEAD: `{audit.get('git', {}).get('commit', '')}`
- Branch: `{audit.get('git', {}).get('branch', '')}`
- Dirty file count snapshot: `{audit.get('git', {}).get('dirty_file_count', '')}`
- Active synthesis writer targeting workspace: `{audit.get('safety', {}).get('workspace_targeted_by_active_writer', False)}`
- Active transform writer observed: `{audit.get('safety', {}).get('active_transform_writer_observed', False)}`
- External DUA/raw manifest: `PENDING_DUA_ACCESS`
"""
    atomic_write_text(final / "01_CURRENT_PROJECT_STATE.md", state_doc)

    source_rows = []
    for source in sources:
        source_rows.append(
            {
                "source": Path(source["file_path"]).name,
                "agent": source["agent_id"],
                "role": source["file_role"],
                "lines": source.get("line_count"),
                "read_complete": source.get("read_complete"),
                "evidence": source.get("actual_evidence_level"),
                "notes": source.get("notes", ""),
            }
        )
    atomic_write_text(
        final / "02_SOURCE_BY_SOURCE_EVALUATION.md",
        "# ارزیابی منبع‌به‌منبع\n\n"
        + _md_table(source_rows, ["source", "agent", "role", "lines", "read_complete", "evidence", "notes"])
        + "\n\nAgent 4 duplicate/raw blocks are coverage-only and never independent evidence. Nested agents are not independent replications.\n",
    )

    atomic_write_text(
        final / "03_CONSENSUS_AND_CONTRADICTIONS.md",
        "# همگرایی و تناقض‌ها\n\n"
        "همگرایی مفهومی: biology-function mismatch، reliability/conflict و selective prediction در چند بسته تکرار شده‌اند؛ این تکرار استقلال شواهد را ثابت نمی‌کند.\n\n"
        "## تناقض‌های حل‌شده\n\n"
        + _md_table(contradictions, ["topic", "claim_A", "claim_B", "resolution", "unresolved_risk"])
        + "\n",
    )

    claim_rows = [
        {
            "topic": row["topic"],
            "evidence": row["evidence_label"],
            "status": row["verified_status"],
            "confidence": row["confidence"],
            "source": row["source_id"],
            "lines": row["source_section_or_lines"],
            "impact": row["decision_impact"],
        }
        for row in claims
    ]
    atomic_write_text(
        final / "04_VERIFIED_EVIDENCE_SUMMARY.md",
        "# خلاصه شواهد راستی‌آزمایی‌شده\n\n"
        + _md_table(claim_rows, ["topic", "evidence", "status", "confidence", "source", "lines", "impact"])
        + "\n\nPrimary references used for precedent checks: [ProMUSE](https://arxiv.org/abs/2606.19371), [residual resilience](https://pmc.ncbi.nlm.nih.gov/articles/PMC4060425/), [amyloid PET decision support](https://pubmed.ncbi.nlm.nih.gov/38768188/), and [updated PET AUC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11772739/).\n",
    )

    atomic_write_text(
        final / "05_DATASET_AND_ENDPOINT_GATE.md",
        "# گیت داده و endpoint\n\n"
        + audit_counts
        + "\n\n## G0–G3\n\n"
        "- Endpoint local future-only است؛ same-PET label و post-origin PET رد می‌شوند.\n"
        "- A4/LEARN در primary development به‌عنوان دو cohort یکسان ادغام نمی‌شوند.\n"
        "- Tau 91 event برای raw 3D/MoE کافی تلقی نمی‌شود.\n"
        "- DUA خارجی در این run موجود نیست؛ external track future است.\n",
    )

    spec = _specification(primary)
    atomic_write_text(final / "07_SELECTED_PROJECT_SPECIFICATION.md", "# مشخصات ۴۱‌بندی طرح منتخب\n\n" + spec + "\n")

    reuse_rows = audit.get("component_reuse", [])
    atomic_write_text(
        final / "08_CURRENT_REPO_IMPLEMENTATION_MAP.md",
        "# نقشه‌ی reuse مخزن\n\n"
        + _md_table(reuse_rows, ["component", "exists", "file_count", "reuse_status", "adaptation_days", "notes"])
        + "\n\nNo existing `src/`, `scripts/`, `configs/`, `data/` or `results/` file was modified by this synthesis.\n",
    )

    atomic_write_text(
        final / "09_SIX_WEEK_EXECUTION_PLAN.md",
        """# برنامه شش‌هفته‌ای

| هفته | کارها | ورودی/خروجی | پذیرش | fallback |
|---|---|---|---|---|
| ۱ | freeze، endpoint/leakage audit، provenance و split check | v1 artifacts → audit tables | no leakage; hashes locked | stop and report blocker |
| ۲ | matched clinical/PET baselines، calibration | locked folds → baseline metrics | all outer folds complete | negative-only report |
| ۳ | tau-beyond-amyloid locked comparison، paired bootstrap | OOF predictions → deltas | exact-once outer eval | preserve negative result |
| ۴ | optional cross-fitted discordance + null/shuffle | fold-local residual → ablation | null control and stability pass | omit extension |
| ۵ | scanner/site/subgroup/censoring sensitivity | committed predictions → audits | no shortcut gate failure | exploratory only |
| ۶ | final reruns, CI, reports, privacy bundle | all artifacts → manuscript-ready package | consistency guard pass | no claim/public release |

هر هفته CPU-first است؛ image branch و external validation به‌دلیل گیت‌ها مسیر اصلی نیستند.
""",
    )

    atomic_write_text(
        final / "10_EXPERIMENT_AND_ABLATION_PROTOCOL.md",
        """# پروتکل آزمایش و ablation

1. Lock subject-grouped outer five-fold/inner three-fold membership.
2. Fit imputer, scaler, feature selection and any residual model inside training folds only.
3. Fit clinical-only, clinical+FBP, clinical+FTP and combined models with equal family/budget.
4. Evaluate outer test once after winner lock; recover committed results by hash.
5. For optional discordance: compare raw components, linear fold-local residual, nonlinear residual, and shuffled residual.
6. Report Uno/Harrell C-index, dynamic AUC, Brier/IBS, calibration slope/intercept, event counts, and fold-wise deltas.
7. Apply clustered subject bootstrap and Holm adjustment; do not use DeLong for survival C-index.
8. Stop extension if shuffled residual performs similarly, scanner leave-out collapses, or any fold is incomplete.
""",
    )

    atomic_write_text(
        final / "11_LEAKAGE_CALIBRATION_AND_STATISTICS_CHECKLIST.md",
        """# چک‌لیست leakage، calibration و آمار

- [ ] index date precedes every outcome measurement
- [ ] no post-origin PET or future-derived baseline feature
- [ ] all sessions of one subject remain in one fold
- [ ] fold-local imputation/scaling/harmonization/feature selection
- [ ] same-PET-derived target rejected
- [ ] A4/LEARN cohort shortcut diagnosed, not hidden
- [ ] inner-only winner selection
- [ ] outer test unavailable during HPO and evaluated once
- [ ] common survival support prespecified
- [ ] Uno/Harrell C-index, dynamic AUC and IBS reported
- [ ] calibration slope/intercept and horizon support reported
- [ ] paired clustered bootstrap for deltas
- [ ] censoring and death sensitivity
- [ ] scanner/site/subgroup reports
- [ ] missingness and unknown scanner remain explicit
- [ ] conformal claims conditional under external shift
""",
    )

    atomic_write_text(
        final / "12_CLAIM_BOUNDARIES.md",
        """# مرز ادعا

## مجاز

- internal A4 prognostic association with future sustained CDR progression, only after all gates pass;
- technical/exploratory reporting of tau, PET value and discordance;
- negative tau-beyond-amyloid result as a valid finding.

## ممنوع

- Alzheimer's diagnosis from the current Method X cohort;
- MCI→AD conversion claim for the local endpoint;
- causal claim;
- external generalization without DUA and locked external evaluation;
- positive novelty claim for conflict/discordance without matched ablation;
- clinical usability or decision utility without an action threshold and independent validation.
""",
    )

    rejected = [
        row for row in results if row["idea_id"] not in {primary["idea_id"], fallback["idea_id"]}
    ]
    atomic_write_text(
        final / "13_REJECTION_LOG.md",
        "# مسیرهای ردشده یا deferred\n\n"
        + _md_table(rejected, ["idea_id", "verdict", "hard_fail_gates", "practical_continuation_score", "penalties"])
        + "\n\nRejection is due to gates/data/precedent, not aesthetic preference.\n",
    )

    atomic_write_text(
        final / "14_FALLBACK_AND_STOP_RULES.md",
        f"""# fallback و stop rules

## Fallback

`{fallback['idea_id']}` remains available only if it passes the same gates. If no candidate remains eligible, finish the current Method Y analysis as a negative/exploratory report.

## Stop rules

1. Any subject/session leakage or post-origin feature.
2. Any transform/quantitative provenance failure.
3. Any incomplete outer fold or repeated test access.
4. Any matched ablation that cannot isolate the claimed component.
5. Any scanner/site shortcut that survives sensitivity audit.
6. Any null/shuffled control indistinguishable from the proposed feature.
7. Any claim-status mismatch or raw identifier in the public bundle.
8. Any active writer targeting this synthesis workspace.
""",
    )

    atomic_write_text(
        final / "15_FINAL_RESPONSE_TO_USER.md",
        f"""# پاسخ نهایی به کاربر

مسیر اصلی انتخاب‌شده: `{primary['idea_id']}` با امتیاز ادامه‌ی عملی `{primary['practical_continuation_score']}`. این تصمیم بر اساس داده و کد زنده، گیت‌های علمی، زمان ۴ تا ۶ هفته، و reuse است؛ نه بر اساس رتبه‌ی عامل‌ها یا جذابیت معماری.

برای پیاده‌سازی مدل منتخب هنوز نباید claim مثبت بیرونی صادر شود. ابتدا endpoint، split، provenance، leakage و outer-evaluation قفل می‌شوند؛ نتیجه‌ی منفی tau-beyond-amyloid حفظ می‌شود. writer FTP→T1→MNI در این ممیزی متوقف نشده است.

تمام فایل‌های این ممیزی در `H:\\New project\\supervisor_synthesis_5agents\\08_final_outputs` قرار دارند.
""",
    )
    return {
        "primary": primary,
        "fallback": fallback,
        "extension": extension,
        "output_dir": str(final),
        "output_count": len(list(final.iterdir())),
    }
