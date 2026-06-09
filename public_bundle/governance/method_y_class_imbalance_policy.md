# Method Y Class Imbalance Policy (Frozen 2026-06-08)

## Scope

Method Y exploratory branch only. Method X-Strict unchanged.

## Approved training tier

**`primary_feasible_90`** — all linked FBP samples with |PET−T1| ≤ 90 days (266 patients after one-PET-per-patient selection).

| Tier | N | Pos | Neg | Min class | Role |
|------|---|-----|-----|-----------|------|
| **primary_feasible_90** | 266 | 232 | 34 | **34** | **Primary exploratory training** |
| primary_strict (30–90 d only) | 191 | 168 | 23 | 23 | Sensitivity / high-gap subset |
| gold_standard_30 (≤30 d) | 75 | 64 | 11 | 11 | **Report-only** — violates `min_neg_for_auc=20`; **forbidden for primary training** |

## gold_standard_30 explicit block

64 pos / 11 neg (85% positive) violates `minimum_counts.min_neg_for_auc=20`. Use only as high-validity **report-only** sensitivity; never as sole training tier.

## Why not `primary_strict` alone?

Policy block threshold (`min_per_class_block=100`) targets headline Method X-scale training. Method Y at 168/23 fails that gate **by design** — not a bug.

Exploratory gate: **min class ≥ 30** → 34 passes.

## Mandatory mitigations

1. `class_weight='balanced'` (or equivalent reweighting)
2. Stratified, patient-grouped CV
3. Report **PR-AUC**, sensitivity, specificity, balanced accuracy — not ROC-AUC alone
4. Bootstrap CIs; explicit imbalance caveat in Results/Limitations
5. MRI-only vs PET comparisons on **identical hash intersection**

## 266 PET vs 1700 MRI — scientific framing

- **1700** = Method X-Strict T1 MRI enrolled cohort (nearly all have T1).
- **266** = Method Y eligible subset with **both** amyloid PET **and** native T1 within 90-day pairing window.
- Method Y models are trained/evaluated on the **266**; MRI-only baselines for fusion comparison use the **same 266**, not 1700.
- Full-cohort Method X MRI results (1700) remain the protected thesis anchor on a **different sample definition**.

## Claims allowed / forbidden

**Allowed:** exploratory PET-assisted biomarker-status modeling on N=266 with imbalance-aware metrics.

**Forbidden:** implying PET model trained on 266 validates on or replaces full 1700 MRI cohort without selection caveat.
