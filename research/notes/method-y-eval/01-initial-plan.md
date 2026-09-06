> DRAFT 2026-09-06 — Phase 1 of 5-stage program. Adapts the ORIGINAL Method Y post-training program (items 1–21) to CURRENT transformed state. Pending: strict-supervisor review → Phase-2 deep research → Phase-3 combined plan.

# 01 — Initial Post-Training Evaluation Plan (DRAFT)

## State deltas vs the original program (what changed, and why the plan changes)
1. **External evaluation OPTIONAL** (was gating): T08 branch only; `EXTERNAL_UNAVAILABLE_NONBLOCKING` caps claims, never blocks internal completion. Deliverable downgraded to frozen intake contract + embargo tests (no data access).
2. **Missing-info gates non-blocking:** scanner/manufacturer unknowns → `UNKNOWN`/`NOT_MEASURABLE` with evidence, never PASS-by-default, never pipeline-blocking (except where counting requires it, e.g., tau GE/Philips 9 events → LMO inestimable there).
3. **New redesigned models:** 5 factorial arms (A0 baseline / A1 +aug / A2 ResNet10-GN / A3 +aug+R10 / A4 R18-GN w0.25) replace the old single image-CNN; selection = inner-only IPCW-IBS + one-SE over 5 outer-training partitions; outer test exactly-once; 50-cell verifier.
4. **CAM attribution built** (LayerCAM primary / Score-CAM pilot / Grad-CAM sanity) — runs post-verifier-PASS only; feeds E11, not selection.
5. **Multi-stage judgment preserved:** every section below ends in a machine gate (PASS/BLOCKED/NOT_MEASURABLE/TIMEOUT), never a human override; negative results published verbatim.
6. **Fusion-readiness (cross-branch):** every subject-level output keyed by `patient_hash` + outer fold + run digest → future multimodal fusion registry. No branch may emit ID-keyed rows outside private/.

## Transparent selection criteria (applied to every section)
- (i) censoring-correctness (IPCW where censoring matters); (ii) zero outer-test reuse; (iii) deterministic/reproducible (seed-locked, rerun-identical); (iv) prespecified before seeing results; (v) computable from LOCKED artifacts (no retraining, no refit); (vi) fusion-keyed outputs.

## Sections: options → selection (S = selected)
### E1. Winner lock + final fits + outer eval exactly-once (orig 1–3)
- O1 status-quo locked orchestrator flow / O2 rebuild flow. **S: O1** — in-flight; redesign mid-run violates exactly-once. Criterion: (ii).
### E2. Discrimination: Uno C + IBS + dynamic AUC (orig 4)
- O1 Harrell's C / O2 Uno-IPCW C / O3 incident/dynamic AUC / O4 cumulative/dynamic AUC. **S: O2 + IBS + O4** (CRAN/sksurv-compatible, censoring-correct). Reject O1 (censoring bias), O3 (needs incident-density assumptions). Criterion: (i).
### E3. Calibration: slope/intercept + curves (orig 5)
- O1 horizon-specific IPCW calibration curves + slope/intercept / O2 ICI/E50/Emax descriptors / O3 D-calibration p-value. **S: O1 + O2**; O3 as sensitivity only (D-calibration fragile under heavy censoring). Criterion: (i)+(v).
### E4. At-risk/event tables per horizon (orig 6)
- O1 risk tables at locked horizons (547/1095/1640 d) from OOF / O2 KM overlays. **S: O1** (+O2 figure if cheap). Criterion: (v).
### E5. Paired bootstrap for added value (orig 7)
- O1 percentile bootstrap (B=2000) on paired dIBS/dC / O2 BCa / O3 permutation null. **S: O1 primary + O3 sensitivity**; BCa dropped (cost, instability at n=75 tau events). Criterion: (iii)+(v).
### E6. Leave-manufacturer/site-out (orig 8)
- O1 LMO / O2 leave-site-out / O3 scanner-stratified bootstrap. **S: O1 where estimable (≥10 events/manufacturer) else NOT_MEASURABLE with evidence; O3 as backup when site labels exist.** Tau GE/Philips expected NOT_MEASURABLE (9 events). Criterion: (iv).
### E7. Scanner/cohort probes (orig 9)
- O1 AUC probe on OOF embeddings (existing) / O2 manufacturer-stratified C / O3 pooled GE+Philips. **S: O1 + O2**; O3 FORBIDDEN (pooling rule). Criterion: prespecification.
### E8. Subgroup CIs (orig 10: sex/age/treatment/cognition/QC/cohort)
- O1 prespecified list + min-n rule (30 subjects/10 events) + NOT_ESTIMABLE / O2 data-driven subgroups. **S: O1**; O2 forbidden (fishing). Criterion: (iv).
### E9. PACC mixed-effects secondary (orig 11)
- O1 statsmodels MixedLM prespecified (random intercept+trend, treatment×time) / O2 GEE / O3 Bayesian hierarchical. **S: O1** (auditable, deterministic); O2 sensitivity if converges; O3 dropped (sampler nondeterminism budget). Criterion: (iii).
### E10. Tau-beyond-amyloid negative preservation (orig 12)
- O1 prespecified paired dIBS (amyloid vs amyloid+tau) with non-superiority retention (CI includes zero → retain negative) / O2 superiority retest loop. **S: O1**; O2 forbidden (reruns for significance). Applies to source-regional paired AND image tau branch (internal improvement ≠ claim).
### E11. XAI/CAM (orig 13–16 + attribution build)
- Decided: LayerCAM primary / Score-CAM pilot-convergent / Grad-CAM sanity + fidelity + randomized sanity + tabular-share + legs table (see `research/notes/cam-attribution/`). Strict `strict=True` checkpoint loads; arch built from locked config.
### E12. External intake only (orig 17–18, OPTIONAL)
- Frozen intake registry (tracer/QC/endpoint/censoring compatibility checklist, all UNMET) + embargo tests + DUA-pending record. NO mapping lock, NO evaluation, NO synthetic external. **S: contract-only.**
### E13. TRIPOD-AI/PROBAST-AI/CLAIM reports (orig 19)
- O1 machine-checked items from artifact bindings + human-attested rest, all recorded / O2 fully manual checklist. **S: O1** (evidence-linked; manual-only items marked ATTESTED with author+date).
### E14. Public bundle + privacy (orig 20)
- Existing recursive privacy scanner over public reports + aggregate-only enforcement; extend to new E-artifacts. **S: status-quo extended.**
### E15. Final claims compiler (orig 21)
- Rule engine over gate outputs: positive/external stay FALSE unless quantitative+scanner+external gates pass (they won't) → exploratory stamp + limitation ledger. **S: as specified.**
### E16. Thesis registration (new requirement)
- Writer mapping every E-result → thesis section (Results/Methods/Discussion) + limitations appendix; machine-generated from report JSONs into `docs/thesis_traceability/method_y_post_training/`.

## Explicitly out of scope (unchanged)
Retraining/refit; outer-test reuse; threshold/selection on outer; PET-in-Method-X; diagnosis/causal/external claims; synthetic external evidence; Method X files.
