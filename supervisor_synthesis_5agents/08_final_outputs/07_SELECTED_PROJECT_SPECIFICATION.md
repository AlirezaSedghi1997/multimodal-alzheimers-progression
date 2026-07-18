# مشخصات ۴۱‌بندی طرح منتخب

1. **Title:** Prespecified, leakage-closed prognostic Method Y with a cross-fitted biology-function discordance secondary analysis.
2. **Question:** Does baseline PET/clinical information predict future sustained CDR progression within A4, and does a fold-local discordance feature add reproducible information?
3. **Hypothesis:** A prespecified future-progression survival analysis with locked negative tau increment is more defensible than a new complex model.
4. **Conservative novelty claim:** The contribution is an auditable, cross-fitted operationalization and matched negative-result protocol; no claim of a new disease mechanism is made.
5. **Five nearest precedents:** ProMUSE (multimodal mismatch), residual-based resilient brain aging, amyloid-PET decision support, updated PET appropriate-use criteria, and modern survival prognostication.
6. **Substantive difference:** Index-time, future-outcome, subject-grouped evaluation with explicit scanner/provenance gates and a locked negative tau-beyond-amyloid result.
7. **Cohort:** A4 development; LEARN is internal transport only; ADNI/SCAN/OASIS-3 are future external tracks after DUA.
8. **Inclusion:** Baseline CDR=0, valid index PET before outcome, valid clinical/cognitive covariates, subject-level linkage, and complete follow-up definition.
9. **Exclusion:** Post-origin PET, ambiguous linkage, in-brain NaN/Inf, unknown tracer, same-PET-derived target, and unresolved scanner/geometry failure.
10. **Modalities:** FBP amyloid PET, FTP tau PET, T1 MRI where a validated transform exists, clinical/cognitive tables.
11. **Outcome:** Future sustained CDR progression; PACC decline is secondary only when its future-only contract is complete.
12. **Horizon:** Index PET to first confirmed worsening, with censoring at last reliable follow-up.
13. **Preprocessing:** Tracer-specific, quantitative-scale-preserving, fold-local scaling/harmonization, transform provenance locked.
14. **Architecture:** Primary CoxNet/clinical regional survival; no new raw 3D architecture in MVP.
15. **Components:** Clinical+cognition baseline, FBP global/regional burden, FTP regional burden, optional cross-fitted residual.
16. **Losses:** Cox partial likelihood for survival; IBS for inner selection; no binary classifier for censored data.
17. **Missingness:** No outcome-based imputation; fold-local imputation with missingness indicators and explicit exclusions.
18. **Uncertainty:** Clustered subject bootstrap, fold variability, and calibration intervals.
19. **Fusion:** Matched additive models first; discordance is a secondary scalar, not an opaque MoE.
20. **Interpretation:** Coefficient/feature stability, calibration, subgroup and scanner audits; XAI only after model contract lock.
21. **Split:** Subject-grouped locked five outer folds and three inner folds.
22. **Nested CV:** All feature fitting, selection, imputation and calibration inside inner training data; outer test exactly once after lock.
23. **External validation:** Without refit or reselection on ADNI/SCAN/OASIS-3 after DUA and mapping lock.
24. **Baselines:** Clinical+cognition; clinical+global FBP; clinical+regional FTP; matched combined model.
25. **Ablations:** Remove PET, remove discordance, shuffled/null residual, alternative normative specification, and scanner leave-out.
26. **Subgroup:** Cohort, scanner manufacturer, site where powered, sex, age bands, treatment arm, and QC strata.
27. **Robustness:** Censoring sensitivity, competing-risk/death sensitivity, unknown-scanner exclusion, and leave-site/manufacturer-out.
28. **Calibration:** Slope/intercept and horizon-specific plots with prespecified support; no test-selected horizon.
29. **Statistics:** Uno/Harrell C-index, dynamic AUC, Brier/IBS, paired clustered bootstrap, Holm adjustment for families.
30. **Compute:** CPU-first regional models; one RTX 4060 only for optional secondary image checks.
31. **Difficulty:** Moderate; mostly reuse and audit, with one small residual-feature module.
32. **Timeline:** Six weeks maximum for MVP and full internal evaluation.
33. **Risks:** Scanner shortcut, incomplete transforms, event scarcity in tau, and external DUA delay.
34. **Fallback:** Finish current Method Y as a negative/ exploratory result with no new feature if any gate fails.
35. **MVP:** Reproduce locked endpoint, rerun matched baseline/PET models, and publish audit-safe negative/positive deltas.
36. **Extended version:** Add cross-fitted biology-function discordance only if null controls and fold stability pass.
37. **Figures:** Cohort flow, endpoint timeline, nested-fold diagram, calibration curves, paired deltas, and scanner audit.
38. **Tables:** Data lineage, candidate scorecard, fold metrics, subgroup metrics, and claim checklist.
39. **Reviewer criticisms:** No independent cohort, internal confounding, modest tau events, and direct discordance precedents.
40. **Abandonment criteria:** Any leakage, incomplete outer fold, invalid provenance, failed null control, or no measurable incremental value.
41. **Contribution statement:** A reproducible, conservative prognostic audit that distinguishes technical completion from scientific claim and preserves negative evidence.
