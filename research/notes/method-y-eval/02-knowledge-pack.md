> Assembled 2026-09-06 from: (A) subagent brief via Crossref+PubMed, (B) subagent brief via Crossref+PubMed, (C) direct inline retrieval (the dedicated R-C subagent cancelled twice — root cause below — so Topic C was executed manually with a 6-query budget). Tags: [V]=verified via API this session, [I]=expert inference, UNCERTAIN=explicitly unverified.

# 02 — Post-Training Evaluation Knowledge Pack (A+B+C)

## R-C incident record (subagent failure → fix)
Two consecutive `Task cancelled` on the "similar PET studies" agent (broad 4-topic scope + flaky hosts). Fix applied: narrowed to proven hosts only (Crossref + PubMed E-utilities), 6-query budget, UTF-16 PowerShell redirect handled, esummary retries. Lesson: keep research agents ≤10 calls, single host family, partial delivery = success.

## A. Survival discrimination/calibration [V unless noted]
1. **Uno's C (Stat Med 2011, DOI 10.1002/sim.4154, PMID 21484848):** IPCW + truncated tau; needs `T⟂C`, `G(tau)>0`, train-fold G; tail variance explodes → prespecify tau (largest t with train G≥0.15 + ≥10–15 test events), truncate IPCW weights (cap 10–20). PRIMARY.
2. **Harrell's C (JAMA 1982; Stat Med 1996):** censoring-biased, cohort-incomparable → sensitivity-only.
3. **Antolini Ctd (Stat Med 2005, DOI 10.1002/sim.2427):** curve-based, mixes calibration in → secondary only.
4. **IBS (Graf 1999; Gerds 2006 IPCW-consistency, DOI 10.1002/bimj.200610301):** common-support grid = locked horizons only, train-G, plus Brier(t) curve + scaled-vs-null IBS. Co-primary with Uno.
5. **C/D vs I/D AUC (Heagerty 2000/2005):** cumulative/dynamic at prespecified horizon + mean over grid; incident/dynamic exploratory only (1 event vs risk set, unstable at 75 events).
6. **Calibration slope/intercept per horizon (van Houwelingen 2000):** IPCW-logistic, 1–2 prespecified horizons, slope CI + plot + prediction rug; descriptive, never gate input.
7. **ICI/E50/Emax (Austin 2019, DOI 10.1002/sim.8281):** numeric secondary only; Emax outlier-driven; survival extension = adaptation (UNCERTAIN provenance).
8. **D-calibration (Haider JMLR 2020):** NOT verified via allowed APIs → UNCERTAIN; exploratory only, exact bin counts published.
9. **Determinism checklist:** lock tau, grid, risk-score def (`1-S(tau)` or sum-hazard — state one), tie rule 0.5, marginal-KM train-G (pooled vs fold-specific stated), seeds/versions, B + RNG, weight truncation rule.

## B. Validation methodology [V unless noted]
1. **Paired bootstrap (percentile primary B≥2000 event-stratified + BCa sensitivity):** paired Δ removes fold/patient effects; captures test noise only (NOT training/tuning variance — state it); studentized NOT primary at low events; permutation null = within-patient A/B label swap, censoring fixed, p secondary to CI.
2. **Nested CV (Bates/Hastie/Tibshirani JASA 2023, DOI 10.1080/01621459.2023.2197686):** outer estimates procedure error, not final-model error; selection stability reported; no outer re-tuning.
3. **LMO renamed:** true LMO = refit (FORBIDDEN here) → `STRATIFIED_MANUFACTURER_REPORT`, descriptive, never gate; 9-event cells = NOT_MEASURABLE (C SE ~0.15–0.20); pooled primary + audit-only manufacturer; GE+Philips pooling FORBIDDEN.
4. **Subgroups:** prespecified + interaction-first (Brookes 2001, PMID 11701102); within-subgroup superiority needs interaction significance; min bar ≥25 events for CIs else descriptive/NOT_MEASURABLE; k-suppression for small public cells; BH/FWER across factorial contrasts.
5. **TRIPOD+AI (BMJ 2024, PMID 38626948) / PROBAST+AI (BMJ 2025, PMID 40127903) / CLAIM (2020 PMID 33937821; 2024 PMID 38809149):** machine-checkable = splits/IDs/seeds/N-events/metric defs/B/pairing/code+env hashes; judgement (bias, utility, leakage) = TWO independent attesters + quote + digest link, capped ATTESTED_ONLY.
6. **Negatives:** non-significance = NOT SUPERIOR (never EQUIVALENT); non-inferiority needs prespecified margin + retention test; publish full Δ/CI tables, retain null arms, prespecified hierarchy (omnibus → Holm contrasts).

## C. Similar studies (direct inline retrieval, 6-query budget) [V items listed]
1. Amyloid-PET survival direction: JAD 2026 (PMID 42517808, bidirectional survival analyses), Brain 2026 (PMID 42019008, network failure across AD spectrum), Brain Commun 2025 (PMID 41089377, resilience) — field uses time-to-event framing; supports our survival endpoint choice.
2. Voxel-based AD ensemble precedent: IEEE JBHI 2017 (PMID 28113481) — old but supports aggregate voxel-attribution reporting style.
3. Augmentation-ablation comparison paper exists (harp.264, brain-MRI augmentation techniques) — supports matched augmentation-isolation pairs (A0/A1, A2/A3).
4. PACC-LME methodology: top hits off-target (diet/plasma) — NO verified prespec practice found; our MixedLM stays DESCRIPTIVE with locked convergence criteria (supervisor H4).
5. XAI sanity (Adebayo 2018): 0 PubMed hits (ML venue, expected) — landmark status UNCERTAIN via allowed APIs; our randomized-model + fidelity gates implement its substance regardless.
6. Tau-PET progression standards: fetch FAILED (0 bytes) — recorded MISS; tau branch stays exploratory with closed claims.
7. Nested-CV methodology: covered by Bates (B) — no separate item needed.

## Query log (condensed)
Crossref: Uno/Antolini/Harrell/Heagerty×2/Graf(fixed DOI)/Gerds/Austin/vHouwelingen/Efron(partial)/Bates — all HIT except Haider (MISS) and one malformed DOI (fixed). PubMed: Uno/Harrell/Graf/TRIPOD+AI/PROBAST+AI/CLAIM×2/Riley-partial/Brookes — HIT; PROBAST first-hit off-target (corrected); D-calibration 0 hits; PACC-LME off-target; sanity-Adebayo 0 hits; tau fetch FAIL. S2/OpenAlex/arxiv-export/unpaywall: resets (not used).
