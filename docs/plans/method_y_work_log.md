# Method Y — work log

Append-only record of implementation sessions. Newest last. Written so a
different agent, or a person, can tell exactly what changed and what did not.

This file does **not** start Method Y training.

---

## 2026-08-15 — Pretraining closeout then defect repair (no A4 run)

Two passes on live branch `method-y-post-revival-sanitized-20260812` and on
publishable `methody/pretraining-sanitized-v2`. Method X owns the GPU; this
session did not start or stop MX-02.

**Canonical write-up:** `docs/plans/agent_session_ledger_20260815.md`
**Machine twin:** `results/method_y_revival_v2/journals/session_ledger_20260815.json`

### Pass 1 (retracted as a green closeout)

- Ported Codex P2–P6 contracts onto live `src/method_y_revival/`.
- Commit `9fcf5c105cede77b61805e5cc1518aa2ff398678`.
- Overclaimed P0 (history still leaky), Task 4 `COMMITTED` with schema 2.0 stubs,
  and `training_allowed=true`. **Do not resume from that status.**

### Pass 2 (current)

- Live commit `b1b99df32f4ab0a2a7cc4d4cc887eb0e6638fcbb`.
- Session docs record `f36218d339ef7c743aee0f65c4f8564e142fe207`.
- Sanitized twin `46520bfd058cf3c81f79581e3f242e4d9e496944` (parquet-free).
- Evidence schema 2.1; Task 4 demoted to `AWAITING_AGENT_MULTI_PASS_QC`;
  adversarial quality FAIL; `training_allowed=false`.
- `--preflight-only` exists; live preflight is `PREFLIGHT_BLOCKED_TASK4_NOT_COMMITTED`.
- Targeted Method Y v2 tests 150 passed. Full pytest 1547/36/1; failures out of scope.
- Live HEAD still `NOT_PUBLISHABLE`. Do not push leaky refs.

### Explicitly not done

Method Y `--execute-training`, A4 HPO, live `filter-repo`, leaky push, Method X
file edits, post-training stages 1–21.

---

## 2026-08-16 — Remaining launch/label holes (no A4 run)

Closed on the live tree without training, push, or live history rewrite:

- Legacy `scripts/run_method_y_revival.ps1` is preflight-only. Python
  `--train-*-regional` fails closed unless v2 Task 4 is proof-bearing
  `COMMITTED`.
- v2 `--project-root` and `build_method_y_branch_readiness_v2.py` reject any
  tree other than this checkout, including other paths on `H:`.
- AMYLCENT amendment is bound in `branch_readiness.py`. Locked amyloid
  feature-contract bytes were not mutated. Amyloid claim now includes
  `AMYLOID_OFFICIAL_CENTILOID_CTX_EQUIVALENCE_NOT_PROVEN`.
- Rubric `docs/METHOD_Y_TASK4_SPEC_QUALITY_V1.md`. Adversarial re-review
  SPEC `PASS`, QUALITY `FAIL`. `finalize_task4_agent_review` not called.
- Honest rebuild of `branch_readiness_v2.json`: source-regional
  `development_allowed=false` while Task 4 is `AWAITING_AGENT_MULTI_PASS_QC`.
- `PRETRAINING_READINESS_V2.json` refreshed; `training_allowed=false`.
- Isolation note: Method X ResNet18 C staging step **2420** (not owned here).
- Targeted v2 suite: **158 passed**. Live `--preflight-only`:
  `PREFLIGHT_BLOCKED_TASK4_NOT_COMMITTED`. `training_allowed=false`.

Not done: Method Y `--execute-training`, Task 4 `COMMITTED`, leaky push, live
`filter-repo`, post-training stages 1–21.

---

## 2026-08-16 — P4/P5 hash reconfirm and append-only amendments (no A4 run)

Executed the agreed P4/P5 programme against `H:\A4 Learn` without mutating
locked Task 4 contract bytes and without calling `finalize_task4_agent_review`.

- P4: clinical and PACC source snapshots still match
  `a8793e40…` / `cf994d09…`. Locked endpoint/split/scanner/external hashes
  match. No rebuild.
- P5 Tau: 7 previously UNKNOWN rows were list-root sidecars with Siemens
  Manufacturer on the already-linked file. GE and Philips remain 9 events.
  Locked scanner JSON unchanged. Claim stays closed.
- P5 external: hashed public NACC UDS3 RDD/DED and OASIS-3 imaging
  dictionary. ADNI DATADIC stays LONI-gated. DUA `PENDING`. No outcomes.
- AMYLCENT: A4 codebook `183.07 × SUVRCER − 177.26` matches 4492 SUBJINFO
  rows (max abs error ~4e-14). Official CTX VOI still unproven.
- `training_allowed=false`. Live HEAD still `NOT_PUBLISHABLE`.

Not done: Task 4 `COMMITTED`, Method Y training, leaky push, live `filter-repo`.

---

## 2026-08-16 — Four-gate Method Y execution (no A4 HPO)

Fail-closed P0–P4 on live `method-y-post-revival-sanitized-20260812`. Method X
files were left unstaged. No push. No live `filter-repo`. Locked Task 4 science
bytes were not rebuilt.

- P0: six Task 4 science hashes `MATCH_NO_REBUILD`. Implementation snapshot
  rebound to Method-Y-only commit `54057162` (`branch_readiness.py` AMYLCENT
  honesty gate). Private blob still reachable; working-tree parquet empty.
  GPU was not owned by Method X at measurement.
- P1: targeted pytest **165 passed**, junit
  `e1f55917956f3e44a1997e6c2f04a6f491a5ee74221199af7d970f244f64d743`.
  Hash-compare of six artifacts `changed_count=0`. Adversarial QUALITY `PASS`
  under sanitized-SOT rubric amendment (live HEAD recorded
  `NOT_PUBLISHABLE`, not a fake history PASS).
  `finalize_task4_agent_review` wrote `COMMITTED` +
  `AGENT_MULTI_PASS_QC_PASS`. Receipt
  `d531a07d0ae99de20983de50b880f3ffd4ef510163653dd19a57c6ca123c01f8`.
- P2: rebuilt `branch_readiness_v2.json`. Source-regional
  `development_allowed=true`; image / outer / claim / external stayed false.
  Retargeted only the feature-state `branch_readiness_v2.json` hash.
  `--preflight-only` → `PREFLIGHT_ONLY_TRAINING_NOT_STARTED`.
  `PRETRAINING_READINESS_V2.json` keeps `training_allowed=false` and is **not**
  the Task-6 execute receipt.
- P3: separate Task-6 receipt
  `TASK6_SOURCE_REGIONAL_SYNTHETIC_SMOKE_RECEIPT_V2.json` with
  `test_only=true`. One hash-bound synthetic-smoke dispatch; resume returned
  the same result without a second executor call. Label
  `SYNTHETIC_SMOKE_ONLY`. Not A4 CoxNet/RSF/GBSA. No `inner_hpo_state.json`,
  no `outer_result.json`.
- P4: Tau identity still 0/1314, `BLOCKED` / `NO_CLAIM`, GE/Philips still 9
  events. CTX equivalence still `false`. DUA `PENDING`.
  `external_outcomes_accessed=false`. Isolation tests 27 passed.

Terminal status:
`TASK4_COMMITTED + PREFLIGHT_ONLY_TRAINING_NOT_STARTED + SMOKE_DISPATCHED + SCIENCE_GATES_STILL_CLOSED`.

---

## 2026-08-24 — Local post-training completion, automated layered review

Executed against the versioned `results/method_y_revival_v2` root without an
independent human-review dependency and without opening either image branch.

- Verified all 35 committed regional outer artifacts against checkpoint,
  run-contract, split-membership, inner-selection, and state/result hashes.
- Rebuilt locked-OOF calibration and subgroup reports with 2,000 subject-level
  bootstrap replicates.
- Completed the longitudinal PACC mixed-effects analysis: 1,105 subjects and
  9,595 measurements; exploratory PET-by-time estimate -0.271667, 95% CI
  [-0.323367, -0.219968].
- Completed 24 RSF development challenger receipts and locked global
  development challengers. No outer family-superiority claim was made.
- Committed LEARN transport once (`access_count=1`) as internal transport only;
  it was not used for selection and is not external validation.
- Preserved the paired A/T negative result: Tau incremental value beyond
  Amyloid was not demonstrated.
- Reconciled source-regional provenance. Amyloid remains A4 AVID-VOI AMYLCENT,
  not proven official GAAIN CTX; Tau remains source PERSI white-matter SUVR, not
  proven CenTauR.
- Screened fully open datasets. No dataset matched the locked primary future-CDR
  external contract. Ran DLBS/OpenNeuro ds004856 v1.2.0 exactly once as a
  secondary exploratory longitudinal-cognition replication: 215 subjects, 562
  observations, Amyloid-by-time estimate -0.0148405, 95% CI
  [-0.02724998, -0.00243101], p=0.01908. It is not PACC-equivalent, not primary
  CDR validation, not a model-transport result, and not a generalization claim.
- Rebuilt the aggregate public bundle; recursive privacy scan passed and no
  individual-level output was included.
- Finalizer made idempotent: unchanged evidence now preserves timestamp and
  report hash across reruns.

Terminal status:
`LOCAL_INTERNAL_PIPELINE_COMPLETED_CLAIMS_RESTRICTED + IMAGE_OPTIONAL_BLOCKED + EXTERNAL_PRIMARY_OPTIONAL_UNAVAILABLE`.

Final verification addendum:

- Method Y focused tests: **340 passed**.
- Repository-wide strict tests: **1956 passed, 1 skipped, 0 failed**, with
  `FutureWarning` and `RuntimeWarning` promoted to failures; zero warnings were
  emitted.
- Public bundle rebuilt after the test run; privacy status `PASS`.
- Post-training report remained byte-stable over repeated finalization:
  `d241a75873942d3a1e3c2d7d0581e602afc1c5d08625851cdd088d9d4c3231f4`.
- Consistency guard: `PASS_WITH_SCIENCE_GATES_BLOCKED`; no positive, image,
  causal, diagnostic, or external-generalization claim was opened.
- BrainIAC readiness drift was reconciled only through its official anchored,
  append-only transition path; CAFT training remains blocked and only
  `PREPARE_RAW_T1` is authorized.

## 2026-08-25 - image v3 Plan 010 executed + strict audit closure (ox-alpha)

Commit 4ef3495 (+ this audit commit). Report:
`advisor-plans/agent-reports/method-y-image-v3-plan-010-20260825T103000Z.md`.

- Provenance-bound pattern cache: contract now binds
  `readiness_run_contract_sha256` + `source_artifact_sha256` +
  compression; HDF5 group `source_sha256` attr verified against membership
  (`HDF5_GROUP_SOURCE_SHA_MISMATCH` fail-closed); tracer attr enforced via
  new `expected_tracer` kwarg (`CACHE_GROUP_TRACER_MISMATCH`) and written on
  commit; launcher passes branch tracer and verifies readiness run-contract
  digest (reads `run_contract.json:digest`, not file hash).
- Runtime exact-resume hardening: idempotent
  `finalize_terminal_checkpoint` recovers crash between terminal epoch and
  locked checkpoint/summary without retraining; locked-state comparison uses
  tensor-wise equality; second-run summary carries resumed pointers.
- Tests added this card: 6 provenance + tracer-mismatch + 6 resume/launcher
  (focused file set 17 passed). Full Method Y suite re-run live during audit:
  **450 passed, 0 failed**.
- Caches verified against summary hashes post-audit: amyloid
  `31981b33...` 895/895, tau `4095fcbc...` 306/306, both COMPLETED, H-only,
  float16 with finite_fraction=1.0; quantitative_scale_claim_allowed=false
  preserved. No outer-test, no v1/v2/raw mutation.

## 2026-09-08 - Method Y v4 thesis reporting (read-only, no retrain)

- Verdict: training FINALIZED 50/50, verifier --level final PASS + readiness PASS (live re-run, CPU). No rerun. Terminal-snippet privacy FAIL = absolute H:\ paths in 10 scanned_root fields, remediated append-only via sanitized twin + transaction (original preserved). Per-pair privacy 10/10 PASS. Science blockers by-design (quantitative/scanner/external + post-hoc exposure).
- Analysis: inner amyloid A4 / tau A1 (locked, outer not consulted); outer paired NI (dIBS 0.01/dUno 0.02, bootstrap 2000): IBS RETAIN_NEGATIVE 9/10 (tau A3vsA4 +0.016 [0.0059,0.0262] INCONCLUSIVE_UNDERPOWERED, outer-mean 0.11576->0.13118); Uno RETAIN 3/10 rest INCONCLUSIVE_UNDERPOWERED; omnibus p=0.101; scanner probe macro AUC 0.43-0.58 (supervisor-corrected) vs null, LMO NOT_ESTIMABLE; calibration 1640d NOT_ESTIMABLE; subgroups descriptive-only. Ceiling POST_HOC_INTERNAL_EXPLORATORY.
- Thesis (append-only, 3 new files + governance mirror, no finalized bytes mutated): docs/thesis_traceability/method_y_v4_results_20260908.md, docs/thesis_traceability/method_y_v4_discussion_limitations_20260908.md, results/method_y_revival_v4_image_factorial_v1/post_eval/THESIS_EXPANDED_20260908.md; mirror governance/method_y_v4_thesis_report_20260908.json. Skills: tdd/statistical-analysis/scientific-writing + strict supervisor Stages 1-2 PASS (Stage-2 corrected scanner range). R2 all-10-IBS-retained line superseded by locked JSON on tau A3vsA4 (reported honestly). Final Stage-3 audit pending. No push (P0).

## 2026-09-08 - Stage-3 FINAL supervisor audit: PASS sign_off=true, no violations

- Checks 21/21 PASS (completion, verifier, err, privacy path-only+sanitized, by-design blockers, inner A4/A1, IBS 9/10 + tau depth +0.016 inconclusive, Uno 3/10, omnibus p=0.101, stability, scanner 0.43-0.58 LMO NOT_ESTIMABLE, calibration 1640 NOT_ESTIMABLE, subgroups descriptive, ceiling 4+3, no fabrication, no positive/external claim, NOT_MEASURABLE honored, no finalized edit, no push/training, R2 discrepancy disclosed, scanner corrected).
- Residual risks (accepted, documented): no timestamped verifier rerun artifact; R2 note lives in mirror+log not prose; THESIS_EXPANDED under git-ignored results/; pre-existing dirty MX files unrelated; tau underpowered Uno misread risk with explicit warnings.

## 2026-09-08 - Attribution repair session (screenshot edit completed + hardened)

- Found: 4-file attribution stack UNTRACKED + mid-edit; launcher pilot crashed deterministically; legs table un-PASSable; evidence cited-not-written; tau dim mismatch; determinism theater; DRAFT ungated.
- Audits: A-code 12 findings, B-science 7, C-provenance 14 (all BLOCKED); post-fix verifier 8 new issues (HIGH run-determinism effective-flags + 3 MEDIUM evidence/NaN/ledger) — all repaired, re-proven.
- Repairs: per-cell reload + (cell,subject) keys; pure build_production_legs; 4 evidence files written (+projection/DAG/sensitivity/dry-run variants); dims 11/12 + bins inference + dropout 0.25; resume guards; deadline TIMEOUT; CONFIG_NOT_FINAL gate (DRAFT kept, production BLOCKED by design); BAN parity; 5-arm preflight + determinism assert; h5py context; weights_only; tamper BLOCK; privacy allowlist/POSIX-UNC; project_root relative.
- Proof: 30/30 tests GREEN; dry-run PASS (4 files); preflight PASS (verifier/layers/determinism/ban, digest 969ebd0f); run BLOCKED exit 3 (designed). Provenance: results/.../provenance/attribution_repair_session_20260908.json. Thesis: docs/thesis_traceability/method_y_v4_attribution_status_20260908.md. No checkpoint mutated; no push (P0).

## 2026-09-08 - Post-fix re-validation (verifier + live recompile, 3-layer supervision)

- Protocol D approved-with-7-amendments (scratch-only outputs, 3-tier compare, dirty-code pinning, joint STAND gate, mutation guard, allowlist commit). Verifier pristine + locked configs pristine (HEAD c6ccef5).
- Live: readiness PASS, final PASS 50/50; post_eval preflight+run to _revalidation_20260908 PASS; scratch report BYTE-identical bf963e82 (canonical + 1e-12 science: 10/10 contrasts, omnibus p=0.10077603 — zero mismatches); all 6 locked hashes unchanged; thesis_results.md identical.
- E evidence audit PASS (stand approved, no corrections); own layer-3 cross-check confirmed (IBS 9/10, tau depth +0.0160399, Uno 3/10 exact, omnibus p). Recorded: docs/thesis_traceability/method_y_v4_revalidation_20260908.md + provenance/revalidation_session_20260908.json. Claim: post_eval recompilation STANDS under R2-dirty code; thesis numbers current, zero corrections. No push.

## 2026-09-08 - Attribution PRODUCTION on locked finals (operator-ordered, CPU run)

- Versioned promotion DRAFT->FINAL (operator order); GPU verified idle (0%, no compute tenant) but launcher has no device path so execution was CPU-single-process (~1h45; deterministic, effective flags true). Fresh preflight PASS on FINAL (5 arms, ban parity, digest 969ebd0f).
- Incidents (all root-caused + fixed, tests added): FINAL_DIGEST_MISMATCH -> frozen-chain provenance binding (exact bytes + self-hashed FINALIZED outer state; embedded 20c7.. generation digest recorded); wrong-cohort v2-splits enumeration -> development-role cap intersection (fold_0 109 == locked n_test, coverage 560/306); raw-tabular object crash -> locked FoldTabularTransformer per cell; pilot OOM 1.72GiB -> agreement capped at 20 rows float32; one transient preflight stall (rerun PASS, no hang).
- Output: 50 cells COMPLETE, 4330 maps, report PASS (exploratory). Agreement RED_FLAG (rho 0.085, dice 0.145) -> LEG-04 NOT_MEASURABLE; fidelity 15/30 over 45 cells; share mean 0.875 (4 below floor named; 5 tau/o0 unevaluated); sensitivity PASS; LEG-07 NOT_MEASURABLE. Supervisor F audited (1 BLOCKER privacy pilot-list remediated by re-tier to count+sha; 2 HIGHs as disclaimers). Thesis: docs/thesis_traceability/method_y_v4_attribution_results_20260908.md. 32/32 tests GREEN. No checkpoint mutated; results/ ignored; no push.
