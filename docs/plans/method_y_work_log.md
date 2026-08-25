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
