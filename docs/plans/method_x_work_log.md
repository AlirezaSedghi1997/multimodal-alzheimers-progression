# Method X — work log

Append-only record of implementation sessions. Newest last. Written so a
different agent, or a person, can tell exactly what changed and what did not.

Cross-track 2026-08-15 Method Y session (this checkout, Method X battery
untouched): `docs/plans/agent_session_ledger_20260815.md`.

---

## 2026-08-09 — Corrective programme, Tasks 4–8 and gap closure G1–G7

Agent session. Implemented the checkpoint-staging programme, the BrainIAC
scaffold, then audited the corrective plan line by line and closed the gaps that
audit exposed.

**No training was started. No finalized checkpoint, metric, or manifest row was
modified. The locked v3 YAML was not edited.**

### State on arrival

Tasks 1–3 of the corrective plan were already complete — verified against live
state rather than assumed: manifest reconciled, five-model amendment applied,
fail-closed scheduler and `--mode resume-backbone` present.

### Task 4 — checkpoint staging (the real bottleneck)

`train_3d_mri.py` wrote a full resume payload after *every* optimizer step to a
USB HDD: 533 MB for ResNet18, 1.02 GB for ResNet34. One ResNet18 epoch is 228
steps ≈ 119 GB of writes.

Now: journal every step to H (~79 KB/epoch), stage a full payload to NVMe every
32 steps, mirror to H at epoch/fold boundaries, prune C only after a
hash-verified commit.

| One ResNet18 epoch | Before | After |
|---|---:|---:|
| Full payload writes | 228 | 7 |
| Bytes written to H | ~119 GB | ~79 KB |

Measured **2.5× median speedup** over three repetitions (6.69 s → 2.71 s) against
the required 1.15×. `best_model_state` was lifted out of the repeating payload
into a hash-locked H sidecar written only on validation improvement.

The digest split (`scientific` / `runtime_resume` / `storage_policy`) is a pure
addition: legacy contract bodies hash identically, so all 32 live checkpoints
still resume with zero scientific drift.

New: `src/training/checkpoint_staging.py`, four test suites.
Modified: `scripts/train_3d_mri.py`, `src/training/method_x_resume.py`.

### Tasks 5–7 — BrainIAC scaffold

`research/brainiac_caft_20260808/`, 8 artifacts. No clone, no weights, no
C-side environment. Weight identity and module paths are
`UNRESOLVED_PENDING_VENDOR` — the playbook is explicit that import paths must
not be copied from secondary documents.

New: `src/research/brainiac_provenance.py`, `brainiac_adapter.py`,
`brainiac_caft.py`.

### Gap audit against the corrective plan

Re-read the plan section by section. Six gaps found and closed:

| Gap | Finding | Fix |
|---|---|---|
| G1 | §2 step C staging flags absent from the runner (0 occurrences) | Launch-time overlay, excluded from the stored command and retry digest |
| G2 | §9 requires nine supervision layers; seven existed | Added `checkpoint_integrity` and `fold_finalization`; report now fails closed if any required layer did not run |
| G3 | Six §2 step C contract fields missing from the payload | Added as top-level keys, never inside the digested contract body |
| G4 | §H-00.6 baseline and §4.2 runtime amendment missing | Generated both; verdicts `CLEAN_NO_PROJECT_ARTIFACT_ON_C` and `RUNTIME_ONLY_MIGRATION_SAFE` |
| G5 | Redesigned roadmap not recorded anywhere agents would find it | `docs/plans/method_x_project_roadmap.md` + JSON twin + CLAUDE.md pointer |
| G6 | R0 registry at 18 of 24–30 sources | See "Open items" |

**G1 detail — why an overlay.** The runner stores the exact training command in
the manifest and refuses to resume a row whose stored command differs
(`STORED_TRAINING_COMMAND_SCIENTIFIC_DRIFT`). Appending staging flags to
`build_training_command` would have stranded every resumable checkpoint. They are
applied only where the child is spawned, and a regression test asserts they never
appear in the stored builder or the retry-authorization digest.

**G2 detail.** A layer that never ran cannot be evidence of safety, so
`_supervision_report` now emits `layers_not_evaluated` and blocks on
`REQUIRED_SUPERVISION_LAYER_NOT_EVALUATED`. Early-return paths (storage BLOCK,
unresolvable path) legitimately evaluate a subset, so the check only adds a
blocker when nothing else already blocked.

**G3 detail.** Also added a fail-closed guard: a negative optimizer step, batch
cursor, or checkpoint generation raises rather than being written, because such a
value can only come from a counter bug and would corrupt the replay bound.

### Verification

- 249 tests added across this programme, all passing.
- Full repository suite: `1239 passed, 1 skipped, 5 failed`. All 5 failures are
  pre-existing Method Y/Z text-artifact guards (one CSV dated 2026-07-26) with
  zero references to any module added here.
- All 32 live checkpoints re-verified resumable after every trainer edit:
  `legacy_ok=True`, `sci=0`.
- Manifest unchanged; no `.pt` file has an mtime inside this session.
- C staging root was never created.

### Findings worth carrying forward

**Geometry mismatch is larger than the plan assumed.** The cache is 128³ at
1.5 mm — a 192 mm field of view. A direct 128→96 resize preserves that extent,
giving **2.0 mm** effective spacing, not BrainIAC's declared 1.0 mm. And 96³ at a
literal 1 mm spans 96 mm, which cannot contain an adult brain (~180 mm), so the
upstream figure is ambiguous in the secondary sources and is recorded as
`UNRESOLVED_PENDING_VENDOR`. Registration families also differ (project: SPM
non-linear; BrainIAC: rigid). Recorded in `geometry_gate_manifest.json` as
`GEOMETRY_DOMAIN_SHIFT_DOCUMENTED`, scope `BRAINIAC_ONLY_MAIN_BATTERY_UNAFFECTED`.

**A pre-existing reporting bug was fixed.** `contract_diff` classified
`loader.batch_size`, `num_workers`, and `persistent_workers` as benign runtime
changes. They alter data order and exact-resume semantics, so they are scientific.
Harmless before Task 4 (runtime changes mapped to `NEW_VERSIONED_RUN` anyway) but
would have become a real misclassification once runtime changes became legitimately
resumable.

### Open items

1. **Bitwise equality is synthetic-only.** Proven on the staging harness. The real
   claim needs an actual interrupted-vs-uninterrupted MRI run compared tensor by
   tensor — cannot be closed without running training.
2. **The 2.5× benchmark used a 2 MB payload.** Direction is certain; the figure on
   real 533 MB / 1.02 GB payloads is unverified.
3. **R0 count target met, depth target not.** 24 primary sources, 15
   peer-reviewed. None is full-text reviewed: all 24 are stamped
   `SEARCH_SUMMARY_ONLY`, so status stays `INCOMPLETE_PENDING_R0_COMPLETION`
   with blocker `FULL_TEXT_REVIEW_BELOW_TARGET`. See the G6 entry below.
4. **BrainIAC is unvendored.** Weight identity, module paths, and true input FOV
   remain `UNRESOLVED_PENDING_VENDOR`.
5. **Memory policy unverified this session** — no GPU work was performed.

## G6 — R0 source gap closed to the count target (2026-08-09)

Three searches on the prespecified axes added six genuine primary sources:
**18 → 24 primary, 15 peer-reviewed, 9 preprints.** The count target is met.

`scripts/extend_brainiac_r0_registry.py` is idempotent on source id and can be
re-run after further searching.

### A depth gate was added to prevent a self-inflicted overstatement

Reaching 24 flipped the registry to `COMPLETE`, which was wrong: these sources
were recorded from search results and abstracts, not read. Counting search hits
as reviewed evidence is the easiest way to inflate a scoping review.

`evidence_registry_status` now tracks `screening_stage` per source
(`FULL_TEXT_REVIEWED` / `TITLE_ABSTRACT_SCREENED` / `SEARCH_SUMMARY_ONLY`) and
blocks on `FULL_TEXT_REVIEW_BELOW_TARGET` independently of the count. A source
with no stage is assumed **unread**, never credited. Status is correctly back to
`INCOMPLETE_PENDING_R0_COMPLETION`.

Three tests that encoded "count is the only gate" were updated, and two added:
one proving screened-but-unread cannot complete the review, one proving an
unstamped source is not silently credited. 51 pass.

### Four findings changed the design record

1. **s19 narrows the research gap.** An AJNR 2025 study predicts PET-defined
   amyloid from MRI on ADNI + OASIS3 + **A4**, externally validated on an
   independent Stanford ADRC cohort: T1-only AUC 0.61, T1+FLAIR 0.67 internal /
   0.65 external, MCI 0.71. "PET-free amyloid status in A4" is no longer an open
   gap. The gap must be stated as the five-component combination, and any gain
   argued against this envelope — not against ADNI-only numbers. Recorded as
   `closest_prior_art` in `novelty_matrix.json`.
2. **s23 strengthens the confound-audit component.** Harmonization is still
   evaluated by image similarity or downstream accuracy, not by whether site
   identity stays linearly decodable from embeddings.
3. **s24 constrains the abstention component.** Patient distribution shift breaks
   split-conformal exchangeability, so reported coverage is an internal marginal
   guarantee, not a transportable one.
4. **s21 supports the gated design.** In a same-data head-to-head, a hybrid using
   hippocampal volume beat 3D ViT. Scale alone does not win this endpoint.

Artifacts: `evidence_registry.json`, `novelty_matrix.json`, `review_protocol.md`,
`search_log.md` (wave 2 ledger) in `research/brainiac_caft_20260808/`.

### Still open

`prior_art_exact_match_found` remains `false`, but the honest reading is that the
gap is **narrower than the plan assumed**. If full-text review of s19 shows it
covers more of the five components than its abstract suggests, the downgrade rule
applies and the claim becomes replication/extension.

### Next action

`G-A` in the roadmap: resume ResNet18 (fold 2, optimizer step 1321). Command in
`docs/plans/method_x_project_roadmap.md` §3.


---

## 2026-08-10 — Codex supervisory audit review and defect resolution

Codex audited the work done in the 2026-08-09 session and produced a corrective
plan (A0–A6 + R0/R1 gates). I reviewed that plan against the live codebase,
verified what Codex completed correctly, confirmed which findings are real, and
fixed all defects it identified.

### What Codex did correctly (A0, A1)

- **A0 — artifact freeze.** `PRE_REPAIR_ARTIFACT_FREEZE.json` written. ResNet18
  status restored from `BLOCKED_TRAINING_EXECUTION` to `PAUSED_RESUMABLE_2_OF_5`
  in the manifest via transactional reconciliation.

- **A1 — source-only migration receipts.** Five compatibility receipts written
  with full migration chains linking old checkpoint SHA → old outer-result digest
  → new outer-result digest → new checkpoint SHA, proving `y_true/y_prob/logits`
  are identical and `evaluation_count=1`. The `COMMITTED_OUTER_EVALUATION_INVALID`
  error is resolved. This was real work that I did not do and should have.

### What Codex's plan got wrong

- **"ResNet18 is blocked"** in the audit intro was stale — its own A1 fix had
  already resolved it by the time the plan was written. The live manifest now
  shows `PAUSED_RESUMABLE_2_OF_5`.

- **A2/A3 cadence=1 raise** is a deliberate safety gate, not a defect. The
  trainer raises `STAGED_RECOVERY_REPLAY_NOT_ZERO` until the real-model bitwise
  equivalence test proves replay is safe. This is exactly the plan's own
  requirement (§3, "bitwise equality"), just implemented before the test exists.

- **"Trainer only reads H"** was correct for the pre-staging case (which is the
  only case that runs today) and is the intended behaviour until staging is
  activated.

### Defects I fixed (from Codex's P0/P1 lists)

| # | Defect | Severity | Fix |
|---|---|---|---|
| 1 | Promotion gate passes on NaN/inf/out-of-range | **P0** | Added `_validated_metrics()` guard; non-finite values raise before any comparison |
| 2 | Promotion gate uses raw AUC for confound probe | **P0** | Added `confound_decodability()` = max(AUC, 1-AUC); AUC 0.10 now correctly blocks |
| 3 | Absent evidence defaults to False instead of blocking | **P1** | `None` for bootstrap/coverage/sanity now emits `*_EVIDENCE_MISSING` blocker |
| 4 | Geometry gate `passed=True` with unresolved findings | **P0** | Changed to `status == "PASS" and not self.findings` |
| 5 | Strict-load passes on two empty state dicts | **P0** | Added `STRICT_LOAD_EMPTY_STATE_STATE_DICT` guard |
| 6 | `np.roll` wraps anatomy across volume boundary | **P1** | Replaced with `_shift_no_wrap()` that uses slicing, vacated positions = False |
| 7 | Conformal method misnamed as `label_conditional` | **P1** | Renamed to `split_conformal_marginal`; returns `coverage_guarantee` and `class_counts` |
| 8 | Coverage report `zip` silently truncates mismatched lengths | **P1** | Added length/probability/label validation before comparison |
| 9 | FOV finding over-claims "96 mm too small" | **P1** | Renamed to `DECLARED_TARGET_FOV_UNRESOLVED_SHAPE_NOT_PHYSICAL_EXTENT` |

31 regression tests in `test_brainiac_fail_open_hardening.py` lock in all nine fixes.

### R0 registry overcount

Codex correctly identified that the 24-source count includes duplicates:
- s02 and s19 are the same paper (DOI 10.3174/ajnr.A8899)
- s22 and s23 are connected via the same PMID
- s21 was published July 2024, outside the prespecified 2025-01-01 window

After dedup, the real count is ~21 unique sources (~12 peer-reviewed). The
registry overstates. This is an honest counting error, not a fabrication: each
entry was independently surfaced by search, but the dedup step was missed.

### What remains for Method X to be "ready to train"

Three items block the training commands from being valid:

1. **Dedup the R0 registry.** Merge duplicates, reclassify s21 as background,
   recompute counts. Real state will be ~21 sources, ~12 peer-reviewed, 0
   full-text reviewed — still `INCOMPLETE_PENDING_R0_COMPLETION`.

2. **A5 real-model bitwise equivalence test.** A tiny `nn.Module` + optimizer +
   scheduler run twice (once uninterrupted, once killed/restarted) with tensor
   comparison. The synthetic staging benchmark is not enough; this is the
   plan's own hard gate.

3. **A6 readiness issuance.** Depends on A5. Produces
   `READY_FOR_MAIN_BATTERY_TRAINING.json` with test counts, hashes, and live
   status from the manifest.

None of these is a code defect. They are completion gaps that block readiness
assertion, not training itself.

---

## 2026-08-15 — Pre-training programme closed; GPU training not started

Later the same calendar day as the R0 full-text + official-weight session:

- R0 remains a completed scoping review (29 sources, 25 full-text, novelty
  `NOVELTY_CANDIDATE`). Live roadmap twins were amended to
  `COMPLETE_SCOPING_REVIEW`. A6 `write_bundle` must not be re-run.
- Official `BrainIAC.ckpt` stays on H; vendor-equivalent strict load is now a
  library helper (`validate_vendor_equivalent_strict_load`). Naive MONAI 1.5.2
  `strict=True` is still a documented FAIL.
- Step B future-run YAML added:
  `configs/experiments/method_x_claim_benchmark_v3_five_model_future.yaml`.
  Locked v3 YAML and runner default are unchanged.
- Isolated BrainIAC venv still has no packages; requirements are frozen for R1a.
- Post-training cards stored with programme order
  MX-03 → R1a → R1b → R1c → MX-04 → MX-05 → MX-06 → R2 → R3 → MX-07.
- Method Y PET training was not started and remains blocked.
- Next action: ResNet18 `--mode resume-backbone` from optimizer step 1321.

---

## 2026-08-15 — Programme gap closeout (resume unblocked; GPU not started)

Constructor fix from earlier the same day stands: `MethodX3DClassifier` accepts
trainer kwargs including `gradient_checkpointing`; locked ResNet18 fold-0 keys
still match (`alignment_projection` is bias-free Linear).

**Task 2 evidence after source-only migration.** The failed resume printed
`RESUME_CONTRACT_SOURCE_ONLY_MIGRATED` (`22bd960a…` → `68d94730…`) and rewrote
about 320 bytes of contract metadata on
`models/resnet18_3d/method_x_3d_fold_2_progress.pt`. Weights were not rewritten.
Live file: `byte_size=532607474`, SHA-256
`7040411787a19b1efddf980ef5b3c10a9fcf32623b729130fc2ba0f9d1cd2213`,
`optimizer_step_count=1321`. Autosupervisor still pinned the pre-migration
bytes and raised `TASK2_PROGRESS_BYTES_MISMATCH`.

Repair (append-only; `PRE_REPAIR_ARTIFACT_FREEZE.json` untouched):

- Journal: `journals/resnet18_task2_evidence_after_source_only_migration_20260815.json`
- Only the ResNet18 `resumable_evidence` row in `battery_manifest.json` was
  updated (size / sha / digest `68d94730…`).
- Bind sidecar:
  `models/resnet18_3d/source_only_contract_migrations/22bd960a8c3f43d9_to_68d94730f5aa1041.fold_2.progress_bind.json`
- `reports/BLOCKED_PREFLIGHT.json` and `reports/BLOCKED_AUTOSUPERVISOR.json`
  moved to `archive/superseded_blockers/task2_evidence_20260815/` (not deleted).

**Durable hook for ResNet34.** Source-only migration receipts still list only
FINALIZED folds. The trainer now writes a versioned paused-`progress.pt` bind
sidecar after `SOURCE_ONLY_MIGRATED`. `validate_resumable_checkpoint_evidence`
accepts that sidecar when fold `source_history` hops the evidence digest. The
first ResNet34 resume must not repeat `TASK2_PROGRESS_BYTES_MISMATCH`.

**Architecture persisted** in `docs/plans/method_x_project_roadmap.md` (mermaid
+ battery table + next command + do-not-do) and
`results/method_x_registry/project_roadmap.json` field `architecture`.
`docs/method_x_task_cards.md` intro now says five effective models + MX-02a/b
(bottom overrides unchanged).

**Small guards.** Trainer fail-closes with exit 79 and
`METHOD_X_TF32_OR_COMPILE_FORBIDDEN` if `--allow_tf32` or `--torch_compile` is
set. Defaults stay off and are excluded from the scientific contract.
Programme name `BLOCKED_STORAGE_LOCALITY.json` was never implemented; the live
equivalent is `reports/BLOCKED_PREFLIGHT.json`. Do not create a duplicate file.

**PID 6820.** Not present in live evidence (superseded by 5140 / 33084). PID
16572 was already archived. Closed; no process-list repair remains for those
IDs.

GPU training was not started. Next operator command is the same ResNet18
`resume-backbone` (never `--no_resume`).

---

## 2026-08-15 — Leftover finalized progress bind; repair gate

Operator `resume-backbone` hit `RUN_CONTRACT_MIGRATION_PROGRESS_MARKED_FINALIZED`.
Cause: live ResNet18 (and ResNet34) keep `method_x_3d_fold_{N}_progress.pt` beside
immutable finalized `method_x_3d_fold_{N}.pt`. The new bind loop treated those
leftovers as a paused progress source.

**Supervision first.** Added a 10-layer fail-closed repair gate (Codex-style:
missing layer = BLOCK, skipped pytest = BLOCK, no human override):
`src/training/method_x_repair_gate.py`,
`scripts/run_method_x_repair_gate.py --phase diagnose|authorize-resume`.
This does not replace the training autosupervisor and does not start GPU.

**Fix.** `classify_source_only_progress_bind` ignores leftover finalized-fold
progress (`keep_finalized`, bytes unchanged) and only binds the TRAINING fold
progress. Finalized `.pt` / outer-eval were not rewritten.

**Restore.** `BLOCKED_TRAINING_EXECUTION.json` archived under
`archive/superseded_blockers/leftover_progress_bind_20260815/`; ResNet18 row
returned to `PAUSED_RESUMABLE_2_OF_5`. Journal:
`journals/resnet18_leftover_progress_bind_restore_20260815.json`.

Repair gate `authorize-resume` decision: `AUTHORIZE_RESUME` (all 10 layers PASS).
Report: `reports/REPAIR_GATE_AUTHORIZE_RESUME.json`.

---

## 2026-08-15 — Staged payload still carried `best_model_state_dict`

Operator `resume-backbone` passed leftover-progress bind and constructed the
model, then died on the first optimizer-step staging write:

`BEST_STATE_MUST_NOT_BE_STAGED:best_model_state_dict`

**Cause.** `omit_best_state=True` nulled the tensors but still emitted the
keys. `CheckpointStager.write_full_checkpoint` rejects key *presence*, not a
None value. Overlay cadence=1, so every step hits that check. The wiring test
only asserted that the flag existed.

**Also closed (would have been the next GPU failure).** Task 2 hop validation
required a *direct* evidence→live bind. The next source-only migration writes
previous-live→new-live. Without a unique-successor chain of bind sidecars,
resume would have blocked again with `CURRENT_FOLD_STATE_INVALID` /
`TASK2_PROGRESS_MIGRATION_BIND_MISSING`.

**Fix.** `progress_payload(omit_best_state=True)` drops forbidden keys.
`FORBIDDEN_STAGED_PAYLOAD_KEYS` includes `best_averaged_state_dict`.
`seed_offloaded_best_state_sidecar` copies resumed best weights onto the H
sidecar before any omit-write, so later H mirrors cannot drop `best_epoch`
weights. Task 2 walks the unique successor chain and hashes adjacent binds.

Finalized `.pt` / outer-eval were not rewritten. Fold 2 progress remains at
optimizer step 1321.

**Restore.** Blocker archived under
`archive/superseded_blockers/best_state_staged_keys_20260815/`; ResNet18 row
returned to `PAUSED_RESUMABLE_2_OF_5`. Journal:
`journals/resnet18_best_state_staged_keys_restore_20260815.json`.

---

## 2026-08-15 — Watch inventory race mislabeled as contract mismatch

Operator `resume-backbone` constructed the model, migrated the source-only
digest `f1524bc5…` → `6a30a3b4…`, completed fold 2 epoch 6 (`EPOCH_SAVED`,
val_loss 0.3958), then continued optimizer steps on C to **1534**. The parent
then printed `BLOCKED_RUN_CONTRACT_MISMATCH`. Child logs had only AMP
`FutureWarning`s. Returncode 78 came from the autosupervisor killing a healthy
trainer.

**Cause.** Cadence-1 staging rotates ~400 MB `resume_current.pt` on C every
optimizer step. The 15s watch probe `lstat`/`os.walk` collided with
`os.replace`, reported `C_STAGING_INVENTORY_UNREADABLE`, fail-closed, and
`classify_child_failure` treated any supervisor 78 as
`SCIENTIFIC_CONTRACT_MISMATCH`. This was not scientific drift.

**Fix (source digest unchanged).** Watch defers an inventory-only UNREADABLE
blocker after one retry and does not kill. Quota / reparse / forbidden-file
still kill. `classify_child_failure` maps that race to
`STAGING_INVENTORY_WATCH_RACE` and the scheduler pauses instead of writing a
contract-mismatch blocker. Edited only `scripts/run_method_x_claim_benchmark.py`
and `src/training/storage_locality.py` (neither is in
`method_x_training_source_paths()`), plus tests.

Finalized `.pt` / outer-eval / H `fold_2_progress.pt` were not rewritten.
C staging `6a30a3b4cffbafc7` was not moved. H progress remains stale at
optimizer step 1321; live resume is C at **1534**.

**Restore.** Blocker archived under
`archive/superseded_blockers/staging_inventory_watch_race_20260815/`; ResNet18
row returned to `PAUSED_RESUMABLE_2_OF_5`. Journal:
`journals/resnet18_staging_inventory_watch_race_restore_20260815.json`.

Repair gate `authorize-resume` decision: `AUTHORIZE_RESUME` (all 10 layers PASS).
Report: `reports/REPAIR_GATE_AUTHORIZE_RESUME.json`.

---

## 2026-08-15 — Manifest replace WinError 5 mislabeled as contract mismatch

Operator `resume-backbone` printed `BLOCKED_RUN_CONTRACT_MISMATCH` with no
`EPOCH_SAVED` and no source-only migration. Child stdout/stderr were empty
(0 bytes). Duration 0.25 s. C staging `6a30a3b4cffbafc7` mtimes unchanged at
16:32 UTC, still optimizer step 1534.

**Cause.** `_record_child_pid` (supervisor `on_start`) writes
`battery_manifest.json` via `os.replace` on H. Windows returned
`PermissionError: [WinError 5] Access is denied` on that replace. The
supervisor fail-closed (`SUPERVISOR_START_HOOK_ERROR`) and killed the child
with exit 78. Two classifier bugs then mislabeled it:

1. `classify_child_failure` treated any 78 without a watch-inventory race as
   `SCIENTIFIC_CONTRACT_MISMATCH`.
2. Even a correct `TRANSIENT_IO` label would still have become
   `BLOCK_RUN_CONTRACT` because `classify_training_returncode(78)` is
   lifecycle authority unless the category is scientific.

The previous inventory-race fix was right for that incident and did not apply
here. The shared defect is: supervisor 78 is not a scientific contract.

**Fix (source digest unchanged).** Manifest writes retry WinError 5/32.
The redundant pre-spawn manifest write was removed so start does not replace
the same USB file twice in ~100 ms. Start-hook access-denied is
`TRANSIENT_IO` and the parent retries the spawn. Supervisor infrastructure
78 can no longer become `BLOCK_RUN_CONTRACT`.

Finalized `.pt` / outer-eval / H progress / C staging were not rewritten.

**Restore.** Blocker archived under
`archive/superseded_blockers/manifest_replace_winerror5_20260815/`; ResNet18
row returned to `PAUSED_RESUMABLE_2_OF_5`. Journal:
`journals/resnet18_manifest_replace_winerror5_restore_20260815.json`.

Repair gate `authorize-resume` decision: `AUTHORIZE_RESUME` (all 10 layers PASS).
Report: `reports/REPAIR_GATE_AUTHORIZE_RESUME.json`.

---

## 2026-08-15 later — Method Y defect repair on this checkout (Method X untouched)

A later agent session on the same live branch repaired Method Y v2 pretraining
contracts and demoted a theater Task 4 `COMMITTED` receipt. **This work log
entry records isolation only.** Method X battery status, C staging step 1534,
locked v3 YAML, and trainer sources were not part of that session’s edits.

Do not treat Method Y `training_allowed=false` / Task 4 `AWAITING_AGENT_MULTI_PASS_QC`
as a Method X blocker. Do not start Method Y training from Method X cards.

Full Method Y record: `docs/plans/agent_session_ledger_20260815.md`.
Isolation note: `results/method_x_registry/SESSION_NOTE_METHOD_Y_DEFECT_REPAIR_20260815.json`.

---

## 2026-08-16 — Resolved-path inventory race mislabeled as training execution block

Operator `resume-backbone` for ResNet18 ran fold 2 epochs 7–10 (`EPOCH_SAVED`),
then continued C staging to optimizer step **2420** (generation 1104, payload
`880d4219…`). After ~7669 s the parent wrote `BLOCKED_TRAINING_EXECUTION`.
The next `resume-backbone` refused immediately with
`TARGET_BACKBONE_NOT_RESUMABLE:resnet18_3d:BLOCKED_TRAINING_EXECUTION` and
`training_started: false`. Child logs had only AMP `FutureWarning`s. Returncode
78 came from the autosupervisor killing a healthy trainer.

**Cause.** Cadence-1 was rotating `resume_current.pt` (watch saw the live
`.1104.tmp`). Lexical `storage_locality` **PASS**ed. The immediate resolved-path
revalidation hit `C_STAGING_INVENTORY_UNREADABLE` and wrapped it as
`RESOLVED_PATH_STORAGE_REVALIDATION_FAILED`. The 2026-08-15 watch defer only
inspected top-level `blocking_reasons`, so it did not treat the wrapper as the
already-known inventory race. `classify_child_failure` then labeled
`NON_RETRYABLE_EXECUTION` and the scheduler wrote a live
`BLOCKED_TRAINING_EXECUTION.json`. This was not scientific drift.

**Fix (source digest unchanged).** `effective_watch_inventory_blockers` unwraps
an inventory-only nested `UNREADABLE` under the resolved-path wrapper. Watch
defers that race after one retry; persistent inventory-only UNREADABLE is
`CONTINUE`, not a kill. Nested quota / reparse / forbidden-file still kill.
Edited `src/training/storage_locality.py` and
`scripts/run_method_x_claim_benchmark.py` (neither is in
`method_x_training_source_paths()`), plus tests.

Finalized `.pt` / outer-eval / H `fold_2_progress.pt` / C staging files were
not rewritten. C `resume_current.pt` hashes to the journaled step-2420 payload.
H progress remains stale at optimizer step 1321; H epoch mirror is 2280; live
resume is C at **2420**.

**Restore.** Blockers archived under
`archive/superseded_blockers/resolved_path_inventory_watch_race_20260816/`;
ResNet18 row returned to `PAUSED_RESUMABLE_2_OF_5`. Journal:
`journals/resnet18_resolved_path_inventory_watch_race_restore_20260816.json`.

Repair gate `authorize-resume` decision: `AUTHORIZE_RESUME` (all 10 layers PASS).
Report: `reports/REPAIR_GATE_AUTHORIZE_RESUME.json`.

---

## 2026-08-16 — Per-fold epoch cap (20 × 5 = 100)

Live ResNet18 fold 2 logged `EPOCH_SAVED.epoch = 21` because the trainer treated
locked YAML `epochs: 100` as the per-fold loop bound. The intended protocol is
20 epochs per fold (folds 0–4), 100 passes per model.

Fix: `protocol_epochs_per_fold()`; loop is `range(start_epoch, epochs_per_fold)`.
Run-contract still records `epochs: 100` so live resume digests do not move.
Locked v3 YAML was not edited. Append-only amendment:
`configs/experiments/method_x_claim_benchmark_v3_epochs_per_fold_amendment.yaml`.

The running GPU process will not see this until a controlled stop and resume.
Do not `--no_resume`. ResNet10 is not retrained (already completed under early
stopping below the cap).
# 2026-08-21 — P09 final reconciliation and readiness

- Confirmed no active Method X writer and no battery/reconciliation lock before
  live mutation.
- The first official preflight changed only `battery_manifest.updated_at_utc`,
  moving its SHA from the receipt-bound `ce0f8fc6...` to `df781f61...`. It did
  not start training or touch an outer loader, but the byte drift correctly made
  exact-resume evidence fail closed.
- Added RED→GREEN transaction tests and a pinned recovery path. The drifted
  manifest was archived under
  `journals/preflight_manifest_recovery/afc89cf6.../`; the canonical manifest was
  restored byte-for-byte and an append-only receipt was issued. Finalized
  artifact rewrites and outer-loader invocations were both zero.
- Changed preflight preparation so a valid receipt-bound paused manifest is
  validated on a copy and never receives telemetry writes before exact-resume
  authorization.
- Official preflight then exited 0, all ten autonomous layers were evaluated,
  and the manifest SHA was unchanged before/after.
- Fresh related suite: 791 passed, 0 failed, 0 errors; one nonblocking AMP API
  deprecation warning.
- Final readiness:
  `results/method_x_registry/READY_FOR_MAIN_BATTERY_TRAINING_20260821_FINAL.json`.
  It authorizes only MedNeXt-S fold 1 generation 1396 / optimizer step 1390 /
  cursor 224 / replay 0. ResNet10/18/34/DenseNet121 remain completed; ResNet50
  and EfficientNet-B0 remain excluded.
- H remains project authority; C is allowed only for checkpoint payload staging.
  External evaluation is optional/nonblocking and unavailable internal evidence
  remains `ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION`.

## 2026-08-22 (afternoon) — ZCode supervision session

- Live verification: MedNeXt fold-2 training confirmed active and advancing
  (runner PID lock consistent, journal growing, GPU busy, C free 54.7GB,
  staging 6.4GB <= 20GB quota). The 12:11 crash had been detected and the
  official `--mode run` resume succeeded under the exit-75 contract.
- Storage locality re-verified with an ABSOLUTE output root:
  `STORAGE_LOCALITY_VERIFY_20260822_R2_ABSOLUTE.json` = PASS, zero blockers.
  The 08:14 BLOCK receipt was an invocation artifact (relative output root),
  not a locality violation; both receipts are retained.
- Focused contract tests re-run during live training: 135 passed
  (storage-locality, checkpoint staging backend, battery-state reconciliation,
  runner staging overlay, checkpoint recovery; basetemp on H).
- Hygiene fixes: desktop playbook path documented correctly
  (`PLAYBOOK_PATH_CORRECTION_20260822.json`), colocated
  `models/pretrained/brainiac/PROVENANCE.v1.json`, partial duplicate vendor
  extraction quarantined under
  `vendor/BrainIAC/_quarantine_partial_duplicate_20260822/` with inventory,
  einops installed into the isolated brainiac venv (pytorch-lightning stays
  deferred to post-terminal per the pinned requirements comment), and
  vendor-declared geometry evidence extracted from code
  (`vendor_declared_geometry_evidence_20260822.json`; status stays UNKNOWN,
  fail-closed).
- R0 completion wave ZCODE_R0_FULLTEXT_20260822: 22 evidence-bound full-text
  reviews (parallel subagents, note sha256 verified, review_basis stamped)
  applied by `research/brainiac_caft_20260808/apply_r0_fulltext_20260822.py`
  into `evidence_registry_v3.json`. Registry now 29 sources, 13 peer-reviewed,
  28 full-text reviewed, 0 invalid, 0 pending; s19 (AJNR A8899) honestly kept
  abstract-tier (paywalled). Project validator returns R0 status COMPLETE;
  novelty ceiling remains NOVELTY_CANDIDATE pending final lock; closest prior
  art: s30 (d3+d4, Aβ AUROC 0.79 external) and s19 — neither reaches the
  five-component CAFT combination.
- `r1_readiness_status_20260822.json` issued (R0 blocker resolved; six
  fail-closed blockers remain; execution still HARD_DISABLED) and
  `BRAINIAC_P08_STATUS_20260822.json` rebuilt (next action:
  AWAIT_FIVE_CNN_BATTERY_TERMINAL_THEN_RUN_R1A_GATES).
- Post-training execution memory v3 written:
  `results/method_x_registry/post_training_execution_memory_20260822.json`
  (cards D02/MX-03/R1a/R1b/R1c/MX-04/MX-05/MX-06/T08/MX-07 + watchdog
  contract). A 2-hourly read-only watchdog automation is active (auto-resume
  via the official command only; POST sequence starts with D02 once the
  battery is terminal).

## 2026-08-25 - BrainIAC governance unblock + P003 enablement + strict cross-audit (ox-alpha)

Commits 2c901f8 / e8de978 / this commit. Evidence file:
`.superpowers/sdd/brainiac_caft_training_readiness_20260822/agent-reports/plan-AUDIT-cross-session-20260825T090500Z.md`.

- Governance drift fixed: predecessor binding re-pinned (`e3dd7aaf`/3880B ->
  amendment `6b313303`), registry re-hashed to live notes (`1c45b3f4`, 49
  entries), vendor CRLF->LF restores tree `c26b42ba` (strict_load PASS).
- Battery reached TERMINAL mid-session: manifest `40a4a009`, lock ABSENT.
  Ledger HEAD_DRIFT fixed by slimming terminal battery report to
  `models_digest` (anchor message back under EventLog 32k limit); transition 8
  `860911fc` anchored at OS record 93876. Gate PREPARE_RAW_T1 verified PASS.
- P003 runner enabled end-to-end: preflight PASS; single resumable command is
  `scripts\research\prepare_brainiac_raw_t1_v2.py --mode full --device cuda`
  (exit 75 safe-stop, same-command resume). Recurring stale
  `.method_x_reconciliation.lock` reconciled only after DEAD-PID proof.
- Strict audit fixes (RED->GREEN): real A6 `validate_claim` /
  `claim_from_three_d_aggregate` restored as pure fail-closed predicates
  (POLICY_VERSION a6-callable-1.1.0) replacing my earlier fail-open stub;
  restored lost dirty-era symbols `fit_source_counts`,
  `resolve_method_x_training_device`, `migrate_source_only_resume_artifacts`,
  `inspect_exact_run_contract`/`ensure_exact_run_contract`,
  `method_x_split_contract_digest`; main() now carries the exact-resume
  contract gate flags (`--cache_sha256`, `--spm_snapshot_sha256`,
  `--require_exact_resume_contract`, `--contract_preflight_only`) with
  CONTRACT_PREFLIGHT_PASS read-only path; backbone_factory regains
  `amuse_x_hybrid3d` + `AVAILABLE_BACKBONES`.
- Source digest baseline moves `8c02bd51...` -> `9b0f90a5a14f8764397c6a15ff8016462f0872e2f827b2ec47fe54ab773871b3`.
  Test evidence: lifecycle 9/9, A6+callable+spm 23/23, transformation 6/6
  (brainiac venv), preprocessing_v2 45, anchor 17. No outer/test-subject I/O;
  PET-free boundary untouched.
