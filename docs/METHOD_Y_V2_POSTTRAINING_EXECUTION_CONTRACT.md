# Method Y v2 post-training execution contract

Status: `PRESPECIFIED_NOT_EXECUTED`

This is the canonical project-memory contract for every Method Y action after
the pretraining release. It records required inputs, ordering, failure modes,
and claim boundaries. It does **not** report a trained model, outer result,
external result, or positive scientific finding.

## 1. Locked scope

- Amyloid predictors are FBP-derived only; Tau predictors are FTP-derived only.
- The primary endpoint is sustained future CDR progression after the baseline
  PET index. Longitudinal PACC is secondary.
- Primary models are source-regional CoxNet, RSF, and GBSA. Image models are a
  separately gated secondary branch.
- Development uses five outer folds and three inner folds, grouped by subject.
- HPO, transforms, imputation, scaling, harmonization, and feature selection
  are fitted on the applicable training partition only.
- Inner validation selects configurations by integrated Brier score. Outer
  outcomes cannot influence selection, thresholds, features, harmonization,
  calibration choices, or stopping.
- A `COMPLETED` workflow state never implies a passed science gate or a claim.
- No independent human sign-off or annotation is required. Gate promotion uses
  invariant tests, deterministic reconstruction, and an adversarial agent
  review, each bound by SHA-256 evidence. Legal account actions such as
  accepting a DUA remain the data owner's responsibility.

## 2. Entry gate

No training command may pass preflight unless all of these immutable inputs
exist and their byte hashes match:

1. `results/method_y_revival_v2/readiness/PRETRAINING_READINESS_V2.json` with
   `status=READY_FOR_SOURCE_REGIONAL_TRAINING`, `training_allowed=true`, and no
   image, outer, external, or claim permission.
2. `results/method_y_revival_v2/pretraining_contracts/pretraining_validation_state.json`
   with a valid layered-agent review receipt and no training/outer/external
   access recorded.
3. `results/method_y_revival_v2/feature_contracts/source_regional_validation_state.json`
   plus the Amyloid, Tau, and paired A/T feature contracts.
4. Future-only endpoint contracts and private five-outer/three-inner membership
   artifacts referenced by the public split contract.
5. `configs/method_y_training_control_v2.yaml` and the code/runtime/dependency
   digests named by the training-control contract.
6. A process snapshot proving no other writer targets the same versioned
   output root. Method X writers in a different output root are not stopped.

Any hash, seed, membership, endpoint, feature, code, dependency, or model-family
change requires a new versioned output root. It cannot resume under an old
contract.

## 3. Ordered post-training state machine

```text
PRETRAINING_RELEASE_LOCKED
-> INNER_SELECTION_RUNNING
-> INNER_SELECTION_COMPLETE
-> OUTER_WINNERS_LOCKED
-> OUTER_MODELS_CHECKPOINT_LOCKED
-> OUTER_EVALUATION_COMMITTED
-> OOF_METRICS_COMMITTED
-> INCREMENTAL_VALUE_AUDITED
-> SCANNER_SITE_SUBGROUP_AUDITED
-> SECONDARY_PACC_COMMITTED
-> IMAGE_BRANCH_PENDING_OR_COMMITTED
-> EXTERNAL_PENDING_OR_COMMITTED
-> REPORTS_CONSISTENT
-> PUBLIC_BUNDLE_VERIFIED
-> FINALIZED
```

An interruption resumes only from an atomically committed state. An ambiguous
outer-test access without a valid committed result is terminal for that fold;
the test set is not accessed a second time.

## 4. Stage contracts

### Stage A — inner selection and winner lock

Required inputs:

- pretraining readiness receipt;
- branch endpoint, feature, and private split digests;
- candidate registry with equal budgets for CoxNet, RSF, and GBSA;
- code/runtime/dependency digests.

For each outer fold, the controller performs all candidate-by-inner-fold units,
owns fold-local preprocessing and canonical IBS calculation, and atomically
stores the score. The one-standard-error rule is applied only after every unit
is committed. A pre-existing winner is accepted only after recomputing its
summary from the committed units. The winner record names the outer fold,
selected family/configuration, inner mean IBS, standard error, complexity rank,
run-contract digest, and unit-evidence digest.

Outputs:

```text
development/<branch>/outer_<k>/inner_hpo/run_contract.json
development/<branch>/outer_<k>/inner_hpo/unit_results/*.json
locked/<branch>/outer_<k>_winner.json
```

### Stage B — final fit within each outer fold

Prerequisite: the matching outer-fold winner is valid and locked.

The selected pipeline is refit on the complete outer-training membership only.
No outer-test row is used for preprocessing, fitting, early stopping,
calibration, or threshold selection. Checkpoints include estimator and
preprocessor state, completed step/epoch, RNG states where applicable, input
digests, and the immutable run-contract digest. Resume continues after the
last atomically completed step.

Output:

```text
locked/<branch>/outer_<k>_checkpoint/
```

### Stage C — exactly-once outer evaluation

Prerequisite: the outer checkpoint is locked and its hash matches the winner
and run contract.

Each fold follows:

```text
CHECKPOINT_LOCKED
-> OUTER_EVAL_IN_PROGRESS
-> OUTER_EVAL_COMMITTED
-> FINALIZED
```

The committed result must contain prediction/result hashes, event count,
follow-up support, at-risk counts, metric status, and test-access counter. If
one of five folds is missing or failed, the branch status is
`FAILED_INCOMPLETE_OUTER_CV`; remaining folds are not presented as a complete
five-fold result.

Outputs:

```text
outer_evaluation/<branch>/outer_<k>/outer_result.json
outer_evaluation/<branch>/aggregate.json
```

### Stage D — locked OOF metrics and calibration

Prerequisite: all five outer results are committed.

Compute from paired out-of-fold predictions:

- Uno C-index using a censoring distribution fitted on training data only;
- integrated Brier score on common survival support;
- cumulative/dynamic AUC at prespecified estimable horizons;
- calibration slope, intercept, and plots;
- event and at-risk counts for every horizon.

A horizon is `NOT_ESTIMABLE` rather than silently removed when it is outside
common support or has fewer than the prespecified at-risk count. No horizon is
chosen because it looks favorable on outer data.

### Stage E — incremental-value gates

Prerequisite: matched models share the same outer folds, preprocessing policy,
HPO budget, and OOF membership.

Prespecified comparisons are:

```text
Amyloid: M0 clinical+cognition vs M1 M0+Centiloid
          and M1 vs M2 M1+prespecified spatial extent, if that feature is valid
Tau:     T0 clinical+cognition vs T1 T0+source-regional Tau
Paired:  P0 clinical+cognition vs P1 P0+Amyloid vs P2 P1+Tau
```

Use 2,000 subject-level paired bootstrap replicates. A positive incremental
gate requires the upper 95% CI of `delta_IBS = IBS(augmented)-IBS(base)` below
zero, the lower 95% CI of `delta_Uno_C >= 0`, all outer folds complete, and no
failed scanner sensitivity gate. Otherwise the result remains exploratory or
negative. The locked v1 negative Tau-beyond-Amyloid finding is preserved unless
this prespecified v2 analysis supplies valid new evidence.

### Stage F — scanner, site, and subgroup audits

Prerequisite: locked OOF predictions; scanner/site are audit labels, not
predictors in the primary model.

Run performance by manufacturer, leave-manufacturer-out, leave-site-out where
support is sufficient, UNKNOWN-excluded sensitivity, and subgroup estimates
with confidence intervals for sex, prespecified age strata, treatment arm, and
cohort/transport role. GE and Philips are never pooled to manufacture support.
Tau and paired A/T retain a claim blocker while a second manufacturer lacks at
least 30 subjects and 10 events. Small subgroups are reported as
`NOT_ESTIMABLE`, not interpreted from unstable point estimates.

### Stage G — longitudinal PACC secondary analysis

Prerequisite: the independently rebuilt PACC contract, exact measurement dates,
locked PET feature set, and completed primary outer evaluation.

Fit the prespecified mixed-effects model with subject random intercept and
slope:

```text
PACC ~ time + PET_feature + time*PET_feature
       + age + sex + education
       + treatment_arm + time*treatment_arm
       + (1 + time | subject)
```

The PET term of interest is `time*PET_feature`; raw baseline-to-follow-up
difference is not substituted. Any changed feature set is versioned and does
not retroactively modify the primary survival result.

### Stage H — image branch and XAI

Prerequisite: branch-specific image readiness, not source-regional readiness.

- Amyloid image work requires licensed quantitative masks, documented
  Centiloid/SUVR space and scale, and image-to-source-regional validation.
- Tau image work requires validated FTP identity, reference-region/scale and
  CenTauR mapping, plus a valid PET-to-T1-to-MNI transform chain for every
  included scan.
- Image survival uses a censoring-aware survival head; a binary event CNN is
  not the primary survival model.

XAI reconstructs the exact architecture from the locked contract, loads the
checkpoint with `strict=True`, and uses a licensed atlas in the matching space.
Grad-CAM++ where architecturally valid, Integrated Gradients, and occlusion are
checked by weight randomization, label permutation, fold stability, and
volume-matched random-ROI occlusion. A failed sanity/fidelity test closes XAI
publication. Public outputs are aggregate maps and region intervals only.

### Stage I — external validation

Prerequisite: DUA/data availability, locked data dictionary and variable/region
mapping, locked feature intersection, and a final internal model that has never
seen external outcomes.

Evaluate the locked model once on SCAN/NACC, ADNI, or OASIS-3 without refit,
feature reselection, threshold reselection, or outcome-based harmonization.
Initial external metrics use the internal preprocessing/calibration pipeline.
Any external recalibration is secondary, explicitly labeled, and cannot replace
the unrecalibrated transport result. Until these conditions hold, status is
`EXTERNAL_PENDING` and external generalization claims are forbidden.

### Stage J — reports, privacy, and claims

Prerequisite: all available committed stages and a consistency audit.

Regenerate survival metrics, calibration, incremental value, scanner/site,
subgroup, PACC, image-to-regional, Tau quantitative, XAI, external, negative
results, and claim-policy reports. Align the manuscript with TRIPOD+AI,
PROBAST+AI, and CLAIM where applicable.

The public bundle contains aggregate/hash-level artifacts only. Recursive scans
must reject raw IDs/paths, row-level predictions, private memberships,
checkpoints, individual images, and unredistributable atlas files. A positive
claim is impossible when its branch science gate is failed, blocked, or lacks
external evidence required by the wording.

## 5. Claim boundaries

- Internal Amyloid wording may concern incremental prognostic information for
  future progression in the prespecified internal A4 setting only.
- Tau wording is positive only if the matched incremental-value gate passes;
  otherwise: "Tau-PET incremental value was not demonstrated under the
  prespecified evaluation protocol."
- Paired A/T with 91 events remains complexity-limited and exploratory unless
  the prespecified power/complexity gate and all evaluation gates pass.
- Method Y does not establish causal effects, diagnose Alzheimer disease, or
  justify external generalization without locked external validation.
- Image/XAI attribution describes model-associated regions, not causal regions
  or the anatomical origin of Alzheimer disease.

## 6. Resume and terminal rules

- Re-running the unchanged launcher resumes the last committed
  stage/fold/unit; it does not delete or restart completed work.
- Any contract mismatch stops before model or outcome access.
- Early terminal folds are not reopened.
- A committed outer result is verified and reused; it is never recomputed.
- An ambiguous uncommitted outer access is terminal and requires a new
  prespecified analysis version, not a retry in the same result set.
- Missing external data ends locally as `EXTERNAL_PENDING`, not pipeline
  failure and not external validation success.

## 7. Required final layered review

Before finalization, three independent machine-verifiable layers must pass:

1. invariant and regression test suite;
2. deterministic artifact/hash reconstruction and privacy/consistency scan;
3. adversarial agent review targeting leakage, optional stopping, repeated
   test access, scanner shortcuts, forged claims, and resume mismatches.

No stage in this contract relies on an independent human reviewer. Any later
clinical deployment study, regulatory use, or care decision is outside this
research contract and would require a separate governance process.
