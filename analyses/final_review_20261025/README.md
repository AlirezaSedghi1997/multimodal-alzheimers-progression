# Fixed-prediction review addendum — 25 October 2026

This append-only addendum retains all original primary estimates and frozen
models. New specifications were recorded before these executions, after the
historical outcome-informed development. These are post-hoc internal checks,
not preregistered confirmation, independent external validation or deployment.

| Record | Executed check and main implication |
|---|---|
| A10 | Day 730–1095.75 IBS on all 20 original matched contrasts; 40,000 paired within-fold draws, separate K=20 family. Every adjusted interval includes zero. Day 730 survival was interpolated from the frozen grid; the primary window is unchanged. |
| A11 | Reconstructed training-KM subset null IBS and unrounded IPA. Tau Cox/forest/MRI IPA was 7.34→12.51/4.41→7.06/6.08→11.65 percent; descriptive ratios do not replace paired uncertainty. |
| A12 | 90 original component preprocessing pipelines already include StandardScaler. A preceding affine transform canceled to within 2.632e-12. This does not apply to pooling different experts. |
| Concordance support | Pair-weighted tau-Cox/MRI and common-sample contrasts remained estimable in 40,000/40,000 draws. The220-person comparison omitted 37.3% under the original all-fold condition. Pair weighting changes the aggregation estimand; it is not proven unbiased or conservative. |
| Sex diagnostics | Existing MRI/PET subgroup contrasts and nominal conditional intervals are exposed. They do not establish an interaction, equality, fairness or transportability. |

The joint standardized-and-safely-tuned pooled head was not evaluated. Ordinary
inner folds on pooled OOF rows can inherit validation-outcome information via
other experts. No full Jacobian-channel CNN, pretrained/native-space experiment,
PET scan recovery or head-size explanation was added. Coverage and shrinkage
mechanisms remain literature-informed hypotheses requiring direct tests.

## Verification and reconstruction

Run `python -B verify_aggregate.py` in this directory. The standard-library
checker verifies hashes, strict JSON, aggregate identities and reported scope.
It does not import the executed analysis runner, recompute participant results
or certify statistical validity.

`executed_source/review_analyses.py` is the exact executed provenance snapshot.
It reads the authorized original `H:\New project` data tree, prior dated
analysis workspaces and original V7 ESM_3 precision records. The repository's
24 October and 14 October addenda retain the earlier source dependencies. Review
and deliberately remap paths in a separate authorized reconstruction workspace.
The available commands are `late`, `nulls`, `support` and `invariance`; do not
import or execute this runner to inspect public results. It creates private
input receipts when run. The public bundle omits those receipts, participant
rows, predictions, memberships, scans and fitted weights.

The late-window calculation was reconciled against 400 installed-library calls
(maximum absolute error 5.55e-17). This numerical check is distinct from
independent end-to-end image-training reproduction, which remains unverified.
New sources were checked by actual primary full text; their narrow scopes and
the limits of the 42-source identity refresh are recorded in the aggregate
reference summary. No Google service endorsement, author signature or journal
approval is claimed.
