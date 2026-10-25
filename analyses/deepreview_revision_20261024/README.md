# Post-hoc review analyses — 24 October 2026

This bundle adds executed internal analyses prompted by the supplied independent
review. It preserves the original primary estimates. The specifications were
dated before the corresponding new executions, after historical development;
they are not a preregistered confirmatory study or independent validation.

## What was executed

| Record | Question and result | Scope |
|---|---|---|
| A01 | Exact original 40000-draw tau reconstruction, same-estimable-draw IBS and all-draw undefined-C policies | K=3 Cox ranking conclusions persisted; broad K=40 tabular conclusions depended on fold aggregation |
| A02 | Paired raw pooled-OOF tau C increments, alongside IBS | C increments 0.0730 and 0.0746; adjusted intervals positive, IBS uncertain |
| A03 | Training-only within-expert centering/scaling before pooled OOF heads | 180 heads, frozen FP32 experts, no established MRI/PET probability-accuracy gain |
| A04 | Nested plasma/tau comparison in the same 220 participants | 60 total / 26 three-year events; no established superiority or equivalence; only 25085/40000 all-fold C-estimable draws |
| A05 | Actual original-head EDF and nested component-head penalty selection | 360 hash-verified original heads; all 45 augmented MRI heads chose alpha=10; uncertain MRI increment |
| A06 | Dated PACC denominators and CDR/tau timing | Ten excluded people had one eligible PACC score; retained 1049/8557; tau lead 0–77 days |
| A07 | Bounded PET source-provenance audit | Manifest/cache schemas and eight retained headers; source-space masks and quantitative voxel units unestablished |
| A08 | Actual image trainer on synthetic 128-cubed inputs and partition assertions | Four optimizer updates, validation, checkpoint/reload and inference; 90 original aggregate partition assertions |
| A09 | Four fixed log-Jacobian summaries in all 1059 participants | Higher IBS versus clinical-plus-amyloid; uncertain addition beyond regional volumes |

The deformation-summary analysis is distinct from the historical compact
CNN/Jacobian pilot, whose no-go decision is preserved. It is not a full
Jacobian-channel CNN, native-space representation test or pretrained encoder.

## Files and safe inspection

- `aggregate/`: public aggregate results and verification records. No participant
  rows, individual predictions, scans, membership digests, fitted weights or
  deformation fields are released. The PET header description field was removed.
- `executed_source/`: exact copies of the scripts and dated specifications used
  in the authorized Windows workspace. These are provenance snapshots, not
  plug-and-play scripts for an arbitrary data tree.
- `manifest.json`: hashes of every file in this bundle except itself.
- `verify_aggregate.py`: standard-library-only integrity, strict JSON, privacy
  pattern and limited numerical-consistency checks. It never imports a research
  runner or reads restricted study data.

From the repository root:

```bash
python -B analyses/deepreview_revision_20261024/verify_aggregate.py
python -B check_package.py
```

Success means package/aggregate consistency only. It does not recompute clinical
results, establish valid bootstrap coverage, validate every reference inference,
or grant restricted-data access.

## Authorized scientific reproduction

The exact executed sources reference `H:\New project` and the original V7 ESM_3
workspace. The required selected-analysis helper and original tau bootstrap
record are also preserved in `../reviewer_sensitivities_20261014/` under
`revision_analyses/` and `aggregate_results/`; the original image-training
snapshot is there under `source_snapshot/`. The repository root supplies the
other original `scripts/xyz_*` modules. Review imports and remap these source
roots deliberately before running. Several audit scripts perform work at import
time. Do not run them merely to inspect this release.

An authorized reconstruction needs the original dated endpoint/fold contracts,
the complete frozen FP32 expert predictions/checkpoints, reference censoring
data, approved A4 source tables and caches, and—for A09—the retained forward
ANTs fields, template geometry and existing approximate atlas bridge. Public
aggregate assertions cannot replace private membership. Keep private output
roots outside the repository and do not bypass provenance/hash gates.

The environment used Python 3.11.15, NumPy 2.3.5, SciPy 1.15.3,
scikit-learn 1.8.0, scikit-survival 0.27.0 and PyTorch 2.5.1+cu121,
with ANTsPy/nibabel for A09 and an RTX 4060 Laptop GPU for the synthetic trainer.
These describe the executed environment, not a completed fresh-machine rebuild.

The main analysis entry actions are `tau`, `raw`, `overlap`, `harmonized` and
`tune`; the deformation, timing/provenance, original-head EDF and synthetic
scripts are separate. The A04 amendment fixes its explicit Brier implementation
on the original time grid rather than changing the horizon when a convenience
API rejects a small validation fold. See the exact source and plans.

## Statistical boundaries

Intervals are conditional on the recorded fits and censoring distributions.
Corrections are per metric/family (K=2, 3 or 4 for the stated new comparisons),
not a correction for every historical choice. Original C inference requires
every fold to be estimable. Zero increments, renormalization and pair weighting
are separate sensitivity policies; pair weighting sums only within-fold IPCW
pairs and changes the aggregation estimand. No cross-fold pairs were introduced.

Within-expert score standardization used each expert's disjoint training meta
rows without outcomes and locked the transform for outer-test prediction.
Nested tuning was confined to a frozen expert's wholly unseen meta pool;
ordinary inner CV on the pooled OOF matrix was not used.

The A09 adverse result does not establish absence of structural information or
identify why the original CNN increment was weak. The approximate atlas bridge
and four-summary compression remain limitations. Anterior parahippocampal is
not relabeled entorhinal. PET calibration and background-only NaN recovery are
not claimed. External validation and full A4 image-training reproduction remain
uncompleted. No manuscript submission, author/journal approval or new license
is implied by this repository update.
