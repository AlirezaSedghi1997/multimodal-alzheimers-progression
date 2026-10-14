# Research update and reproducibility boundary — 14 October 2026

The [analysis bundle](../analyses/reviewer_sensitivities_20261014/README.md)
is the entry point for all research/statistical changes since the first
scientific review. The original source tree and existing results remain intact.
Only new research code, aggregate findings and supporting documentation are added.

## Scope and evidence

The bundle retains every final aggregate result from the first and second
review analyses byte for byte, then adds completed OOF contrasts, their score
heterogeneity diagnostic, absolute concordance intervals and original training
histories. A separate [pilot record](CNN_JACOBIAN_PILOT.md) supplies compact-CNN
and log-Jacobian code and its no-go decision without numerical pilot outcomes.
The [inventory](RESEARCH_UPDATE_INVENTORY_20261014.json) maps each copied file to
its source hash and distributed hash; machine-path changes are explicit.

The already published historical sample-size, regional, demographic and
noncognitive-PACC work remains available under the statistical supplement.
Old tau intervals are historical: the 40,000-draw estimates in the new bundle
take precedence. No existing unfavorable result has been deleted.

## No-data checks

From the repository root:

```text
python -B check_package.py
python -B analyses/reviewer_sensitivities_20261014/replay_aggregates.py --output OUTSIDE_CHECKOUT/replay
python -B -m pytest tests/test_cnn_jacobian_pilot_v1.py -q -p no:cacheprovider --basetemp OUTSIDE_CHECKOUT/pilot-tests
```

From `analyses/reviewer_sensitivities_20261014/oof_stacking_extension`, run:

```text
python -B -m pytest tests/test_xyz_oof_stacking_v1.py -q -p no:cacheprovider --basetemp OUTSIDE_CHECKOUT/oof-tests
```

Create the external output directory before supplying a child directory to
pytest's `--basetemp` option. Use a fresh child path for each test run.
ANTs/PyTorch are required for
the geometry/image tests; CUDA-specific assertions skip if no GPU is available.
The analysis guide distinguishes synthetic implementation tests, aggregate
replay, earlier participant-level re-execution and model reconstruction.
These checks do not certify data permission, scientific validity or clinical use.

The publication checkout passed all 19 OOF tests and all 9 CNN/Jacobian tests,
including deterministic CUDA backward propagation. The synthetic metric check
also passed. An initial test invocation had temporary-directory setup errors;
after creating the missing parent directory, both complete suites passed without
changing model or statistical logic. The release audit records these checks
separately from the earlier participant-data replay receipts.

## Publication boundary

The destination is the existing repository owner's GitHub repository, which is
public at the time of this update. No repository-visibility change is made.
Files are selected by an allowlist rather than staging the research worktree.
No manuscript or thesis, individual record, linkage key, scan, prediction object,
trained weight, credential, private receipt or pilot performance table is added.
Dataset access requires separate approval; GitHub access does not supply it.
The project has no newly assigned open-source license. Existing source notices
and provider terms remain applicable.
