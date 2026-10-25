# Reproduction guide

## Level 1 — inspect this release

Run `python -B check_package.py` with Python 3.11. It uses only the standard library.
Inspect the source/result inventories and compare individual aggregate files with
their lineage record. Tests under `tests/` are selected regressions, not proof of
a complete clean-machine scientific reproduction.

## Level 2 — reproduce statistics

Authorized researchers need the original locked predictions, matched endpoint and
fold membership, censoring references, immutable model/selection contracts and
repair amendments. None is replaced by an aggregate table. Preserve all failed,
negative and not-estimable analyses. The corrected post-fusion route is
`scripts/run_xyz_postfusion_final_v1.ps1`; default mode is preflight. Do not run it
against a production tree merely to inspect a package.

## Level 3 — reconstruct from source data

Use access granted by the data provider, obtain the documented atlas/template
assets under their terms, restore the required historical cache/contract lineage,
and reproduce preprocessing before nested training. `docs/active_routes.json`
identifies the actual entrypoints; it is not an all-stages launch command.

The historical Windows runners are deliberately machine/root-bound. Their source
is preserved rather than cosmetically rewritten. Exact resume requires the original
contract, software/runtime, protected input hashes and recorded location. A fresh
installation or different location must use a reviewed new run contract and output
root; copying a config and deleting a lock is not a migration procedure.

`requirements.txt` is the current direct dependency specification;
`environment/installed_versions.json` is the observed full environment, **not** a
hash-locked wheel archive or a verified installation on another machine. MATLAB/SPM,
ANTs/ANTsPy, FreeSurfer and template availability must be checked for the selected
preprocessing route. Do not download or redistribute these assets without terms.

No new environment was installed and no model was rerun while producing this
package. A clean-machine end-to-end restore and current vulnerability triage remain
release checks. Do not automatically upgrade dependencies of a frozen study.

## Statistical supplement

The [11 October supplement](STATISTICAL_SUPPLEMENT.md) has explicit input and private-output arguments. It reuses original locked runs; it is separate from the original package-assembly operation described above. Participant-level inputs are not bundled.


## Post-hoc review extension — 24 October 2026

See [the extension guide](../analyses/deepreview_revision_20261024/README.md) for the exact executed-source roots, prerequisites, estimands and safe offline verifier. It preserves environment-specific runners without claiming arbitrary clean-machine reproduction. The bounded synthetic trainer execution and numerical reconciliation do not replace a complete A4 rebuild.
