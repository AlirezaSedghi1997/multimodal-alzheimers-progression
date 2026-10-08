# Evidence guide

Every aggregate file has a stable source ID, source SHA256, scientific role and
an explicit projection flag. `result_lineage.json` maps it to its original artifact.
Private fields and potential row arrays were omitted. These files are **not**
claimed to be byte-identical originals and must not be used to resume a study run.
Omission does not mean zero observations or a missing scientific effect.

Use the complete matched comparison, its population, direction, confidence
interval, multiplicity policy and estimability status. Lower IBS and higher Uno
C-index have different directions. The bootstrap conditions on fitted models;
it is not uncertainty for the entire retraining/model-selection procedure.
Do not compare model metrics from different populations as an incremental gain.

Current limits include cohort-label confounding for X, quantitative PET provenance
limits, observational/internal selection, low support in tau and four-way subsets,
incomplete attribution controls and no external cohort. Execution completion is not
evidence of a positive effect, causal localization or clinical utility.

The aggregate projections support audit of reported summaries. Re-estimation of
metrics requires authorized individual predictions, outcomes, folds and censoring
references listed only in the supervisors' restricted-input inventory.
