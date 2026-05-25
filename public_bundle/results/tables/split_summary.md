# Split Summary

- rows_total: `1700`
- patients_total: `1700`
- external_rows: `340`
- dev_rows: `1360`
- external_label_counts: `{'1': 233, '0': 107}`
- dev_label_counts: `{'1': 930, '0': 430}`
- cohort_label_confound: `True`
- external_holdout_type: `locked_pseudo_external_random_patient_holdout`

## Interpretation

The local catalog has strong cohort-label coupling: A4 is positive and LEARN is negative. The external holdout is therefore pseudo-external and must not be described as independent.
