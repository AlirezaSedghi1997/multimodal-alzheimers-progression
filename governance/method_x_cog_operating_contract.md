# Method X-Cog Operating Contract

Method X-Cog is an exploratory extension. It must not alter the Method X claim.

## Primary Boundary

Method X uses only T1 MRI plus age, sex, education_years, apoe4_carrier, and mmse_base.

Method X-Cog may add cognitive and psychological test summaries for an incremental-value analysis.

## Leakage Rules

The following columns or concepts are forbidden as model inputs:

- PET.
- tau.
- SUVR.
- amyloid.
- centiloid.
- scan_date.
- raw image path.
- raw identifiers.

Derived files that mix clinical scores with PET-derived fields must be explicitly allowlisted by column.

## Output Rules

Method X-Cog may write hashed sample-level feature tables and aggregate model summaries only. Public reporting must not include raw identifiers, raw clinical rows, raw paths, subject-level predictions, MRI/PET images, or HDF5 caches.

## Reporting

Reports must state that Method X-Cog is exploratory and that cohort-label confounding remains severe in the locked A4/LEARN catalog.
