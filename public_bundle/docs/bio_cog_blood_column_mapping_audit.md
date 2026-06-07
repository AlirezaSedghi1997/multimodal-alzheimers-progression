# Bio-Cog / Method Z — Blood Column Mapping Audit

- status: `MAPPING_AUDIT_COMPLETED`
- data root: `H:\A4 Learn`

Empty NfL/pTau181/Ab42_40_ratio columns in the locked catalog are explained by LBTESTCD alias mismatch (NF-L, TPP181, AMYLB42/40, BP42/BP40), not missing source data.

## Roche LBTESTCD (top)

| code | rows |
|---|---:|
| GFAP | 2345 |
| AMYLB40 | 2345 |
| AMYLB42 | 2345 |
| APOE4 | 2345 |
| NF-L | 2345 |
| TPP181 | 1693 |

## Feature mapping mismatches

- **NfL**: legacy filter matched=False, observed via aliases=2341, status=`MISMATCH`
- **pTau181**: legacy filter matched=False, observed via aliases=1693, status=`MISMATCH`
- **Ab42**: legacy filter matched=False, observed via aliases=11785, status=`MISMATCH`
- **Ab40**: legacy filter matched=False, observed via aliases=11763, status=`MISMATCH`

## Recommended fixes

- Strip LBTESTCD whitespace before matching in build_branch_catalogs.py.
- Map Roche NF-L -> NfL and TPP181 -> pTau181 (not NFL/PTAU181 substring filters).
- Populate Ab42/Ab40 from Roche AMYLB42/AMYLB40; fallback AB_Test BP42/BP40.
- Re-run build_branch_catalogs.py then audit_bio_cog_method_z_readiness.py.
