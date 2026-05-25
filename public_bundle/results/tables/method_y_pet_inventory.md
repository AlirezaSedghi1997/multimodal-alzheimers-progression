# Method Y PET Inventory/QC Summary

- status: `COMPLETED`
- scope: `aggregate_inventory_qc_sensitivity_only`
- Method X claim unchanged: `True`
- PET as Method X input: `False`
- PET as Method X-Cog input: `False`
- raw identifiers exported: `False`
- raw paths exported: `False`
- subject-level outputs exported: `False`

## Locked Catalog

- rows: `1700`
- patients: `1700`
- label_counts: `{'1': 1163, '0': 537}`
- substudy_counts: `{'A4': 1163, 'LEARN': 537}`
- PET/T1 gap bins: `{'missing': 0, '<=60': 154, '61-90': 112, '91-120': 64, '>120': 1370}`

## PET Image Inventory

| tracer_folder | directory_present | image_file_count | parseable_filename_count | locked_patient_matches |
|---|---:|---:|---:|---:|
| amyloid_fbp | True | 5661 | 2889 | 1698 |
| tau_ftp | True | 1576 | 1572 | 447 |

## Aggregate Sensitivity

- amyloid PET label source rows: `4492`
- locked catalog label rows matched in memory: `1700`
- overall/consensus agreement counts: `{'both_present': 18, 'agree': 16, 'disagree': 2}`
- tau PET source rows: `296679`
- locked catalog patients with tau records matched in memory: `447`

## Boundary

PET image, SUVR, centiloid, amyloid, and tau-derived values remain forbidden as Method X and Method X-Cog model inputs. This artifact is aggregate-only.
