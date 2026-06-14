# Extended Modality Inventory

Data root hash: `dr_d0dc25f21ce4e709`

## Imaging modalities

| Modality | NIfTI count | Branch | Method X input |
|----------|------------|--------|----------------|
| flair | 14020 | mri_roi_wmh_reads | no |
| dwi | 662 | exploratory_dwi | no |
| fmri_rest | 0 | exploratory_connectivity | no |
| t2 | 0 | exploratory_t2 | no |
| t2star | 0 | mri_roi_wmh_reads | no |

## Clinical extensions

- **imaging_retinal.csv** (498 rows): available → exploratory_retinal
- **imaging_MRI_reads.csv** (1797 rows): integrated_tabular → mri_roi_wmh_reads
- **imaging_FLAIR_WMH_QC.csv** (6793 rows): integrated_tabular → mri_roi_wmh_reads
- **cogstate_battery.csv** (278212 rows): integrated_cognitive → method_x_cog
- **cogstate_cpath.csv** (801134 rows): available → method_x_cog
- **cogstate_macq.csv** (192097 rows): available → method_x_cog
- **cdr.csv** (0 rows): feasibility → clinical_endpoint
- **PACC.csv** (0 rows): feasibility → clinical_endpoint
- **biomarker_pTau217.csv** (5813 rows): integrated_blood → bio_cog_method_z
- **biomarker_Plasma_Roche_Results.csv** (13418 rows): integrated_blood → bio_cog_method_z
- **biomarker_AB_Test.csv** (31480 rows): integrated_blood → bio_cog_method_z

## Recommended integrations

1. **flair**: T1+FLAIR WMH sensitivity ablation (tabular WMH already integrated)
2. **biomarker blood panels**: Re-run build_branch_catalogs with fixed Roche aliases; Method Z ablation
3. **cdr.csv + PACC.csv**: Clinical endpoint GNOVA surrogate (Phase H)
4. **dwi**: Exploratory FA/MD extraction for subset with paired T1
5. **fmri_rest**: Defer — high compute, no amyloid-status literature support at this scale
