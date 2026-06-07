# Method Y PET Manifest Summary

- stage: `method_y_pet_manifest`
- tracer: `fbp`
- source: `project_inventory`
- private_manifest_path_redacted: `data<REDACTED>/<tracer>_private_manifest.csv`
- excluded_manifest_path_redacted: `data<REDACTED>/<tracer>_excluded_manifest.csv`
- n_candidates: `5661`
- counts_by_substudy: `{'A4': 2100, 'LEARN': 789, 'SF': 2772}`
- counts_by_tracer: `{'fbp': 5661}`
- link_summary: `{'catalog_rows': 1699, 'candidate_rows': 5661, 'linked_work_items': 266, 'excluded_rows': 1433, 'one_pet_per_patient': True, 'requested_tier': 'primary_strict', 'linked_tiers': {'primary_strict': 191, 'gold_standard_30': 75}, 'linked_labels': {'1': 232, '0': 34}, 'note': 'Training-ready linking requires private manifest or project resolver; filename fallback is dry-run only.'}`
- linked_gap_tiers: `{'primary_strict': 191, 'gold_standard_30': 75}`
- raw_paths_exported: `False`
- regex_fallback_used: `False`
- project_inventory_used: `True`
- training_ready_manifest: `True`
- warning: `Filename regex fallback is allowed for dry-run only; training-ready PET requires private manifest<REDACTED>`
