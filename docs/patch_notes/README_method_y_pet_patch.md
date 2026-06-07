# Method Y PET Patch Pack

This patch pack contains proposed files to move Method Y-PET/MM from design-only status toward a controlled Phase-2 preprocessing stage without changing Method X.

Files:

```text
scripts/preprocess_pet.py
scripts/method_y_gap_filter.py
docs/method_y_preregistration.md
configs/model/method_y_gap_cohort_eval_policy.yaml
governance/method_y_pet_phase2_gate_policy.md
tests/test_pet_preprocess_contract.py
```

Recommended copy target: repository root.

Suggested first checks:

```bash
python -m py_compile scripts/preprocess_pet.py scripts/method_y_gap_filter.py
pytest tests/test_pet_preprocess_contract.py -q
python scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv
python scripts/preprocess_pet.py --data_root "$DATA_ROOT" --tracer fbp --dry_run
```

Boundary:

- PET remains forbidden in Method X.
- Method Y remains exploratory.
- Public outputs must be aggregate-only.
