# Pooled OOF stacking extension — completed, exploratory

This exact source snapshot accompanies the completed 180-head experiment. It reuses frozen FP32 CNN features; it does not retrain CNNs. The primary mixed-precision analysis and its source snapshot remain unchanged in the parent archive.

Reproduction requires the authorized research-tree inputs, not supplied here. Place this source closure in an isolated reproduction copy of that tree, preserving the documented `results/xyz_fusion_v2`, `results/xyz_fp32_spatial_sensitivity_v1` and restricted tau-frame ancestry. Existing original outputs must remain read-only. Run `python scripts/run_xyz_oof_stacking_v1.py --mode all`. Use a fresh output root beginning with `results/xyz_oof_stacking_`, configured in a new config file, to avoid overwriting historical output. This packaged runner resolves the research root from its own location. Config and source hashes bind each new run.

Run unit and synthetic pipeline tests: `python -m pytest tests/test_xyz_oof_stacking_v1.py -q` from this extension. The aggregate JSONs in the parent `aggregate_results` folder are sufficient to inspect all seven contrasts without controlled data. `python oof_distribution_diagnostic.py --repo AUTHORIZED_ROOT --output LOCAL_AGGREGATE.json` reproduces the training-score variance diagnostic; it never refits models or reads outer outcomes for selection.

All seven adjusted IBS intervals cross zero. Raw-score between-expert heterogeneity limits interpretation of the pooling design. This is not independent clinical validation or an equivalence result. No participant rows, identifiers, scan paths, images, trained weights, signatures or tokens are distributed.
