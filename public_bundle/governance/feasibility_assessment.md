# Feasibility Assessment

## Current Assessment

The proposal is technically feasible if WSL2, the `amyloid-tau` Conda environment, and CUDA access to RTX 4060 8GB are available. The primary scientific objective is data-feasible after using `SV.csv` visit timing for T1 alignment.

## Evidence From Local Data Layout

- Required AB label file exists.
- Required tau SUVR file exists.
- Required volumetric MRI metadata file exists.
- Required `SUBJINFO.csv` and `SV.csv` files exist.
- T1 directory contains thousands of NIfTI files.
- Hashed master catalog contains 1700 rows: 1163 AB-positive A4 rows and 537 AB-negative LEARN rows.
- 330 rows are within 120 days after SV-based T1 timing: 288 positive and 42 negative.
- Tau branch has 1274 T1-aligned scans within 120 days and is feasible in FULL mode.

## Main Feasibility Risks

- Scientific performance is not guaranteed; Gate 1 and Gate 2 depend on true signal quality.
- Some T1 filenames may not match the canonical parser and must be reported through QC aggregate counts.
- MRI preprocessing and model training time may exceed the nominal schedule if external drive throughput is low.
- RTX 4060 8GB requires strict batch size and resolution limits.
- The 120-day binary subset is imbalanced, so Gate 1 and Gate 2 performance are uncertain.

## Decision

Proceed with implementation. Freeze the timing hierarchy before split locking and Gate 1.
