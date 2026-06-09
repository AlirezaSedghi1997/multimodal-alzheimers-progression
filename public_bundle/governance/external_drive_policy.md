# External Drive Policy

## Purpose

The A4 Learn dataset is stored outside the repository and must be treated as read-only.

## Rules

- Do not delete, move, rename, or rewrite files under `<LOCAL_PATH_REDACTED> Learn`.
- Do not copy raw MRI/PET images into the repository.
- Do not include raw CSV contents in reports.
- Processed outputs must contain only aggregate values, hashes, or derived tensors in gitignored cache locations.

## Blocking Conditions

- Missing data root.
- Missing required source files.
- T1 file count is zero.
- Required AB label column is missing.

