# Git Publish Safety Policy

Before every GitHub push:

1. Run `git status -sb`.
2. Review the changed files before staging.
3. Stage only explicit intended files with `git add <path>`.
4. Do not use `git add .` unless the whole worktree has been reviewed and is known to be safe.
5. Before commit, run a sensitive-file check against staged paths.
6. Never stage raw data, protected medical images, HDF5 caches, checkpoints, logs, subject-level outputs, or `.agent` runtime state.

Required staged-path check:

git diff --cached --name-only | Select-String -Pattern '(^data/|^results/|^\.agent/|\.nii$|\.nii\.gz$|\.h5$|\.hdf5$|\.dcm$|\.pt$|\.pth$|\.ckpt$|\.xlsx$|\.xls$|master_catalog\.csv)'

The check must return no output before `git commit` and `git push`.
