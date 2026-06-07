from __future__ import annotations

from pathlib import Path

LEAKY_MRI_SURROGATE_REL = Path("data/processed/quarantine/mri_features_leaky_surrogate_negative_control.csv")
LEGACY_LEAKY_MRI_REL = Path("data/processed/mri_features.csv")
QUARANTINE_README_REL = Path("data/processed/quarantine/README.md")


def resolve_leaky_mri_surrogate_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(".")
    quarantine = root / LEAKY_MRI_SURROGATE_REL
    legacy = root / LEGACY_LEAKY_MRI_REL
    if quarantine.exists():
        return quarantine
    return legacy


def ensure_quarantine_readme(project_root: Path | None = None) -> None:
    root = project_root or Path(".")
    readme = root / QUARANTINE_README_REL
    readme.parent.mkdir(parents=True, exist_ok=True)
    if readme.exists():
        return
    readme.write_text(
        "# Quarantined artifacts\n\n"
        "Files here are **negative-control / supplementary only** and must not enter Method X-Strict headline models.\n\n"
        "- `mri_features_leaky_surrogate_negative_control.csv`: retired intensity/shape surrogates (cohort-leakage inflation control).\n",
        encoding="utf-8",
    )
