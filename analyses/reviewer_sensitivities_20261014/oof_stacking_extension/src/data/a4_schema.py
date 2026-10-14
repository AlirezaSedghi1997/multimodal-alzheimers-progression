from __future__ import annotations

from pathlib import Path


REQUIRED_RELATIVE_FILES = {
    "ab_label": Path("Clinical") / "External Data" / "imaging_PET_VA.csv",
    "suvr_amyloid": Path("Clinical") / "External Data" / "imaging_SUVR_amyloid.csv",
    "tau": Path("Clinical") / "External Data" / "imaging_SUVR_tau.csv",
    "volumetric_mri": Path("Clinical") / "External Data" / "imaging_volumetric_mri.csv",
    "subjinfo": Path("Clinical") / "Derived Data" / "SUBJINFO.csv",
    "visits": Path("Clinical") / "Derived Data" / "SV.csv",
}

REQUIRED_FILE_NAMES = {key: rel.name.lower() for key, rel in REQUIRED_RELATIVE_FILES.items()}

AB_LABEL_COLUMN = "overall_score"
PET_DAY_COLUMN = "scan_date_DAYS_T0"
T1_DAY_COLUMN = "Date_DAYS_T0"
TAU_TARGET_CANDIDATES = ("suvr_cer", "suvr_persi", "suvr_crus")
TABULAR_MAP = {
    "age": "AGEYR",
    "sex": "SEX",
    "education_years": "EDCCNTU",
    "apoe4_carrier": "APOEGN",
    "mmse_base": "MMSETSV6",
    "mmse_base_fallback": "MMSETSV1",
}
T1_NAME_PATTERN = r"^(?P<substudy>A4|LEARN)_MR_T1_(?P<bid>B[0-9]+)_(?P<scan>[0-9]{3})\.nii(\.gz)?$"


def resolve_data_root(candidate: str | None = None) -> Path:
    import os
    supplied = candidate or os.environ.get("A4_ROOT")
    if not supplied:
        raise ValueError("Provide a data root or set A4_ROOT; controlled data are not bundled")
    root = Path(supplied).expanduser()
    if not root.is_dir():
        raise FileNotFoundError("The authorized data root does not exist")
    return root


def resolve_required_file(root: Path, key: str) -> Path:
    expected = root / REQUIRED_RELATIVE_FILES[key]
    if expected.exists():
        return expected
    expected_name = REQUIRED_FILE_NAMES[key]
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower() == expected_name:
                return path
    return expected
