from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ExternalDatasetAdapter(ABC):
    """Abstract base class for external dataset adapters.

    Provides a unified interface for downloading, processing, and caching
    external neuroimaging datasets for model validation.
    """

    @abstractmethod
    def download_metadata(self, output_dir: Path) -> pd.DataFrame:
        """Download subject metadata (IDs, labels, demographics)."""

    @abstractmethod
    def get_labels(self, metadata: pd.DataFrame) -> dict[str, int]:
        """Convert dataset-specific labels to binary AD/NC format.

        Returns:
            Dict mapping subject_id to binary label (1=AD/AB+, 0=NC/AB-).
        """

    @abstractmethod
    def get_tabular_features(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Extract tabular features matching project format.

        Must return columns in order: age, education_years, apoe4_carrier, mmse_base, sex_binary
        """

    def validate_cache(self, cache_path: Path) -> dict[str, Any]:
        """Validate that an HDF5 cache matches project format."""
        import h5py

        report = {"valid": True, "errors": [], "warnings": []}
        with h5py.File(cache_path, "r") as h5:
            required_keys = {"image", "label", "sample_hash", "tabular", "status"}
            missing = required_keys - set(h5.keys())
            if missing:
                report["valid"] = False
                report["errors"].append(f"Missing keys: {missing}")

            if "image" in h5:
                shape = h5["image"].shape
                if len(shape) != 5 or shape[1:] != (1, 96, 96, 96):
                    report["warnings"].append(f"Image shape {shape} expected (N,1,96,96,96)")

            if "tabular" in h5:
                tab_shape = h5["tabular"].shape
                if len(tab_shape) != 2 or tab_shape[1] != 5:
                    report["warnings"].append(f"Tabular shape {tab_shape} expected (N,5)")

            if "label" in h5:
                labels = h5["label"][:]
                unique_labels = set(labels.tolist())
                if not unique_labels.issubset({0, 1}):
                    report["warnings"].append(f"Labels {unique_labels} expected binary {{0,1}}")

        return report


class ADNIAdapter(ExternalDatasetAdapter):
    """Adapter for ADNI (Alzheimer's Disease Neuroimaging Initiative).

    Access: Free registration at https://adni.loni.usc.edu/
    Data: T1 MRI (NIfTI), demographics, cognitive scores
    Labels: AD, MCI, NC
    Sample size: ~2000 subjects with T1 MRI
    """

    ADNI_DIAGNOSIS_MAP = {
        "AD": 1,
        "Dementia": 1,
        "Alzheimers Disease": 1,
        "NC": 0,
        "NL": 0,
        "Normal": 0,
        "Normal Cognitively Impaired": 0,
        "MCI": 0,
    }

    ADNI_COLUMN_MAP = {
        "Age": "age",
        "PTEDUCAT": "education_years",
        "APOE4": "apoe4_carrier",
        "MMSE": "mmse_base",
        "Sex": "sex_binary",
    }

    def download_metadata(self, output_dir: Path) -> pd.DataFrame:
        """Download ADNI metadata.

        Note: Requires ADNI LONI IDA credentials.
        For offline testing, provide pre-downloaded CSV at output_dir/adni_metadata.csv
        """
        csv_path = output_dir / "adni_metadata.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"ADNI metadata not found at {csv_path}. "
            "Download from https://adni.loni.usc.edu/ and place as adni_metadata.csv"
        )

    def get_labels(self, metadata: pd.DataFrame) -> dict[str, int]:
        labels = {}
        diag_col = None
        for col in ["DX_bl", "Diagnosis", "diagnosis", "DX"]:
            if col in metadata.columns:
                diag_col = col
                break
        if diag_col is None:
            raise ValueError("No diagnosis column found in ADNI metadata")

        id_col = None
        for col in ["PTID", "Subject", "subject_id", "RID"]:
            if col in metadata.columns:
                id_col = col
                break
        if id_col is None:
            raise ValueError("No subject ID column found in ADNI metadata")

        for _, row in metadata.iterrows():
            subject_id = str(row[id_col])
            diagnosis = str(row[diag_col]).strip()
            labels[subject_id] = self.AJNI_DIAGNOSIS_MAP.get(diagnosis, 0)
        return labels

    def get_tabular_features(self, metadata: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        for adni_col, proj_col in self.AJNI_COLUMN_MAP.items():
            if adni_col in metadata.columns:
                features[proj_col] = metadata[adni_col]
            else:
                features[proj_col] = 0.0
        if "sex_binary" in features.columns:
            features["sex_binary"] = features["sex_binary"].map({"M": 1, "F": 0, "Male": 1, "Female": 0}).fillna(0)
        if "apoe4_carrier" in features.columns:
            features["apoe4_carrier"] = (features["apoe4_carrier"] > 0).astype(float)
        return features[["age", "education_years", "apoe4_carrier", "mmse_base", "sex_binary"]]


class OASISAdapter(ExternalDatasetAdapter):
    """Adapter for OASIS-3 (Open Access Series of Imaging Studies).

    Access: Free registration at https://www.oasis-brains.org/
    Data: T1 MRI (NIfTI), PET (some), cognitive scores
    Labels: AD, MCI, NC
    Sample size: ~1000 subjects
    """

    OASIS_DIAGNOSIS_MAP = {
        "AD": 1,
        "Alzheimers Disease": 1,
        "Dementia": 1,
        "NL": 0,
        "Normal": 0,
        "NC": 0,
        "MCI": 0,
    }

    def download_metadata(self, output_dir: Path) -> pd.DataFrame:
        csv_path = output_dir / "oasis_metadata.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"OASIS metadata not found at {csv_path}. "
            "Download from https://www.oasis-brains.org/ and place as oasis_metadata.csv"
        )

    def get_labels(self, metadata: pd.DataFrame) -> dict[str, int]:
        labels = {}
        diag_col = None
        for col in ["diagnosis", "DX", "Diagnosis", "diagnosis_bl"]:
            if col in metadata.columns:
                diag_col = col
                break
        if diag_col is None:
            raise ValueError("No diagnosis column found in OASIS metadata")

        id_col = None
        for col in ["subject_id", "Subject", "PTID", "RID"]:
            if col in metadata.columns:
                id_col = col
                break
        if id_col is None:
            raise ValueError("No subject ID column found in OASIS metadata")

        for _, row in metadata.iterrows():
            subject_id = str(row[id_col])
            diagnosis = str(row[diag_col]).strip()
            labels[subject_id] = self.OASIS_DIAGNOSIS_MAP.get(diagnosis, 0)
        return labels

    def get_tabular_features(self, metadata: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        col_map = {"Age": "age", "age": "age", "PTEDUCAT": "education_years", "education": "education_years",
                    "APOE4": "apoe4_carrier", "apoe4": "apoe4_carrier", "MMSE": "mmse_base", "mmse": "mmse_base",
                    "Sex": "sex_binary", "sex": "sex_binary"}
        for src, dst in col_map.items():
            if src in metadata.columns:
                features[dst] = metadata[src]
            elif dst not in features.columns:
                features[dst] = 0.0
        if "sex_binary" in features.columns:
            features["sex_binary"] = features["sex_binary"].map({"M": 1, "F": 0, "Male": 1, "Female": 0}).fillna(0)
        if "apoe4_carrier" in features.columns:
            features["apoe4_carrier"] = (features["apoe4_carrier"] > 0).astype(float)
        return features[["age", "education_years", "apoe4_carrier", "mmse_base", "sex_binary"]]


ADAPTERS = {
    "adni": ADNIAdapter,
    "oasis": OASISAdapter,
}


def get_adapter(name: str) -> ExternalDatasetAdapter:
    adapter_cls = ADAPTERS.get(name.lower())
    if adapter_cls is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(ADAPTERS.keys())}")
    return adapter_cls()
