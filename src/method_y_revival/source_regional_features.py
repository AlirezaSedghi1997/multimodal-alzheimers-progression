"""Outcome-blind, privacy-safe contracts for Method Y source-regional features.

This module intentionally contracts *documented source tables*, not image
voxels or resampled caches.  It never publishes a subject-level matrix.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .atomic_io import atomic_write_bytes, atomic_write_json
from .contracts import sha256_mapping


class SourceRegionalFeatureError(ValueError):
    """Raised when a source-regional feature contract is scientifically unsafe."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_KEYS = frozenset(
    {"subject_key", "patient_hash", "bid", "source_subject_id", "raw_path", "scan_key", "session_key"}
)
_FORBIDDEN_TOKENS = (
    "event", "outcome", "followup", "future", "cdr", "pacc", "diagnos",
    "amyloid_positive", "positivity", "label", "target", "image", "cache", "voxel",
)
_REQUIRED_ROW_KEYS = ("subject_key", "visit_code", "scan_date_days_t0", "scan_analyzed")
TAU_PERSI_REGION_ALLOWLIST_V1 = (
    "Amygdala_L", "Amygdala_R", "Angular_L", "Angular_R", "Calcarine_L", "Calcarine_R", "Caudate_L", "Caudate_R", "Cingulum_Ant_L", "Cingulum_Ant_R", "Cingulum_Mid_L", "Cingulum_Mid_R", "Cingulum_Post_L", "Cingulum_Post_R", "Cuneus_L", "Cuneus_R", "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R", "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R", "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R", "Frontal_Mid_L", "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_L_2", "Frontal_Mid_Orb_R", "Frontal_Mid_Orb_R_2", "Frontal_Mid_R", "Frontal_Sup_L", "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R", "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R", "Frontal_Sup_R", "Fusiform_L", "Fusiform_L_ant", "Fusiform_L_post", "Fusiform_R", "Fusiform_R_ant", "Fusiform_R_post", "Heschl_L", "Heschl_R", "Hippocampus_L", "Hippocampus_L_ant", "Hippocampus_L_post", "Hippocampus_R", "Hippocampus_R_ant", "Hippocampus_R_post", "Insula_L", "Insula_R", "Lingual_L", "Lingual_R", "Occipital_Inf_L", "Occipital_Inf_R", "Occipital_Mid_L", "Occipital_Mid_R", "Occipital_Sup_L", "Occipital_Sup_R", "Olfactory_L", "Olfactory_R", "Pallidum_L", "Pallidum_R", "ParaHippocampus_L", "ParaHippocampus_R", "Paracentral_Lobule_L", "Paracentral_Lobule_R", "Parahippocampus_L_ant", "Parahippocampus_L_post", "Parahippocampus_R_ant", "Parahippocampus_R_post", "Parietal_Inf_L", "Parietal_Inf_R", "Parietal_Sup_L", "Parietal_Sup_R", "Postcentral_L", "Postcentral_R", "Precentral_L", "Precentral_R", "Precuneus_L", "Precuneus_R", "Putamen_L", "Putamen_R", "Rectus_L", "Rectus_R", "Rolandic_Oper_L", "Rolandic_Oper_R", "Supp_Motor_Area_L", "Supp_Motor_Area_R", "SupraMarginal_L", "SupraMarginal_R", "Temporal_Inf_L", "Temporal_Inf_R", "Temporal_Mid_L", "Temporal_Mid_R", "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R", "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R", "Temporal_Sup_L", "Temporal_Sup_R", "Thalamus_L", "Thalamus_R",
)
TAU_PERSI_EXCLUDED_VOCABULARY_V1 = (
    "Accumbens_area_lh_VOI", "Accumbens_area_rh_VOI", "Amygdala_lh_VOI", "Amygdala_rh_VOI", "BrainStem_VOI_Opened_seSphere1", "Caudate_lh_VOI", "Caudate_rh_VOI", "DELTA_Z 50PCT N141", "EarlyTau_VOI", "Hippocampus_lh_VOI", "Hippocampus_rh_VOI", "MUBADA Mask", "Pallidum_lh_VOI", "Pallidum_rh_VOI", "Putamen_lh_VOI", "Putamen_rh_VOI", "Thalamus_Proper_lh_VOI", "Thalamus_Proper_rh_VOI", "WMref", "bankssts_lh_VOI", "bankssts_rh_VOI", "caudalanteriorcingulate_lh_VOI", "caudalanteriorcingulate_rh_VOI", "caudalmiddlefrontal_lh_VOI", "caudalmiddlefrontal_rh_VOI", "cerebellum_crus_AVID", "choroid_plexus_lh_VOI", "choroid_plexus_rh_VOI", "cuneus_lh_VOI", "cuneus_rh_VOI", "entorhinal_lh_VOI", "entorhinal_rh_VOI", "frontalpole_lh_VOI", "frontalpole_rh_VOI", "fusiform_lh_VOI", "fusiform_rh_VOI", "inferiorparietal_lh_VOI", "inferiorparietal_rh_VOI", "inferiortemporal_lh_VOI", "inferiortemporal_rh_VOI", "insula_lh_VOI", "insula_rh_VOI", "isthmuscingulate_lh_VOI", "isthmuscingulate_rh_VOI", "lateraloccipital_lh_VOI", "lateraloccipital_rh_VOI", "lateralorbitofrontal_lh_VOI", "lateralorbitofrontal_rh_VOI", "lingual_lh_VOI", "lingual_rh_VOI", "medialorbitofrontal_lh_VOI", "medialorbitofrontal_rh_VOI", "middletemporal_lh_VOI", "middletemporal_rh_VOI", "paracentral_lh_VOI", "paracentral_rh_VOI", "parahippocampal_lh_VOI", "parahippocampal_rh_VOI", "parsopercularis_lh_VOI", "parsopercularis_rh_VOI", "parsorbitalis_lh_VOI", "parsorbitalis_rh_VOI", "parstriangularis_lh_VOI", "parstriangularis_rh_VOI", "pericalcarine_lh_VOI", "pericalcarine_rh_VOI", "postcentral_lh_VOI", "postcentral_rh_VOI", "posteriorcingulate_lh_VOI", "posteriorcingulate_rh_VOI", "precentral_lh_VOI", "precentral_rh_VOI", "precuneus_lh_VOI", "precuneus_rh_VOI", "rostralanteriorcingulate_lh_VOI", "rostralanteriorcingulate_rh_VOI", "rostralmiddlefrontal_lh_VOI", "rostralmiddlefrontal_rh_VOI", "superiorfrontal_lh_VOI", "superiorfrontal_rh_VOI", "superiorparietal_lh_VOI", "superiorparietal_rh_VOI", "superiortemporal_lh_VOI", "superiortemporal_rh_VOI", "supramarginal_lh_VOI", "supramarginal_rh_VOI", "temporalpole_lh_VOI", "temporalpole_rh_VOI", "transversetemporal_lh_VOI", "transversetemporal_rh_VOI",
)
TAU_PERSI_PRIMARY_OFFTARGET_V1 = frozenset({"Caudate_L", "Caudate_R", "Pallidum_L", "Pallidum_R", "Putamen_L", "Putamen_R", "Thalamus_L", "Thalamus_R"})
TAU_PERSI_PRIMARY_ALLOWLIST_V1 = tuple(region for region in TAU_PERSI_REGION_ALLOWLIST_V1 if region not in TAU_PERSI_PRIMARY_OFFTARGET_V1)


def _digest_rows(frame: pd.DataFrame, columns: list[str]) -> str:
    """Digest selected private rows without emitting their identifiers or values."""

    identity = ["subject_key", "visit_code", "scan_date_days_t0"] + (["scan_number"] if "scan_number" in frame.columns else [])
    canonical = frame.loc[:, [*identity, *columns]].copy()
    canonical = canonical.sort_values("subject_key", kind="stable")
    return hashlib.sha256(canonical.to_json(orient="records", date_format="iso", double_precision=15).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value.casefold()) is None:
        raise SourceRegionalFeatureError(f"{name}_SHA256_REQUIRED")
    return value.casefold()


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(provenance, Mapping):
        raise SourceRegionalFeatureError("SOURCE_PROVENANCE_REQUIRED")
    basename = provenance.get("source_basename")
    if not isinstance(basename, str) or not basename or Path(basename).name != basename:
        raise SourceRegionalFeatureError("SOURCE_BASENAME_REQUIRED")
    return {"source_basename": basename, "source_sha256": _require_sha256(provenance.get("source_sha256"), "SOURCE")}


def _analyzed(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().casefold() in {"yes", "true", "1"}) or value == 1


def _select_one_baseline_row(rows: pd.DataFrame, feature_columns: list[str], *, baseline_visit_code: int) -> pd.DataFrame:
    missing = [name for name in _REQUIRED_ROW_KEYS if name not in rows.columns]
    if missing:
        raise SourceRegionalFeatureError(f"BASELINE_COLUMNS_REQUIRED:{','.join(missing)}")
    frame = rows.copy()
    frame = frame.loc[frame["scan_analyzed"].map(_analyzed)].copy()
    frame["visit_code"] = pd.to_numeric(frame["visit_code"], errors="coerce")
    frame["scan_date_days_t0"] = pd.to_numeric(frame["scan_date_days_t0"], errors="coerce")
    frame = frame.loc[frame["visit_code"].eq(baseline_visit_code) & frame["scan_date_days_t0"].notna()].copy()
    if frame.empty:
        raise SourceRegionalFeatureError("NO_EXACT_PRESPECIFIED_BASELINE_ROWS")
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # The exact visit is mandatory.  Within it the earliest documented scan
    # date is outcome-blind; a same-date duplicate is ambiguous, never guessed.
    selected: list[pd.DataFrame] = []
    for _, group in frame.groupby("subject_key", sort=True, dropna=False):
        if group["subject_key"].isna().any() or not isinstance(group["subject_key"].iloc[0], str):
            raise SourceRegionalFeatureError("SUBJECT_KEY_REQUIRED")
        earliest = group["scan_date_days_t0"].min()
        candidates = group.loc[group["scan_date_days_t0"].eq(earliest)]
        if len(candidates) != 1:
            raise SourceRegionalFeatureError("AMBIGUOUS_BASELINE_ROW")
        selected.append(candidates)
    return pd.concat(selected, ignore_index=True).sort_values("subject_key", kind="stable").reset_index(drop=True)


def _reject_forbidden_features(columns: list[str]) -> None:
    for column in columns:
        token = str(column).casefold()
        if any(forbidden in token for forbidden in _FORBIDDEN_TOKENS):
            raise SourceRegionalFeatureError(f"FORBIDDEN_SOURCE_FEATURE:{column}")


def _amyloid_columns(rows: pd.DataFrame) -> tuple[list[str], str]:
    allowed = [name for name in ("AMYLCENT", "global_centiloid") if name in rows.columns]
    if len(allowed) != 1:
        raise SourceRegionalFeatureError("AMYLOID_CONTINUOUS_CENTILOID_REQUIRED")
    return allowed, "SOURCE_CONTINUOUS_CENTILOID"


def _tau_columns(rows: pd.DataFrame, provenance: Mapping[str, Any]) -> tuple[list[str], str]:
    reference = provenance.get("reference_region")
    quantitative_contract_id = provenance.get("quantitative_contract_id")
    if not isinstance(reference, str) or not reference or not isinstance(quantitative_contract_id, str) or not quantitative_contract_id:
        raise SourceRegionalFeatureError("TAU_SOURCE_REFERENCE_CONTRACT_REQUIRED")
    dictionary = provenance.get("reference_dictionary_basename")
    if not isinstance(dictionary, str) or not dictionary or Path(dictionary).name != dictionary:
        raise SourceRegionalFeatureError("TAU_REFERENCE_DICTIONARY_REQUIRED")
    _require_sha256(provenance.get("reference_dictionary_sha256"), "TAU_REFERENCE_DICTIONARY")
    features = sorted(name for name in rows.columns if name.startswith("bi_") and name != reference)
    if not features:
        raise SourceRegionalFeatureError("TAU_SOURCE_REGIONAL_SUVR_REQUIRED")
    if any("centaur" in name.casefold() for name in features):
        raise SourceRegionalFeatureError("TAU_CENTAUR_RELABELING_FORBIDDEN")
    return features, "SOURCE_REGIONAL_SUVR"


def _pivot_tau_long_rows(rows: pd.DataFrame, provenance: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Pivot only an explicit, prespecified source region allow-list.

    The long A4 table uses a literal ``brain_region`` name and ``suvr_persi``;
    selecting by string similarity is deliberately unsupported.
    """
    required = {"brain_region", "suvr_persi", "scan_number", *_REQUIRED_ROW_KEYS}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise SourceRegionalFeatureError(f"TAU_LONG_SCHEMA_REQUIRED:{','.join(missing)}")
    _tau_columns(pd.DataFrame({"bi_placeholder": [1]}), provenance)
    allowed = provenance.get("allowed_region_names")
    if not isinstance(allowed, list) or not allowed or any(not isinstance(value, str) or not value for value in allowed):
        raise SourceRegionalFeatureError("TAU_EXACT_ALLOWED_REGION_LIST_REQUIRED")
    if len(set(allowed)) != len(allowed):
        raise SourceRegionalFeatureError("TAU_ALLOWED_REGION_LIST_DUPLICATE")
    observed = set(rows["brain_region"].dropna().astype(str))
    unknown = set(allowed).difference(observed)
    if unknown:
        raise SourceRegionalFeatureError("TAU_PRESPECIFIED_REGION_MISSING")
    identity = [*_REQUIRED_ROW_KEYS, "scan_number"]
    base = rows.loc[rows["brain_region"].isin(allowed), identity + ["brain_region", "suvr_persi"]].copy()
    duplicate = base.duplicated(identity + ["brain_region"], keep=False)
    if duplicate.any():
        raise SourceRegionalFeatureError("TAU_DUPLICATE_SOURCE_REGION_ROW")
    wide = base.pivot(index=identity, columns="brain_region", values="suvr_persi").reset_index()
    features = sorted(allowed)
    return wide, features


def build_feature_contract(
    *,
    branch: str,
    rows: pd.DataFrame,
    source_provenance: Mapping[str, Any],
    endpoint_contract_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return a public contract and private selected matrix for local computation.

    The returned frame is intentionally the only subject-level object and callers
    must keep it outside tracked/public paths.  The contract itself is aggregate.
    """

    if branch not in {"amyloid_source_regional", "tau_source_regional"}:
        raise SourceRegionalFeatureError(f"UNSUPPORTED_SOURCE_REGIONAL_BRANCH:{branch}")
    provenance = _validate_provenance(source_provenance)
    baseline_visit_code = source_provenance.get("baseline_visit_code")
    if not isinstance(baseline_visit_code, int):
        raise SourceRegionalFeatureError("PRESPECIFIED_BASELINE_VISIT_CODE_REQUIRED")
    endpoint_digest = _require_sha256(endpoint_contract_sha256, "ENDPOINT_CONTRACT")
    if branch == "amyloid_source_regional":
        features, semantics = _amyloid_columns(rows)
        tracer = "FBP"
        units = "Centiloid"
    else:
        if {"brain_region", "suvr_persi"}.issubset(rows.columns):
            rows, features = _pivot_tau_long_rows(rows, source_provenance)
            semantics = "SOURCE_REGIONAL_SUVR"
        else:
            features, semantics = _tau_columns(rows, source_provenance)
        tracer = "FTP"
        units = "SUVR"
    _reject_forbidden_features(features)
    selected = _select_one_baseline_row(rows, features, baseline_visit_code=baseline_visit_code)
    missing = {column: int(selected[column].isna().sum()) for column in features}
    feature_digest = _digest_rows(selected, features)
    contract: dict[str, Any] = {
        "schema_version": "1.0",
        "branch": branch,
        "status": "LOCKED",
        "locked": True,
        "tracer": tracer,
        "feature_semantics": semantics,
        "feature_columns": features,
        "units": units,
        "reference_region": source_provenance.get("reference_region") if branch.startswith("tau") else "documented_source_centiloid",
        "quantitative_contract_id": source_provenance.get("quantitative_contract_id") if branch.startswith("tau") else "documented_source_centiloid",
        "reference_dictionary": ({"basename": source_provenance.get("reference_dictionary_basename"), "sha256": source_provenance.get("reference_dictionary_sha256")} if branch.startswith("tau") else None),
        "excluded_region_policy": source_provenance.get("excluded_region_policy") if branch.startswith("tau") else None,
        "baseline_selection": {
            "outcome_blind": True,
            "required_visit_code": baseline_visit_code,
            "date_rule": "earliest_scan_date_days_t0_within_exact_visit",
            "duplicate_rule": "same_subject_same_visit_same_earliest_date_fails_closed",
            "analyzed_rule": "scan_analyzed must be Yes/true/1",
        },
        "fold_local_processing": {
            "inner_hpo_imputation": "fit_on_each_inner_train_only",
            "inner_hpo_scaling": "fit_on_each_inner_train_only",
            "feature_selection": "fit_on_inner_train_only",
            "inner_hpo_harmonization": "fit_on_each_inner_train_only_if_prespecified",
            "post_winner_refit": "fit_on_entire_outer_train_only",
        },
        "forbidden_dependencies": ["same_pet_derived_target", "fuzzy_roi_mapping", "image_voxels", "image_cache", "image_scale", "future_outcomes"],
        "source": provenance,
        "endpoint_contract_sha256": endpoint_digest,
        "selected_subject_count": int(len(selected)),
        "missingness_by_feature": missing,
        "baseline_row_digest": _digest_rows(selected, features),
        "feature_digest": feature_digest,
    }
    contract["contract_sha256"] = sha256_mapping(contract)
    return contract, selected


def build_subject_level_centiloid_contract(
    *, rows: pd.DataFrame, source_provenance: Mapping[str, Any], endpoint_contract_sha256: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Contract an exact one-row-per-subject AMYLCENT table without inventing PET dates.

    ``SUBJINFO`` is a baseline subject table rather than a scan table.  Its
    time-zero is therefore bound to (and supplied solely by) the locked future
    endpoint; no synthetic visit code or date is created.
    """
    provenance = _validate_provenance(source_provenance)
    endpoint_digest = _require_sha256(endpoint_contract_sha256, "ENDPOINT_CONTRACT")
    required = {"subject_key", "AMYLCENT"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise SourceRegionalFeatureError(f"AMYLOID_SUBJECT_SOURCE_SCHEMA_REQUIRED:{','.join(missing)}")
    frame = rows.loc[:, ["subject_key", "AMYLCENT"]].copy()
    if frame["subject_key"].isna().any() or frame["subject_key"].duplicated().any():
        raise SourceRegionalFeatureError("AMBIGUOUS_SUBJECT_LEVEL_CENTILOID_ROW")
    frame["AMYLCENT"] = pd.to_numeric(frame["AMYLCENT"], errors="coerce")
    if frame.empty:
        raise SourceRegionalFeatureError("NO_ENDPOINT_LINKED_CENTILOID_ROWS")
    digest = hashlib.sha256(frame.sort_values("subject_key").to_json(orient="records", double_precision=15).encode("utf-8")).hexdigest()
    contract = {
        "schema_version": "1.0", "branch": "amyloid_source_regional", "status": "LOCKED", "locked": True,
        "tracer": "FBP", "feature_semantics": "SOURCE_CONTINUOUS_CENTILOID", "feature_columns": ["AMYLCENT"],
        "units": "Centiloid", "reference_region": "documented_source_centiloid", "quantitative_contract_id": "documented_source_centiloid",
        "baseline_selection": {"outcome_blind": True, "source_granularity": "one_documented_subject_level_row", "duplicate_rule": "duplicate_subject_fails_closed", "time_zero_rule": "bound_only_to_locked_endpoint_contract"},
        "fold_local_processing": {"inner_hpo_imputation": "fit_on_each_inner_train_only", "inner_hpo_scaling": "fit_on_each_inner_train_only", "feature_selection": "fit_on_inner_train_only", "inner_hpo_harmonization": "fit_on_each_inner_train_only_if_prespecified", "post_winner_refit": "fit_on_entire_outer_train_only"},
        "forbidden_dependencies": ["same_pet_derived_target", "fuzzy_roi_mapping", "image_voxels", "image_cache", "image_scale", "future_outcomes"],
        "source": provenance, "endpoint_contract_sha256": endpoint_digest, "selected_subject_count": int(len(frame)),
        "missingness_by_feature": {"AMYLCENT": int(frame["AMYLCENT"].isna().sum())}, "baseline_row_digest": digest, "feature_digest": digest,
    }
    contract["contract_sha256"] = sha256_mapping(contract)
    return contract, frame


def build_paired_feature_contract(
    *,
    amyloid_contract: Mapping[str, Any],
    tau_contract: Mapping[str, Any],
    paired_endpoint_contract_sha256: str,
    negative_result_reference: Mapping[str, Any],
    paired_subject_count: int,
    paired_date_window_days: int,
    paired_membership_digest: str,
    paired_membership_rule: str = "exact_private_hash_join_of_locked_baselines_and_locked_paired_endpoint_date_window",
) -> dict[str, Any]:
    """Bind two already-locked source contracts without reinterpreting v1."""

    for name, contract in (("AMYLOID", amyloid_contract), ("TAU", tau_contract)):
        if contract.get("locked") is not True:
            raise SourceRegionalFeatureError(f"{name}_CONTRACT_NOT_LOCKED")
        _require_sha256(contract.get("feature_digest"), f"{name}_FEATURE")
        _require_sha256(contract.get("baseline_row_digest"), f"{name}_BASELINE")
    artifact = negative_result_reference.get("artifact_basename")
    if not isinstance(artifact, str) or Path(artifact).name != artifact:
        raise SourceRegionalFeatureError("NEGATIVE_RESULT_BASENAME_REQUIRED")
    negative_sha = _require_sha256(negative_result_reference.get("sha256"), "NEGATIVE_RESULT")
    conclusion = negative_result_reference.get("conclusion")
    if conclusion != "TAU_INCREMENTAL_VALUE_NOT_DEMONSTRATED":
        raise SourceRegionalFeatureError("NEGATIVE_RESULT_CONCLUSION_REQUIRED")
    conclusion_sha = _require_sha256(negative_result_reference.get("conclusion_sha256"), "NEGATIVE_RESULT_CONCLUSION")
    if conclusion_sha != hashlib.sha256(str(conclusion).encode("utf-8")).hexdigest():
        raise SourceRegionalFeatureError("NEGATIVE_RESULT_CONCLUSION_DIGEST_MISMATCH")
    input_evidence_sha = _require_sha256(negative_result_reference.get("input_evidence_sha256"), "NEGATIVE_RESULT_INPUT_EVIDENCE")
    if paired_subject_count < 1 or paired_date_window_days < 0:
        raise SourceRegionalFeatureError("PAIRED_COUNTS_INVALID")
    membership_digest = _require_sha256(paired_membership_digest, "PAIRED_MEMBERSHIP")
    if not isinstance(paired_membership_rule, str) or not paired_membership_rule:
        raise SourceRegionalFeatureError("PAIRED_MEMBERSHIP_RULE_REQUIRED")
    payload = {
        "schema_version": "1.0",
        "branch": "paired_source_regional",
        "status": "LOCKED",
        "locked": True,
        "join_rule": "exact_join_of_locked_subject_level_baseline_rows_private_only",
        "paired_subject_count": int(paired_subject_count),
        "paired_membership_digest": membership_digest,
        "paired_date_window_days": int(paired_date_window_days),
        "paired_membership_rule": paired_membership_rule,
        "amyloid_feature_digest": str(amyloid_contract["feature_digest"]),
        "tau_feature_digest": str(tau_contract["feature_digest"]),
        "paired_endpoint_contract_sha256": _require_sha256(paired_endpoint_contract_sha256, "PAIRED_ENDPOINT_CONTRACT"),
        "negative_result_reference": {"artifact_basename": artifact, "sha256": negative_sha, "input_evidence_sha256": input_evidence_sha, "conclusion": conclusion, "conclusion_sha256": conclusion_sha},
        "negative_result_preserved": True,
        "forbidden_dependencies": ["future_outcomes", "fuzzy_roi_mapping", "image_voxels", "image_cache"],
    }
    payload["feature_digest"] = sha256_mapping(payload)
    payload["contract_sha256"] = sha256_mapping(payload)
    return payload


def _assert_public_payload(payload: Mapping[str, Any]) -> None:
    def walk(value: Any) -> list[str]:
        problems: list[str] = []
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).casefold() in _PRIVATE_KEYS:
                    problems.append(str(key))
                problems.extend(walk(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                problems.extend(walk(item))
        elif isinstance(value, str) and re.search(r"(?i)(?:[A-Z]:[\\/]|A4[ _]Learn|raw_path|source_subject_id|\bpt_[a-z0-9_-]+|\bB\d{5,}\b|\bLRN[_-]?\d{3,}\b)", value):
            problems.append("private_value")
        return problems
    if walk(payload):
        raise SourceRegionalFeatureError("PRIVATE_FIELD_IN_PUBLIC_CONTRACT")


def write_public_feature_contract(payload: Mapping[str, Any], output_root: str | Path, branch: str) -> tuple[Path, Path]:
    """Publish report then JSON gate atomically; both artifacts are path-free."""

    _assert_public_payload(payload)
    root = Path(output_root)
    md = root / f"{branch}_feature_contract.md"
    js = root / f"{branch}_feature_contract.json"
    report = "\n".join((f"# {branch} source-regional feature contract", "", f"Status: `{payload.get('status', 'BLOCKED')}`", "", "This public contract intentionally contains no subject-level matrix, raw identifier, raw path, image cache, or prediction.", ""))
    atomic_write_bytes(md, report.encode("utf-8"))
    atomic_write_json(js, dict(payload))
    return md, js
