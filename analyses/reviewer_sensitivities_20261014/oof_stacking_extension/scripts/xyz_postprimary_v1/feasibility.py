"""Exact-scan and common-landmark feasibility, not permission to reuse old OOF."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.xyz_fusion_v1.audit import RAW, _table
from scripts.xyz_fusion_v1.intake import sustained_endpoint


def exact_fbp_link(row, membership, mapping, lineage, source_scans=None):
    selected = mapping.loc[mapping.sample_hash.eq(membership.image_key)]
    if len(selected) != 1:
        raise ValueError("AMBIGUOUS_OR_MISSING_EXACT_IMAGE_MAPPING")
    scan = selected.iloc[0]
    if scan.patient_hash != row.patient_hash or str(scan.tracer).upper() != "FBP":
        raise ValueError("PET_IDENTITY_OR_TRACER_MISMATCH")
    if float(scan.visit_code) != float(lineage["fbp_visit"]):
        raise ValueError("PET_IMAGE_IS_NOT_THE_SELECTED_BASELINE_VISIT")
    day = float(scan.pet_day)
    reconciled = False
    if source_scans is not None:
        # Historical manifest dates can be stale. Never subtract an inferred
        # global offset: resolve the exact subject + visit + selected scan row.
        source = source_scans.loc[source_scans.patient_hash.eq(row.patient_hash)
            & pd.to_numeric(source_scans.VISCODE, errors="coerce").eq(float(scan.visit_code))
            & pd.to_numeric(source_scans.scan_number, errors="coerce").eq(float(lineage["fbp_scan_number"]))]
        dates = pd.to_numeric(source.scan_date_DAYS_T0, errors="coerce").dropna().unique()
        if len(dates) != 1: raise ValueError("EXACT_SOURCE_SCAN_DATE_AMBIGUOUS")
        from src.utils.pet_path_resolver import parse_structured_pet_path
        identity = parse_structured_pet_path(str(scan.raw_path_private), "fbp")
        if (identity is None or identity.bid != str(scan.bid).upper()
                or identity.substudy != str(scan.substudy).upper()
                or float(identity.visit_code) != float(scan.visit_code)):
            raise ValueError("RAW_FILENAME_IDENTITY_NOT_CONCORDANT_WITH_EXACT_SOURCE_ROW")
        reconciled = not np.isfinite(day) or abs(day-float(dates[0])) > 1e-6
        day = float(dates[0])
    if not np.isfinite(day) or abs(day - row.fbp_day) > 1e-6:
        raise ValueError("PET_IMAGE_EXACT_DATE_NOT_PROVEN")
    if day > row.landmark_days_t0 or row.landmark_days_t0 - day > 183:
        raise ValueError("PET_IMAGE_LANDMARK_VIOLATION")
    scan = scan.copy()
    scan["verified_day_t0"] = day
    scan["legacy_date_reconciled"] = reconciled
    return scan


def audit_pet_linkage(ctx):
    root = ctx["root"]
    membership = pd.read_parquet(root / "results/method_y_revival_v3_image_v2/private/amyloid_image_membership.parquet")
    membership = membership.set_index("patient_hash", drop=False, verify_integrity=True)
    mapping = pd.read_csv(root / "data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv")
    source = _table(RAW / "External Data/imaging_SUVR_amyloid.csv")
    source = source.loc[source.ligand.str.casefold().eq("florbetapir") & source.scan_analyzed.str.casefold().eq("yes")]
    source_groups = {key: table for key, table in source.groupby("patient_hash")}
    intake = ctx["store"].job("intake")
    lineage = {x["patient_hash"]: x for x in intake["lineage"]}
    rows = []; reasons = Counter(); handles = {}
    try:
        for row in ctx["frame"].itertuples():
            if row.patient_hash not in membership.index:
                reasons["NO_IMAGE_MEMBERSHIP"] += 1; continue
            m = membership.loc[row.patient_hash]
            try:
                if row.patient_hash not in source_groups: raise ValueError("EXACT_FBP_SOURCE_ROWS_MISSING")
                scan = exact_fbp_link(row, m, mapping, lineage[row.patient_hash], source_groups[row.patient_hash])
                path = str(m.image_path_private)
                if path not in handles:
                    handles[path] = h5py.File(path, "r")
                group = handles[path][str(m.h5_group)]
                if (str(group.attrs.get("source_sha256")) != str(m.source_image_sha256)
                        or str(group.attrs.get("tracer")).upper() != "FBP"
                        or tuple(group["image"].shape) != (1, 128, 128, 128)):
                    raise ValueError("PET_CACHE_PARENT_OR_GEOMETRY_MISMATCH")
                rows.append({"patient_hash": row.patient_hash, "image_key": m.image_key,
                             "pet_day": float(scan.verified_day_t0), "legacy_pet_day": float(scan.pet_day),
                             "legacy_date_reconciled": bool(scan.legacy_date_reconciled), "event": int(row.event),
                             "time_days": float(row.time_days), "manufacturer": str(m.manufacturer)})
            except (ValueError, KeyError) as exc:
                reasons[str(exc)] += 1
    finally:
        for handle in handles.values(): handle.close()
    keys = {r["patient_hash"] for r in rows}
    folds = [{"fold": i, "n": sum(k in keys for k in split["test"]),
              "events": sum(r["event"] for r in rows if r["patient_hash"] in set(split["test"]))}
             for i, split in enumerate(ctx["splits"])]
    return {"status": "FEASIBILITY_ONLY_NOT_TRAINING_READY", "exact_scan_date_subjects": len(rows),
            "events": sum(r["event"] for r in rows), "outer_fold_support": folds,
            "legacy_manifest_date_disagreements": sum(r["legacy_date_reconciled"] for r in rows),
            "date_basis": "EXACT_SUBJECT_VISIT_AND_SELECTED_SCAN_SOURCE_TABLE_NOT_OFFSET_CORRECTION",
            "exclusions": dict(reasons), "training_started": False,
            "remaining": ["FRESH_PET_EXPERT_NESTED_CROSSFIT", "JOINT_INNER_ANCESTRY_PROOF",
                          "PATTERN_CACHE_VALIDATION_AND_QUANTITATIVE_CLAIM_SEPARATION"],
            "historical_image_models_reusable": False, "private_rows": rows}


def audit_tau_landmark(ctx):
    tau = _table(RAW / "External Data/imaging_SUVR_tau.csv")
    tau = tau.loc[tau.ligand.str.casefold().eq("flortaucipir") & tau.scan_analyzed.str.casefold().eq("yes")].copy()
    tau["day"] = pd.to_numeric(tau.scan_date_DAYS_T0, errors="coerce")
    tau = tau.dropna(subset=["day"]).drop_duplicates(["patient_hash", "day", "VISCODE", "scan_number"])
    groups = {k: t for k, t in tau.groupby("patient_hash")}
    cdr = _table(RAW / "Raw Data/cdr.csv")
    cdr["day"] = pd.to_numeric(cdr.CDADTC_DAYS_T0, errors="coerce")
    cdr["cdr"] = pd.to_numeric(cdr.CDGLOBAL, errors="coerce")
    visits = {k: t.dropna(subset=["day", "cdr"]) for k, t in cdr.groupby("patient_hash")}
    rows = []; reasons = Counter()
    for row in ctx["frame"].itertuples():
        try:
            if row.patient_hash not in groups: raise ValueError("NO_ANALYZED_DATED_FTP")
            candidates = groups[row.patient_hash].copy()
            # Choose scan without consulting future outcomes or regional intensities.
            candidates = candidates.loc[(candidates.day - row.mri_day).abs().le(183)
                                        & (candidates.day - row.fbp_day).abs().le(183)].sort_values("day")
            if candidates.empty: raise ValueError("FTP_OUTSIDE_BASELINE_WINDOW")
            chosen = candidates.loc[candidates.day.eq(candidates.iloc[0].day)]
            if len(chosen) != 1: raise ValueError("AMBIGUOUS_FTP_SCAN")
            ftp_day = float(chosen.iloc[0].day)
            anchor = max(ftp_day, row.landmark_days_t0)
            cs = visits.get(row.patient_hash, pd.DataFrame(columns=["day", "cdr"]))
            near = cs.assign(gap=(cs.day-anchor).abs()).sort_values(["gap", "day"])
            if near.empty or near.iloc[0].gap > 183 or near.iloc[0].cdr != 0:
                raise ValueError("NO_VALID_CDR_ZERO_AT_PAIRED_LANDMARK")
            origin = max(anchor, float(near.iloc[0].day))
            if origin - min(ftp_day, row.mri_day, row.fbp_day, row.cognition_day) > 183:
                raise ValueError("PAIRED_PREDICTOR_TOO_OLD")
            event, time = sustained_endpoint(list(zip(cs.day, cs.cdr)), origin, 1640.)
            rows.append({"patient_hash": row.patient_hash, "origin": origin, "ftp_day": ftp_day,
                         "event": event, "time_days": time, "landmark_changed": origin != row.landmark_days_t0})
        except ValueError as exc: reasons[str(exc)] += 1
    return {"status": "FEASIBILITY_ONLY_REGION_MAPPING_AND_NEW_CROSSFIT_REQUIRED",
            "subjects": len(rows), "events": sum(r["event"] for r in rows),
            "changed_landmarks": sum(r["landmark_changed"] for r in rows), "exclusions": dict(reasons),
            "power_simulation_completed": False, "training_started": False,
            "legacy_tau_beyond_amyloid_negative_result_preserved": True, "private_rows": rows}
