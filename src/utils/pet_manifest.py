from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

from .pet_spatial_qc import scrub_sensitive_text, sha256_file

IMAGE_SUFFIXES = (".nii", ".nii.gz")

TRACER_ALIASES = {
    "fbp": {"fbp", "florbetapir", "av45", "amyloid"},
    "fbb": {"fbb", "florbetaben"},
    "pib": {"pib"},
    "fmm": {"fmm", "flutemetamol"},
    "ftp": {"ftp", "flortaucipir", "av1451", "tau"},
}


@dataclass(frozen=True)
class PetCandidate:
    raw_path: str
    tracer: str
    substudy: str
    bid: str
    visit_code: str = ""
    pet_day: Optional[float] = None
    scanner: str = ""
    frame: str = ""
    source: str = "unknown"
    file_hash: str = ""
    patient_hash: str = ""
    sample_hash: str = ""
    t1_path: str = ""
    abs_pet_t1_gap_days: Optional[float] = None

    @property
    def redacted_path(self) -> str:
        return scrub_sensitive_text(self.raw_path)

    @property
    def patient_key(self) -> str:
        if self.patient_hash:
            return self.patient_hash
        return f"{self.substudy.upper()}|{self.bid.upper()}"


@dataclass(frozen=True)
class CatalogRow:
    sample_hash: str
    patient_hash: str
    substudy: str
    ab_label: str
    abs_pet_t1_gap_days: Optional[float] = None
    visit_code: str = ""
    t1_path: str = ""
    scanner: str = ""
    tracer: str = ""

    @property
    def is_binary_label(self) -> bool:
        return str(self.ab_label).lower() in {"0", "1", "ab_pos", "ab_neg", "positive", "negative", "pos", "neg"}


@dataclass
class WorkItem:
    candidate: PetCandidate
    catalog: CatalogRow
    tier: str
    selected_reason: str = "selected"

    def export_safe(self) -> dict:
        return {
            "sample_hash": self.catalog.sample_hash,
            "patient_hash": self.catalog.patient_hash,
            "tracer": self.candidate.tracer,
            "substudy": self.catalog.substudy,
            "tier": self.tier,
            "selected_reason": self.selected_reason,
            "file_hash": self.candidate.file_hash,
            "pet_t1_gap_days": self.catalog.abs_pet_t1_gap_days,
        }


def canonical_tracer(name: str) -> str:
    s = (name or "").lower()
    for key, aliases in TRACER_ALIASES.items():
        if s == key or s in aliases or any(a in s for a in aliases):
            return key
    return s


def stable_file_hash(path: str | Path) -> str:
    p = Path(path)
    try:
        return sha256_file(p)[:16]
    except Exception:
        return hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:16]


def parse_pet_filename(path: str | Path, tracer: str) -> Optional[PetCandidate]:
    p = Path(path)
    name = p.name
    patterns = [
        r"(?P<substudy>A4|LEARN|SF).*?(?P<bid>B\d{4,})",
        r"(?P<bid>B\d{4,}).*?(?P<substudy>A4|LEARN|SF)",
        r"(?P<substudy>A4|LEARN|SF)[_-](?P<bid>\d{4,})",
        r"(?P<bid>B\d{4,})",
    ]
    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            substudy = m.groupdict().get("substudy", "").upper() or "UNKNOWN"
            bid = m.group("bid").upper()
            if not bid.startswith("B"):
                bid = "B" + bid
            return PetCandidate(
                raw_path=str(p), tracer=canonical_tracer(tracer), substudy=substudy,
                bid=bid, source="filename_regex_fallback", file_hash=stable_file_hash(p)
            )
    return None


def discover_pet_files(data_root: str | Path, tracer: str) -> list[PetCandidate]:
    root = Path(data_root)
    candidates: list[PetCandidate] = []
    tracer = canonical_tracer(tracer)
    aliases = TRACER_ALIASES.get(tracer, {tracer})
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        lower_path = fp.as_posix().lower()
        if not (lower_path.endswith(".nii") or lower_path.endswith(".nii.gz")):
            continue
        if not any(a in lower_path for a in aliases):
            continue
        c = parse_pet_filename(fp, tracer=tracer)
        if c is not None:
            candidates.append(c)
    return candidates


def _float_or_none(x):
    try:
        return None if x in (None, "") else float(x)
    except Exception:
        return None


def read_private_manifest(path: str | Path, tracer: str | None = None) -> list[PetCandidate]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[PetCandidate] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        lower_map = {name.lower(): name for name in reader.fieldnames or []}
        def get(row, key, default=""):
            return row.get(lower_map.get(key.lower(), key), default)
        for row in reader:
            t = canonical_tracer(get(row, "tracer", tracer or ""))
            if tracer and t and t != canonical_tracer(tracer):
                continue
            raw_path = get(row, "raw_path_private") or get(row, "raw_path") or get(row, "path")
            if not raw_path:
                continue
            out.append(PetCandidate(
                raw_path=raw_path,
                tracer=t or canonical_tracer(tracer or ""),
                substudy=get(row, "substudy").upper(),
                bid=get(row, "bid").upper(),
                visit_code=get(row, "visit_code") or get(row, "viscode"),
                pet_day=_float_or_none(get(row, "pet_day", "")),
                scanner=get(row, "scanner"),
                frame=get(row, "frame"),
                source=get(row, "source", "private_manifest"),
                file_hash=get(row, "pet_file_hash") or stable_file_hash(raw_path),
                patient_hash=get(row, "patient_hash"),
                sample_hash=get(row, "sample_hash"),
                t1_path=get(row, "t1_path") or get(row, "mri_path"),
                abs_pet_t1_gap_days=_float_or_none(get(row, "pet_t1_gap_days") or get(row, "abs_pet_t1_gap_days")),
            ))
    return out


def write_private_manifest(path: str | Path, candidates: Sequence[PetCandidate]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pet_file_hash", "raw_path_private", "substudy", "bid", "visit_code", "pet_day", "pet_t1_gap_days",
        "tracer", "frame", "scanner", "source", "patient_hash", "sample_hash", "t1_path"
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in candidates:
            w.writerow({
                "pet_file_hash": c.file_hash or stable_file_hash(c.raw_path),
                "raw_path_private": c.raw_path,
                "substudy": c.substudy,
                "bid": c.bid,
                "visit_code": c.visit_code,
                "pet_day": "" if c.pet_day is None else c.pet_day,
                "pet_t1_gap_days": "" if c.abs_pet_t1_gap_days is None else c.abs_pet_t1_gap_days,
                "tracer": c.tracer,
                "frame": c.frame,
                "scanner": c.scanner,
                "source": c.source,
                "patient_hash": c.patient_hash,
                "sample_hash": c.sample_hash,
                "t1_path": c.t1_path,
            })


def enrich_candidate(candidate: PetCandidate) -> PetCandidate:
    from .privacy import stable_hash

    if not candidate.substudy or not candidate.bid:
        return candidate
    patient_hash = candidate.patient_hash or stable_hash(candidate.substudy + "|" + candidate.bid, prefix="pt")
    sample_hash = candidate.sample_hash or stable_hash(
        candidate.substudy + "|" + candidate.bid + "|" + (candidate.visit_code or ""),
        prefix="s",
    )
    return PetCandidate(
        raw_path=candidate.raw_path,
        tracer=candidate.tracer,
        substudy=candidate.substudy,
        bid=candidate.bid,
        visit_code=candidate.visit_code,
        pet_day=candidate.pet_day,
        scanner=candidate.scanner,
        frame=candidate.frame,
        source=candidate.source,
        file_hash=candidate.file_hash,
        patient_hash=patient_hash,
        sample_hash=sample_hash,
        t1_path=candidate.t1_path,
        abs_pet_t1_gap_days=candidate.abs_pet_t1_gap_days,
    )


def enrich_candidates(candidates: Sequence[PetCandidate]) -> list[PetCandidate]:
    return [enrich_candidate(c) for c in candidates]


def attach_t1_paths(catalog: Sequence[CatalogRow], preprocessed_dir: str | Path) -> list[CatalogRow]:
    root = Path(preprocessed_dir)
    out: list[CatalogRow] = []
    for row in catalog:
        t1_path = row.t1_path
        if not t1_path:
            candidate = root / f"{row.sample_hash}.nii.gz"
            if candidate.exists():
                t1_path = str(candidate)
        out.append(replace(row, t1_path=t1_path or row.t1_path))
    return out


def read_catalog(path: str | Path) -> list[CatalogRow]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[CatalogRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        lower_map = {name.lower(): name for name in reader.fieldnames or []}
        def get(row, key, default=""):
            return row.get(lower_map.get(key.lower(), key), default)
        for row in reader:
            gap = get(row, "abs_pet_t1_gap_days") or get(row, "pet_t1_gap_days") or get(row, "abs_ab_pet_t1_gap_days")
            rows.append(CatalogRow(
                sample_hash=get(row, "sample_hash") or get(row, "sample_id_hash"),
                patient_hash=get(row, "patient_hash") or get(row, "patient_id_hash"),
                substudy=get(row, "substudy").upper(),
                ab_label=get(row, "ab_label") or get(row, "label"),
                abs_pet_t1_gap_days=_float_or_none(gap),
                visit_code=get(row, "visit_code") or get(row, "viscode"),
                t1_path=get(row, "t1_path") or get(row, "mri_path") or get(row, "t1_nifti_path"),
                scanner=get(row, "scanner") or get(row, "site") or get(row, "scanner_id"),
                tracer=canonical_tracer(get(row, "tracer") or get(row, "pet_tracer")),
            ))
    return rows


def read_catalog_with_t1_paths(path: str | Path, preprocessed_dir: str | Path = "data/preprocessed_mri_96") -> list[CatalogRow]:
    return attach_t1_paths(read_catalog(path), preprocessed_dir)


def tier_for_gap(gap: Optional[float], policy: dict) -> str:
    if gap is None:
        return "missing"
    tiers = policy.get("temporal_pairing", {}).get("eligibility_tiers", {}) or policy.get("temporal_alignment", {}).get("eligibility_tiers", {})
    ordered = ["gold_standard_30", "primary_strict", "secondary_window", "relaxed_exploratory"]
    for name in ordered:
        max_gap = tiers.get(name, {}).get("max_abs_gap_days")
        if max_gap is not None and float(gap) <= float(max_gap):
            return name
    return "excluded_from_training"


def _catalog_match_key(row: CatalogRow) -> str:
    return row.patient_hash or f"{row.substudy}|{row.sample_hash}"


def link_candidates_to_catalog(
    catalog: Sequence[CatalogRow],
    candidates: Sequence[PetCandidate],
    policy: dict,
    requested_tier: str = "primary_strict",
    one_per_patient: bool = True,
) -> tuple[list[WorkItem], list[dict], dict]:
    excluded = []
    by_patient_hash = defaultdict(list)
    by_sample_hash = defaultdict(list)
    by_raw_key = defaultdict(list)
    for c in candidates:
        if c.patient_hash:
            by_patient_hash[c.patient_hash].append(c)
        if c.sample_hash:
            by_sample_hash[c.sample_hash].append(c)
        if c.substudy and c.bid:
            by_raw_key[f"{c.substudy.upper()}|{c.bid.upper()}"].append(c)
    exclude_substudies = set(policy.get("sample_selection", {}).get("exclude_substudies_from_mainline", []))
    work: list[WorkItem] = []
    for r in catalog:
        reason = None
        if r.substudy in exclude_substudies:
            reason = "SUBSTUDY_EXCLUDED"
        elif not r.is_binary_label:
            reason = "MISSING_LABEL"
        tier = tier_for_gap(r.abs_pet_t1_gap_days, policy)
        if reason is None and requested_tier and requested_tier != "all" and tier != requested_tier:
            reason = "GAP_NOT_IN_REQUESTED_TIER"
        if reason:
            excluded.append({"sample_hash": r.sample_hash, "patient_hash": r.patient_hash, "reason": reason, "tier": tier, "gap": r.abs_pet_t1_gap_days})
            continue
        cand = []
        if r.patient_hash in by_patient_hash:
            cand.extend(by_patient_hash[r.patient_hash])
        if r.sample_hash in by_sample_hash:
            cand.extend(by_sample_hash[r.sample_hash])
        if not cand:
            cand = [
                c for c in candidates
                if c.substudy == r.substudy
                and (c.patient_hash == r.patient_hash or (not c.patient_hash and not r.patient_hash))
            ]
        if not cand:
            cand = [
                c for c in candidates
                if c.substudy == r.substudy and not c.patient_hash
            ]
        if not cand:
            excluded.append({"sample_hash": r.sample_hash, "patient_hash": r.patient_hash, "reason": "NO_PET_CANDIDATE_MATCH", "tier": tier, "gap": r.abs_pet_t1_gap_days})
            continue
        cand = [c for c in cand if not r.tracer or c.tracer == r.tracer or c.tracer in {"fbp", "fbb", "pib", "fmm", "ftp"}]
        if not cand:
            excluded.append({"sample_hash": r.sample_hash, "patient_hash": r.patient_hash, "reason": "TRACER_UNSUPPORTED", "tier": tier, "gap": r.abs_pet_t1_gap_days})
            continue
        best = sorted(cand, key=lambda c: (abs(r.abs_pet_t1_gap_days if r.abs_pet_t1_gap_days is not None else c.abs_pet_t1_gap_days or 1e9), c.file_hash))[0]
        if not (r.t1_path or best.t1_path):
            excluded.append({"sample_hash": r.sample_hash, "patient_hash": r.patient_hash, "reason": "MISSING_T1", "tier": tier, "gap": r.abs_pet_t1_gap_days})
            continue
        work.append(WorkItem(candidate=best, catalog=r, tier=tier))
    duplicate_exclusions = []
    if one_per_patient:
        grouped = defaultdict(list)
        for wi in work:
            grouped[wi.catalog.patient_hash].append(wi)
        deduped: list[WorkItem] = []
        for ph, items in grouped.items():
            best = sorted(items, key=lambda wi: (abs(wi.catalog.abs_pet_t1_gap_days or 1e9), wi.candidate.file_hash))[0]
            deduped.append(best)
            for wi in items:
                if wi is not best:
                    duplicate_exclusions.append({"sample_hash": wi.catalog.sample_hash, "patient_hash": ph, "reason": "DUPLICATE_PATIENT_NON_SELECTED", "tier": wi.tier, "gap": wi.catalog.abs_pet_t1_gap_days})
        work = deduped
        excluded.extend(duplicate_exclusions)
    tiers = Counter(wi.tier for wi in work)
    labels = Counter(str(wi.catalog.ab_label) for wi in work)
    summary = {
        "catalog_rows": len(catalog), "candidate_rows": len(candidates), "linked_work_items": len(work),
        "excluded_rows": len(excluded), "one_pet_per_patient": bool(one_per_patient), "requested_tier": requested_tier,
        "linked_tiers": dict(tiers), "linked_labels": dict(labels),
        "note": "Training-ready linking requires private manifest or project resolver; filename fallback is dry-run only.",
    }
    return work, excluded, summary


def write_excluded_manifest(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_hash", "patient_hash", "reason", "tier", "gap"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
