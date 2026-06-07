from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

from src.data.a4_schema import PET_DAY_COLUMN, T1_NAME_PATTERN, resolve_data_root, resolve_required_file
from src.utils.pet_manifest import PetCandidate, canonical_tracer, stable_file_hash

TRACER_FOLDER = {
    "fbp": "FBP",
    "ftp": "FTP",
    "fbb": "FBB",
    "pib": "PIB",
    "fmm": "FMM",
}

LIGAND_TO_TRACER = {
    "florbetapir": "fbp",
    "florbetaben": "fbb",
    "pib": "pib",
    "pittsburgh compound b": "pib",
    "flutemetamol": "fmm",
    "flortaucipir": "ftp",
}

STRUCTURED_PET_NAME = re.compile(
    r"^(?P<substudy>A4|LEARN|SF)_PET_(?P<tracer_tag>[A-Z]+)_(?P<bid>B\d+)_(?P<visit>\d{3})\.nii(\.gz)?$",
    flags=re.IGNORECASE,
)

T1_NAME = re.compile(T1_NAME_PATTERN, flags=re.IGNORECASE)


def _float_or_none(value: str | None) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NULL"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_pet_va_index(data_root: str | Path) -> dict[tuple[str, str, str], dict[str, str]]:
    root = resolve_data_root(str(data_root))
    path = resolve_required_file(root, "ab_label")
    index: dict[tuple[str, str, str], dict[str, str]] = {}
    if not path.exists():
        return index
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            substudy = (row.get("SUBSTUDY") or "").upper()
            bid = (row.get("BID") or "").upper()
            visit = (row.get("VISCODE") or "").strip()
            if substudy and bid and visit:
                index[(substudy, bid, visit)] = row
    return index


def parse_structured_pet_path(path: str | Path, tracer: str) -> Optional[PetCandidate]:
    name = Path(path).name
    match = STRUCTURED_PET_NAME.match(name)
    if not match:
        return None
    tracer_tag = match.group("tracer_tag").upper()
    expected_folder = TRACER_FOLDER.get(canonical_tracer(tracer), canonical_tracer(tracer)).upper()
    if tracer_tag != expected_folder and canonical_tracer(tracer_tag.lower()) != canonical_tracer(tracer):
        return None
    bid = match.group("bid").upper()
    if not bid.startswith("B"):
        bid = "B" + bid
    return PetCandidate(
        raw_path=str(path),
        tracer=canonical_tracer(tracer),
        substudy=match.group("substudy").upper(),
        bid=bid,
        visit_code=match.group("visit"),
        source="project_inventory",
        file_hash=stable_file_hash(path),
    )


def enrich_candidate_from_va(candidate: PetCandidate, va_row: dict[str, str]) -> PetCandidate:
    ligand = (va_row.get("ligand") or "").lower()
    tracer = canonical_tracer(LIGAND_TO_TRACER.get(ligand, candidate.tracer))
    pet_day = _float_or_none(va_row.get(PET_DAY_COLUMN))
    return PetCandidate(
        raw_path=candidate.raw_path,
        tracer=tracer,
        substudy=candidate.substudy,
        bid=candidate.bid,
        visit_code=candidate.visit_code or (va_row.get("VISCODE") or ""),
        pet_day=pet_day,
        scanner=candidate.scanner,
        frame=candidate.frame,
        source=candidate.source,
        file_hash=candidate.file_hash,
        patient_hash=candidate.patient_hash,
        sample_hash=candidate.sample_hash,
        t1_path=candidate.t1_path,
        abs_pet_t1_gap_days=candidate.abs_pet_t1_gap_days,
    )


def discover_pet_from_project_inventory(data_root: str | Path, tracer: str) -> list[PetCandidate]:
    root = resolve_data_root(str(data_root))
    tracer = canonical_tracer(tracer)
    folder = TRACER_FOLDER.get(tracer, tracer.upper())
    image_dir = root / folder / "ImageData"
    if not image_dir.is_dir():
        return []

    va_index = load_pet_va_index(root)
    candidates: list[PetCandidate] = []
    for path in sorted(image_dir.glob("*.nii*")):
        if not path.is_file():
            continue
        candidate = parse_structured_pet_path(path, tracer=tracer)
        if candidate is None:
            continue
        va_row = va_index.get((candidate.substudy, candidate.bid, candidate.visit_code))
        if va_row:
            candidate = enrich_candidate_from_va(candidate, va_row)
        candidates.append(candidate)
    return candidates


def load_native_t1_index(data_root: str | Path) -> dict[tuple[str, str, str], str]:
    root = resolve_data_root(str(data_root))
    t1_dir = root / "T1"
    index: dict[tuple[str, str, str], str] = {}
    if not t1_dir.is_dir():
        return index
    for path in sorted(t1_dir.glob("*.nii*")):
        if not path.is_file():
            continue
        match = T1_NAME.match(path.name)
        if not match:
            continue
        bid = match.group("bid").upper()
        if not bid.startswith("B"):
            bid = "B" + bid
        key = (match.group("substudy").upper(), bid, match.group("scan"))
        index[key] = str(path)
    return index


def resolve_native_t1_path(
    data_root: str | Path,
    substudy: str,
    bid: str,
    visit_code: str,
    index: dict[tuple[str, str, str], str] | None = None,
) -> str:
    index = index or load_native_t1_index(data_root)
    bid = bid.upper()
    if not bid.startswith("B"):
        bid = "B" + bid
    substudy = substudy.upper()
    visit = (visit_code or "").strip()
    if visit:
        exact = index.get((substudy, bid, visit))
        if exact:
            return exact
    matches = [path for (s, b, _visit), path in index.items() if s == substudy and b == bid]
    return sorted(matches)[0] if matches else ""


def attach_native_t1_paths_to_candidates(
    candidates: list[PetCandidate],
    data_root: str | Path,
) -> list[PetCandidate]:
    index = load_native_t1_index(data_root)
    out: list[PetCandidate] = []
    for candidate in candidates:
        native_t1 = resolve_native_t1_path(
            data_root,
            candidate.substudy,
            candidate.bid,
            candidate.visit_code,
            index=index,
        )
        if native_t1:
            candidate = PetCandidate(
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
                patient_hash=candidate.patient_hash,
                sample_hash=candidate.sample_hash,
                t1_path=native_t1,
                abs_pet_t1_gap_days=candidate.abs_pet_t1_gap_days,
            )
        out.append(candidate)
    return out
