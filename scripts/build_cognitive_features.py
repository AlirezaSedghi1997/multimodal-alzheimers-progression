from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import AB_LABEL_COLUMN, PET_DAY_COLUMN, REQUIRED_RELATIVE_FILES, resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


CORE_COGNITIVE_FEATURES = ["pacc", "fctotal96", "ldeltotal", "digittotal", "mmscore"]
CLINICAL_SEVERITY_FEATURES = [
    "cdglobal",
    "cdrsb",
    "cdr_memory",
    "cdr_orient",
    "cdr_judge",
    "cdr_commun",
    "cdr_home",
    "cdr_care",
]
COMPLAINT_FEATURES = ["cfipttotal", "cfspttotal"]
COGSTATE_FEATURES = ["c3comp", "learnwmcomp", "oclonbcomp", "psychattncomp", "attncomp", "c3abrcomp"]
MOOD_ANXIETY_FEATURES = ["gdtotal", "staitotal", "cadcncrn", "iescore"]
COGNITIVE_FEATURES = [
    *CORE_COGNITIVE_FEATURES,
    *CLINICAL_SEVERITY_FEATURES,
    *COMPLAINT_FEATURES,
    *COGSTATE_FEATURES,
    *MOOD_ANXIETY_FEATURES,
]
FORBIDDEN_INPUT_TERMS = ("pet", "tau", "suvr", "amyloid", "centiloid", "scan_date", "raw_path")
SOURCE_FILES = {
    "pacc": Path("Clinical") / "Derived Data" / "PACC.csv",
    "cdr": Path("Clinical") / "Raw Data" / "cdr.csv",
    "cfi": Path("Clinical") / "Raw Data" / "cfi.csv",
    "cfisp": Path("Clinical") / "Raw Data" / "cfisp.csv",
    "cogstate": Path("Clinical") / "External Data" / "cogstate_battery.csv",
    "psychwell": Path("Clinical") / "Raw Data" / "psychwell.csv",
    "concerns": Path("Clinical") / "Raw Data" / "concerns.csv",
    "ies": Path("Clinical") / "Raw Data" / "ies.csv",
}


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "NULL", "."}:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    if math.isnan(value_float):
        return None
    return value_float


def norm_visit(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text.upper()


def norm_substudy(value: str | None) -> str:
    return (value or "").strip().upper()


def subject_key(row: dict[str, str]) -> tuple[str, str]:
    return norm_substudy(row.get("SUBSTUDY")), row.get("BID", "").strip()


def sample_hash_for_pet(row: dict[str, str]) -> str:
    substudy, bid = subject_key(row)
    return stable_hash(substudy + "|" + bid + "|" + str(row.get("VISCODE", "")), prefix="s")


def build_catalog_link(catalog_rows: list[dict[str, str]], pet_rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    wanted = {row["sample_hash"] for row in catalog_rows}
    catalog_by_sample = {row["sample_hash"]: row for row in catalog_rows}
    out: dict[str, dict[str, object]] = {}
    for pet in pet_rows:
        sample_hash = sample_hash_for_pet(pet)
        if sample_hash not in wanted:
            continue
        catalog_row = catalog_by_sample[sample_hash]
        pet_day = to_float(catalog_row.get("pet_day_t0"))
        if pet_day is None:
            pet_day = to_float(pet.get(PET_DAY_COLUMN))
        t1_day = to_float(catalog_row.get("t1_day_t0"))
        target_day = pet_day if pet_day is not None else t1_day
        substudy, bid = subject_key(pet)
        out[sample_hash] = {
            "substudy": substudy,
            "bid": bid,
            "visit": norm_visit(pet.get("VISCODE")),
            "target_day": target_day,
        }
    return out


def index_by_subject(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        substudy, bid = subject_key(row)
        if substudy and bid:
            out[(substudy, bid)].append(row)
    return out


def select_row(
    rows: list[dict[str, str]],
    *,
    target_day: float | None,
    visit: str | None,
    day_columns: tuple[str, ...] = (),
) -> tuple[dict[str, str] | None, str]:
    if target_day is not None and day_columns:
        candidates = []
        for row in rows:
            day = next((to_float(row.get(col)) for col in day_columns if to_float(row.get(col)) is not None), None)
            if day is None or day > target_day:
                continue
            candidates.append((row, day))
        if candidates:
            row, _ = min(candidates, key=lambda item: (target_day - float(item[1]), -float(item[1])))
            return row, "nearest_nonfuture"
    if visit is not None:
        for row in rows:
            row_visit = norm_visit(row.get("VISCODE") or row.get("VISITCD") or row.get("VISIT"))
            if row_visit == visit:
                return row, "exact_visit"
    return None, "missing"


def assign_feature(features: dict[str, object], name: str, value: str | None) -> None:
    parsed = to_float(value)
    features[name] = "" if parsed is None else parsed


def add_pacc(features: dict[str, object], row: dict[str, str] | None) -> None:
    mapping = {
        "pacc": "PACC",
        "fctotal96": "FCTOTAL96",
        "ldeltotal": "LDELTOTAL",
        "digittotal": "DIGITTOTAL",
        "mmscore": "MMSCORE",
    }
    for out_name, src_name in mapping.items():
        assign_feature(features, out_name, row.get(src_name) if row else None)


def add_cdr(features: dict[str, object], row: dict[str, str] | None) -> None:
    mapping = {
        "cdglobal": "CDGLOBAL",
        "cdrsb": "CDRSB",
        "cdr_memory": "MEMORY",
        "cdr_orient": "ORIENT",
        "cdr_judge": "JUDGE",
        "cdr_commun": "COMMUN",
        "cdr_home": "HOME",
        "cdr_care": "CARE",
    }
    for out_name, src_name in mapping.items():
        assign_feature(features, out_name, row.get(src_name) if row else None)


def add_complaints(features: dict[str, object], cfi: dict[str, str] | None, cfisp: dict[str, str] | None) -> None:
    assign_feature(features, "cfipttotal", cfi.get("CFIPTTOTAL") if cfi else None)
    assign_feature(features, "cfspttotal", cfisp.get("CFSPTTOTAL") if cfisp else None)


def add_cogstate(features: dict[str, object], row: dict[str, str] | None) -> None:
    mapping = {
        "c3comp": "C3Comp",
        "learnwmcomp": "LearnWMComp",
        "oclonbcomp": "OCLONBComp",
        "psychattncomp": "PsychAttnComp",
        "attncomp": "AttnComp",
        "c3abrcomp": "C3AbrComp",
    }
    for out_name, src_name in mapping.items():
        assign_feature(features, out_name, row.get(src_name) if row else None)


def add_mood_anxiety(
    features: dict[str, object],
    psychwell: dict[str, str] | None,
    concerns: dict[str, str] | None,
    ies: dict[str, str] | None,
) -> None:
    assign_feature(features, "gdtotal", psychwell.get("GDTOTAL") if psychwell else None)
    assign_feature(features, "staitotal", psychwell.get("STAITOTAL") if psychwell else None)
    assign_feature(features, "cadcncrn", concerns.get("CADCNCRN") if concerns else None)
    assign_feature(features, "iescore", ies.get("IESCORE") if ies else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out", default="data/processed/cognitive_features.csv")
    ap.add_argument("--summary", default="results/tables/cognitive_feature_summary.json")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    catalog_rows = read_csv_dicts(Path(args.catalog))
    pet_rows = read_csv_dicts(root / REQUIRED_RELATIVE_FILES["ab_label"])
    links = build_catalog_link(catalog_rows, pet_rows)
    sources = {name: index_by_subject(read_csv_dicts(root / rel)) for name, rel in SOURCE_FILES.items()}

    rows: list[dict[str, object]] = []
    alignment_counts: Counter[str] = Counter()
    for catalog_row in catalog_rows:
        sample_hash = catalog_row["sample_hash"]
        link = links.get(sample_hash)
        base: dict[str, object] = {"sample_hash": sample_hash}
        for feature in COGNITIVE_FEATURES:
            base[feature] = ""
        if link is None:
            base.update(
                {
                    "cognitive_feature_status": "NO_CATALOG_LINK",
                    "cognitive_features_present_count": 0,
                    "cognitive_features_missing_count": len(COGNITIVE_FEATURES),
                    "alignment_nonfuture_count": 0,
                    "alignment_exact_visit_count": 0,
                    "alignment_missing_count": len(SOURCE_FILES),
                    "contains_pet_input": False,
                    "contains_raw_identifiers": False,
                }
            )
            rows.append(base)
            continue
        key = (str(link["substudy"]), str(link["bid"]))
        target_day = link["target_day"] if isinstance(link["target_day"], float) else None
        visit = str(link["visit"]) if link["visit"] is not None else None

        pacc_row, pacc_alignment = select_row(sources["pacc"].get(key, []), target_day=target_day, visit=visit)
        cdr_row, cdr_alignment = select_row(
            sources["cdr"].get(key, []),
            target_day=target_day,
            visit=visit,
            day_columns=("CDADTC_DAYS_T0",),
        )
        cfi_row, cfi_alignment = select_row(sources["cfi"].get(key, []), target_day=target_day, visit=visit)
        cfisp_row, cfisp_alignment = select_row(sources["cfisp"].get(key, []), target_day=target_day, visit=visit)
        cogstate_row, cogstate_alignment = select_row(
            sources["cogstate"].get(key, []),
            target_day=target_day,
            visit=visit,
            day_columns=("Date_DAYS_T0", "TDate_DAYS_T0", "TESTDATE_DAYS_T0"),
        )
        psychwell_row, psychwell_alignment = select_row(sources["psychwell"].get(key, []), target_day=target_day, visit=visit)
        concerns_row, concerns_alignment = select_row(sources["concerns"].get(key, []), target_day=target_day, visit=visit)
        ies_row, ies_alignment = select_row(
            sources["ies"].get(key, []),
            target_day=target_day,
            visit=visit,
            day_columns=("IEDATE_DAYS_T0",),
        )

        alignments = [
            pacc_alignment,
            cdr_alignment,
            cfi_alignment,
            cfisp_alignment,
            cogstate_alignment,
            psychwell_alignment,
            concerns_alignment,
            ies_alignment,
        ]
        alignment_counts.update(alignments)
        add_pacc(base, pacc_row)
        add_cdr(base, cdr_row)
        add_complaints(base, cfi_row, cfisp_row)
        add_cogstate(base, cogstate_row)
        add_mood_anxiety(base, psychwell_row, concerns_row, ies_row)
        present = sum(1 for feature in COGNITIVE_FEATURES if base.get(feature) != "")
        base.update(
            {
                "cognitive_feature_status": "OK" if present else "NO_FEATURES",
                "cognitive_features_present_count": present,
                "cognitive_features_missing_count": len(COGNITIVE_FEATURES) - present,
                "alignment_nonfuture_count": sum(1 for value in alignments if value == "nearest_nonfuture"),
                "alignment_exact_visit_count": sum(1 for value in alignments if value == "exact_visit"),
                "alignment_missing_count": sum(1 for value in alignments if value == "missing"),
                "contains_pet_input": False,
                "contains_raw_identifiers": False,
            }
        )
        rows.append(base)

    fieldnames = [
        "sample_hash",
        "cognitive_feature_status",
        "cognitive_features_present_count",
        "cognitive_features_missing_count",
        "alignment_nonfuture_count",
        "alignment_exact_visit_count",
        "alignment_missing_count",
        *COGNITIVE_FEATURES,
        "contains_pet_input",
        "contains_raw_identifiers",
    ]
    write_rows(Path(args.out), rows, fieldnames)
    status_counts = Counter(str(row["cognitive_feature_status"]) for row in rows)
    summary = {
        "rows": len(rows),
        "features": COGNITIVE_FEATURES,
        "feature_families": {
            "core": CORE_COGNITIVE_FEATURES,
            "clinical_severity": CLINICAL_SEVERITY_FEATURES,
            "complaints": COMPLAINT_FEATURES,
            "cogstate": COGSTATE_FEATURES,
            "mood_anxiety": MOOD_ANXIETY_FEATURES,
        },
        "status_counts": dict(status_counts),
        "alignment_counts": dict(alignment_counts),
        "forbidden_input_terms": FORBIDDEN_INPUT_TERMS,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "output_granularity": "hashed_sample_level",
        "method_x_cog_exploratory": True,
        "method_x_claim_unchanged": True,
    }
    atomic_write_json(args.summary, summary)
    md_path = Path(args.summary).with_suffix(".md")
    lines = [
        "# Cognitive Feature Summary\n\n",
        f"- rows: `{summary['rows']}`\n",
        f"- status_counts: `{summary['status_counts']}`\n",
        f"- alignment_counts: `{summary['alignment_counts']}`\n",
        "- contains_pet_input: `False`\n",
        "- contains_raw_identifiers: `False`\n",
        "- output_granularity: `hashed_sample_level`\n",
        "- method_x_cog_exploratory: `True`\n",
        "- method_x_claim_unchanged: `True`\n",
    ]
    atomic_write_text(md_path, "".join(lines))
    write_run_manifest("results/logs/cognitive_features_manifest.json", stage="cognitive-features", data_root=str(root), extra={"summary": summary})
    StateManager().append_event("cognitive-features", "COMPLETED", "Method X-Cog hashed cognitive features generated.")
    print(json.dumps({"rows": len(rows), "status_counts": dict(status_counts), "alignment_counts": dict(alignment_counts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
