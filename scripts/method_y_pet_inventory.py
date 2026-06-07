from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import REQUIRED_RELATIVE_FILES, resolve_data_root, resolve_required_file
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


PET_IMAGE_FOLDERS = {
    "amyloid_fbp": "FBP",
    "tau_ftp": "FTP",
}
IMAGE_SUFFIXES = (".nii", ".nii.gz")
FORBIDDEN_MODEL_INPUTS = ("pet_image", "suvr", "centiloid", "amyloid_pet", "tau_pet", "pet_derived_feature")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_label(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "pos", "positive", "ab_pos", "a+", "elevated"}:
        return 1
    if text in {"0", "neg", "negative", "ab_neg", "a-", "not elevated", "nonelevated"}:
        return 0
    if "pos" in text or "elev" in text:
        return 1
    if "neg" in text or "non" in text:
        return 0
    return None


def load_locked_catalog(path: Path) -> tuple[set[str], set[str], list[dict[str, str]]]:
    rows = read_csv_rows(path)
    sample_hashes = {r.get("sample_hash", "") for r in rows if r.get("sample_hash")}
    patient_hashes = {r.get("patient_hash", "") for r in rows if r.get("patient_hash")}
    return sample_hashes, patient_hashes, rows


def pet_t1_gap_bins(rows: list[dict[str, str]]) -> dict[str, int]:
    bins = Counter({"missing": 0, "<=60": 0, "61-90": 0, "91-120": 0, ">120": 0})
    for row in rows:
        gap = safe_float(row.get("abs_pet_t1_gap_days"))
        if gap is None:
            bins["missing"] += 1
        elif gap <= 60:
            bins["<=60"] += 1
        elif gap <= 90:
            bins["61-90"] += 1
        elif gap <= 120:
            bins["91-120"] += 1
        else:
            bins[">120"] += 1
    return dict(bins)


def summarize_label_source(rows: list[dict[str, str]], locked_samples: set[str]) -> dict[str, object]:
    locked_rows = []
    overall_counts: Counter[str] = Counter()
    consensus_counts: Counter[str] = Counter()
    agreement = Counter({"both_present": 0, "agree": 0, "disagree": 0})
    pmod_present = 0
    for row in rows:
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        viscode = row.get("VISCODE") or ""
        sample_hash = stable_hash(substudy + "|" + bid + "|" + str(viscode), prefix="s")
        overall = normalize_label(row.get("overall_score"))
        consensus = normalize_label(row.get("consensus"))
        if overall is not None:
            overall_counts[str(overall)] += 1
        if consensus is not None:
            consensus_counts[str(consensus)] += 1
        if safe_float(row.get("pmod_suvr")) is not None:
            pmod_present += 1
        if overall is not None and consensus is not None:
            agreement["both_present"] += 1
            if overall == consensus:
                agreement["agree"] += 1
            else:
                agreement["disagree"] += 1
        if sample_hash in locked_samples:
            locked_rows.append(row)
    return {
        "source_rows": len(rows),
        "locked_catalog_label_rows_matched_in_memory": len(locked_rows),
        "overall_label_counts": dict(overall_counts),
        "consensus_label_counts": dict(consensus_counts),
        "overall_consensus_agreement": dict(agreement),
        "pmod_suvr_present_rows": pmod_present,
        "pet_derived_columns_detected": ["ligand", "pmod_suvr", "consensus"],
        "pet_derived_columns_used_as_model_inputs": False,
    }


def summarize_tau_source(rows: list[dict[str, str]], locked_patients: set[str]) -> dict[str, object]:
    locked_patient_hashes = set()
    locked_rows = 0
    region_counts: Counter[str] = Counter()
    numeric_rows = 0
    for row in rows:
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        ph = stable_hash(substudy + "|" + bid, prefix="pt")
        if ph in locked_patients:
            locked_patient_hashes.add(ph)
            locked_rows += 1
        region = (row.get("brain_region") or "").strip()
        if region:
            region_counts[region] += 1
        if any(safe_float(row.get(col)) is not None for col in ("suvr_cer", "suvr_persi", "suvr_crus")):
            numeric_rows += 1
    return {
        "source_rows": len(rows),
        "locked_catalog_patients_with_tau_records_matched_in_memory": len(locked_patient_hashes),
        "locked_catalog_tau_rows_matched_in_memory": locked_rows,
        "brain_region_count": len(region_counts),
        "numeric_suvr_rows": numeric_rows,
        "tau_pet_derived_columns_used_as_model_inputs": False,
    }


def parse_pet_image_filename(name: str) -> tuple[str, str] | None:
    match = re.search(r"(?P<substudy>A4|LEARN).*?(?P<bid>B[0-9]+)", name, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group("substudy").upper(), match.group("bid")


def summarize_pet_images(root: Path, locked_patients: set[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for label, folder in PET_IMAGE_FOLDERS.items():
        base = root / folder
        files_seen = 0
        parseable = 0
        locked_patient_matches = set()
        suffix_counts: Counter[str] = Counter()
        if base.exists():
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                lower = path.name.lower()
                if not lower.endswith(IMAGE_SUFFIXES):
                    continue
                files_seen += 1
                suffix_counts[".nii.gz" if lower.endswith(".nii.gz") else ".nii"] += 1
                parsed = parse_pet_image_filename(path.name)
                if parsed is None:
                    continue
                parseable += 1
                substudy, bid = parsed
                ph = stable_hash(substudy + "|" + bid, prefix="pt")
                if ph in locked_patients:
                    locked_patient_matches.add(ph)
        out[label] = {
            "directory_present": base.exists(),
            "image_file_count": files_seen,
            "parseable_filename_count": parseable,
            "locked_catalog_patient_matches_in_memory": len(locked_patient_matches),
            "suffix_counts": dict(suffix_counts),
            "paths_exported": False,
            "images_copied": False,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    locked_samples, locked_patients, catalog_rows = load_locked_catalog(Path(args.catalog))
    if not catalog_rows:
        raise RuntimeError("Missing locked Method X catalog. Run build_master_catalog.py first.")

    ab_rows = read_csv_rows(resolve_required_file(root, "ab_label"))
    tau_rows = read_csv_rows(resolve_required_file(root, "tau"))
    label_counts = Counter(str(r.get("ab_label")) for r in catalog_rows)
    substudy_counts = Counter(str(r.get("substudy")) for r in catalog_rows)
    report = {
        "schema_version": 1,
        "method": "Method Y PET-assisted exploratory",
        "status": "COMPLETED",
        "scope": "aggregate_inventory_qc_sensitivity_only",
        "method_x_claim_unchanged": True,
        "method_x_cog_claim_unchanged": True,
        "pet_as_method_x_input": False,
        "pet_as_method_x_cog_input": False,
        "pet_image_modeling_allowed_under_method_x": False,
        "forbidden_model_inputs": FORBIDDEN_MODEL_INPUTS,
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
        "subject_level_outputs_exported": False,
        "locked_catalog": {
            "rows": len(catalog_rows),
            "patients": len(locked_patients),
            "label_counts": dict(label_counts),
            "substudy_counts": dict(substudy_counts),
            "pet_t1_gap_bins_days": pet_t1_gap_bins(catalog_rows),
        },
        "amyloid_pet_label_source": summarize_label_source(ab_rows, locked_samples),
        "tau_pet_source": summarize_tau_source(tau_rows, locked_patients),
        "pet_image_inventory": summarize_pet_images(root, locked_patients),
        "aggregate_sensitivity": {
            "locked_catalog_uses_pet_label_as_ground_truth": True,
            "label_reproducibility_checked_aggregate_only": True,
            "pet_derived_values_used_for_model_training": False,
            "pet_images_used_for_model_training": False,
        },
        "interpretation": (
            "This report supports Method Y inventory/QC only. It must not be cited as a Method X model input "
            "or as evidence that PET-derived features were used for prediction."
        ),
    }

    atomic_write_json("results/tables/method_y_pet_inventory.json", report)
    image_lines = []
    for name, item in report["pet_image_inventory"].items():
        image_lines.append(
            f"| {name} | {item['directory_present']} | {item['image_file_count']} | "
            f"{item['parseable_filename_count']} | {item['locked_catalog_patient_matches_in_memory']} |\n"
        )
    md = [
        "# Method Y PET Inventory/QC Summary\n\n",
        "- status: `COMPLETED`\n",
        "- scope: `aggregate_inventory_qc_sensitivity_only`\n",
        "- Method X claim unchanged: `True`\n",
        "- PET as Method X input: `False`\n",
        "- PET as Method X-Cog input: `False`\n",
        "- raw identifiers exported: `False`\n",
        "- raw paths exported: `False`\n",
        "- subject-level outputs exported: `False`\n\n",
        "## Locked Catalog\n\n",
        f"- rows: `{len(catalog_rows)}`\n",
        f"- patients: `{len(locked_patients)}`\n",
        f"- label_counts: `{dict(label_counts)}`\n",
        f"- substudy_counts: `{dict(substudy_counts)}`\n",
        f"- PET/T1 gap bins: `{report['locked_catalog']['pet_t1_gap_bins_days']}`\n\n",
        "## PET Image Inventory\n\n",
        "| tracer_folder | directory_present | image_file_count | parseable_filename_count | locked_patient_matches |\n",
        "|---|---:|---:|---:|---:|\n",
        *image_lines,
        "\n## Aggregate Sensitivity\n\n",
        f"- amyloid PET label source rows: `{report['amyloid_pet_label_source']['source_rows']}`\n",
        f"- locked catalog label rows matched in memory: `{report['amyloid_pet_label_source']['locked_catalog_label_rows_matched_in_memory']}`\n",
        f"- overall/consensus agreement counts: `{report['amyloid_pet_label_source']['overall_consensus_agreement']}`\n",
        f"- tau PET source rows: `{report['tau_pet_source']['source_rows']}`\n",
        f"- locked catalog patients with tau records matched in memory: `{report['tau_pet_source']['locked_catalog_patients_with_tau_records_matched_in_memory']}`\n\n",
        "## Boundary\n\n",
        "PET image, SUVR, centiloid, amyloid, and tau-derived values remain forbidden as Method X and Method X-Cog model inputs. This artifact is aggregate-only.\n",
    ]
    atomic_write_text("results/tables/method_y_pet_inventory.md", "".join(md))
    write_run_manifest("results/logs/method_y_pet_inventory_manifest.json", stage="method_y_pet_inventory", data_root=str(root), extra={"status": "COMPLETED"})
    StateManager().append_event("method_y_pet_inventory", "COMPLETED", "Aggregate Method Y PET inventory/QC report generated.")
    print(json.dumps({"status": "COMPLETED", "locked_rows": len(catalog_rows), "pet_as_input": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
