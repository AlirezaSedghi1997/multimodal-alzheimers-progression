#!/usr/bin/env python3
"""Inventory additional A4 Learn modalities not yet in Method X-Strict pipeline."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.a4_schema import resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash

MODALITY_SPECS = {
    "flair": {
        "dirs": ("Flair", "FLAIR"),
        "pattern": r"^(?P<substudy>A4|LEARN)_MR_FLAIR_(?P<bid>B[0-9]+)_(?P<scan>[0-9]{3})\.nii(\.gz)?$",
        "branch": "mri_roi_wmh_reads",
        "integration": "WMH volumetrics via imaging_FLAIR_WMH_QC.csv; FLAIR NIfTI for T1+FLAIR sensitivity",
        "method_x_input": False,
    },
    "dwi": {
        "dirs": ("DWI",),
        "pattern": r"^(?P<substudy>A4|LEARN)_MR_DWI_(?P<bid>B[0-9]+)_(?P<scan>[0-9]{3})\.nii(\.gz)?$",
        "branch": "exploratory_dwi",
        "integration": "DTI microstructure (FA/MD) as tabular features; small subset n~674",
        "method_x_input": False,
    },
    "fmri_rest": {
        "dirs": ("fmri rest", "fMRI rest", "FMRI"),
        "pattern": r"^(?P<substudy>A4|LEARN)_MR_fMRI_(?P<bid>B[0-9]+)_(?P<scan>[0-9]{3})\.nii(\.gz)?$",
        "branch": "exploratory_connectivity",
        "integration": "Network connectivity summaries; high compute; Phase 3+ only",
        "method_x_input": False,
    },
    "t2": {
        "dirs": ("T2",),
        "pattern": r"^(?P<substudy>A4|LEARN)_MR_T2_(?P<bid>B[0-9]+)_(?P<scan>[0-9]{3})\.nii(\.gz)?$",
        "branch": "exploratory_t2",
        "integration": "Redundant with T1 for amyloid status; skip for Method X",
        "method_x_input": False,
    },
    "t2star": {
        "dirs": ("T2 star", "T2star"),
        "pattern": r"^(?P<substudy>A4|LEARN)_MR_T2star_(?P<bid>B[0-9]+)_(?P<scan>[0-9]{3})\.nii(\.gz)?$",
        "branch": "mri_roi_wmh_reads",
        "integration": "Microhemorrhage/siderosis reads already in imaging_MRI_reads.csv",
        "method_x_input": False,
    },
}

CLINICAL_EXTENSIONS = {
    "imaging_retinal.csv": {"branch": "exploratory_retinal", "method_x_input": False},
    "imaging_MRI_reads.csv": {"branch": "mri_roi_wmh_reads", "method_x_input": False, "status": "integrated_tabular"},
    "imaging_FLAIR_WMH_QC.csv": {"branch": "mri_roi_wmh_reads", "method_x_input": False, "status": "integrated_tabular"},
    "cogstate_battery.csv": {"branch": "method_x_cog", "method_x_input": False, "status": "integrated_cognitive"},
    "cogstate_cpath.csv": {"branch": "method_x_cog", "method_x_input": False},
    "cogstate_macq.csv": {"branch": "method_x_cog", "method_x_input": False},
    "cdr.csv": {"branch": "clinical_endpoint", "method_x_input": False, "status": "feasibility"},
    "PACC.csv": {"branch": "clinical_endpoint", "method_x_input": False, "status": "feasibility"},
    "biomarker_pTau217.csv": {"branch": "bio_cog_method_z", "method_x_input": False, "status": "integrated_blood"},
    "biomarker_Plasma_Roche_Results.csv": {"branch": "bio_cog_method_z", "method_x_input": False, "status": "integrated_blood"},
    "biomarker_AB_Test.csv": {"branch": "bio_cog_method_z", "method_x_input": False, "status": "integrated_blood"},
}


def count_modality(root: Path, spec: dict) -> dict:
    pattern = re.compile(spec["pattern"], re.IGNORECASE)
    count = 0
    substudy_counts: dict[str, int] = {}
    for dir_name in spec["dirs"]:
        mod_dir = root / dir_name
        if not mod_dir.exists():
            continue
        for path in mod_dir.rglob("*.nii*"):
            match = pattern.match(path.name)
            if match:
                count += 1
                sub = match.group("substudy").upper()
                substudy_counts[sub] = substudy_counts.get(sub, 0) + 1
    return {
        "nifti_count": count,
        "substudy_counts": substudy_counts,
        "dirs_found": [d for d in spec["dirs"] if (root / d).exists()],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="")
    ap.add_argument("--out_json", default="results/tables/extended_modality_inventory.json")
    ap.add_argument("--out_md", default="results/tables/extended_modality_inventory.md")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    clinical_ext = root / "Clinical" / "External Data"

    modalities = {}
    for name, spec in MODALITY_SPECS.items():
        stats = count_modality(root, spec)
        modalities[name] = {
            **stats,
            "branch": spec["branch"],
            "integration_rationale": spec["integration"],
            "method_x_input_allowed": spec["method_x_input"],
        }

    clinical = {}
    for fname, meta in CLINICAL_EXTENSIONS.items():
        path = clinical_ext / fname
        clinical[fname] = {
            "exists": path.exists(),
            "rows": sum(1 for _ in path.open(encoding="utf-8-sig")) - 1 if path.exists() else 0,
            **meta,
        }

    report = {
        "stage": "extended_modality_inventory",
        "data_root_hash": stable_hash(str(root), prefix="dr"),
        "modalities": modalities,
        "clinical_extensions": clinical,
        "method_x_strict_inputs": ["T1 MRI", "age", "sex", "education_years", "apoe4_carrier", "mmse_base"],
        "recommended_next_integrations": [
            {"priority": 1, "asset": "flair", "action": "T1+FLAIR WMH sensitivity ablation (tabular WMH already integrated)"},
            {"priority": 2, "asset": "biomarker blood panels", "action": "Re-run build_branch_catalogs with fixed Roche aliases; Method Z ablation"},
            {"priority": 3, "asset": "cdr.csv + PACC.csv", "action": "Clinical endpoint GNOVA surrogate (Phase H)"},
            {"priority": 4, "asset": "dwi", "action": "Exploratory FA/MD extraction for subset with paired T1"},
            {"priority": 5, "asset": "fmri_rest", "action": "Defer — high compute, no amyloid-status literature support at this scale"},
        ],
    }

    out_json = PROJECT_ROOT / args.out_json
    out_md = PROJECT_ROOT / args.out_md
    atomic_write_json(out_json, report)

    lines = [
        "# Extended Modality Inventory",
        "",
        f"Data root hash: `{report['data_root_hash']}`",
        "",
        "## Imaging modalities",
        "",
        "| Modality | NIfTI count | Branch | Method X input |",
        "|----------|------------|--------|----------------|",
    ]
    for name, info in modalities.items():
        lines.append(
            f"| {name} | {info['nifti_count']} | {info['branch']} | "
            f"{'yes' if info['method_x_input_allowed'] else 'no'} |"
        )
    lines.extend(["", "## Clinical extensions", ""])
    for fname, info in clinical.items():
        status = info.get("status", "available")
        lines.append(f"- **{fname}** ({info['rows']} rows): {status} → {info['branch']}")
    lines.extend(["", "## Recommended integrations", ""])
    for item in report["recommended_next_integrations"]:
        lines.append(f"{item['priority']}. **{item['asset']}**: {item['action']}")

    atomic_write_text(out_md, "\n".join(lines) + "\n")
    print(json.dumps({"status": "COMPLETED", "modalities": len(modalities), "out": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
