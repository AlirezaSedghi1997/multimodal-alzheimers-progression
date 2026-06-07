#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pet_path_resolver import discover_pet_from_project_inventory, attach_native_t1_paths_to_candidates
from src.utils.pet_manifest import (
    discover_pet_files, read_catalog, read_private_manifest, write_private_manifest,
    link_candidates_to_catalog, write_excluded_manifest, canonical_tracer,
    enrich_candidates, read_catalog_with_t1_paths,
)
from src.utils.pet_spatial_qc import atomic_write_json, atomic_write_text, load_yaml, scrub_sensitive_text


def main() -> int:
    ap = argparse.ArgumentParser(description="Build private Method Y PET manifest with literature-aligned gap/tracer metadata.")
    ap.add_argument("--data_root", required=False, default="")
    ap.add_argument("--tracer", default="fbp", help="fbp, fbb, pib, fmm, ftp")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--policy", default="configs/model/method_y_spatial_preprocess_policy.yaml")
    ap.add_argument("--input_manifest", default=None, help="Existing private manifest or metadata CSV, preferred over regex discovery.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--requested_tier", default="primary_strict")
    ap.add_argument("--allow_regex_fallback", action="store_true", help="Allow filename regex discovery when project inventory is empty.")
    args = ap.parse_args()

    tracer = canonical_tracer(args.tracer)
    policy = load_yaml(args.policy)
    out_path = Path(args.out) if args.out else PROJECT_ROOT / "data" / "preprocessed_pet" / "_private" / "manifests" / f"{tracer}_private_manifest.csv"
    excluded_path = out_path.with_name(f"{tracer}_excluded_manifest.csv")

    if args.input_manifest:
        candidates = read_private_manifest(args.input_manifest, tracer=tracer)
        source = "input_private_manifest"
    elif args.data_root:
        candidates = discover_pet_from_project_inventory(args.data_root, tracer=tracer)
        if candidates:
            source = "project_inventory"
        elif args.allow_regex_fallback:
            candidates = discover_pet_files(args.data_root, tracer=tracer)
            source = "filename_regex_fallback"
        else:
            candidates = []
            source = "none"
    else:
        candidates = []
        source = "none"

    candidates = enrich_candidates(candidates)
    if args.data_root and candidates:
        candidates = attach_native_t1_paths_to_candidates(candidates, args.data_root)
    write_private_manifest(out_path, candidates)
    catalog = read_catalog_with_t1_paths(
        PROJECT_ROOT / args.catalog if not Path(args.catalog).is_absolute() else args.catalog,
        PROJECT_ROOT / "data" / "preprocessed_mri_96",
    )
    work, excluded, link_summary = link_candidates_to_catalog(catalog, candidates, policy, requested_tier=args.requested_tier, one_per_patient=True)
    write_excluded_manifest(excluded_path, excluded)

    counts = Counter([c.substudy for c in candidates])
    tracer_counts = Counter([c.tracer for c in candidates])
    gap_counts = Counter([wi.tier for wi in work])
    summary = {
        "stage": "method_y_pet_manifest",
        "tracer": tracer,
        "source": source,
        "private_manifest_path_redacted": "data/preprocessed_pet/_private/manifests/<tracer>_private_manifest.csv",
        "excluded_manifest_path_redacted": "data/preprocessed_pet/_private/manifests/<tracer>_excluded_manifest.csv",
        "n_candidates": len(candidates),
        "counts_by_substudy": dict(counts),
        "counts_by_tracer": dict(tracer_counts),
        "link_summary": link_summary,
        "linked_gap_tiers": dict(gap_counts),
        "raw_paths_exported": False,
        "regex_fallback_used": source == "filename_regex_fallback",
        "project_inventory_used": source == "project_inventory",
        "training_ready_manifest": source in {"input_private_manifest", "project_inventory"} and link_summary.get("linked_work_items", 0) > 0,
        "warning": "Filename regex fallback is allowed for dry-run only; training-ready PET requires private manifest/project resolver.",
    }
    atomic_write_json(PROJECT_ROOT / "results" / "tables" / "method_y_pet_manifest_summary.json", summary)
    atomic_write_text(PROJECT_ROOT / "results" / "tables" / "method_y_pet_manifest_summary.md", "# Method Y PET Manifest Summary\n\n" + "\n".join([f"- {k}: `{scrub_sensitive_text(v)}`" for k, v in summary.items()]) + "\n")
    print(f"Private manifest written: {out_path}")
    print(f"Candidates: {len(candidates)} | Linked: {link_summary.get('linked_work_items', 0)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
