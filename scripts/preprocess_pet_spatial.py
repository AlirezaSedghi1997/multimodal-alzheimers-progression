#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pet_manifest import (
    read_private_manifest,
    discover_pet_files,
    link_candidates_to_catalog,
    write_excluded_manifest,
    canonical_tracer,
    enrich_candidates,
    read_catalog_with_t1_paths,
)
from src.utils.pet_resume import PetResumeManifest
from src.utils.pet_spatial_qc import (
    assert_method_y_output_root, atomic_write_json, atomic_write_text, assess_qc_gates,
    compute_volume_qc, load_yaml, normalize_pet_array, scrub_sensitive_text, sha256_file,
    sha256_text, validate_single_source_forbidden_terms, write_overlay_png, normalized_cross_correlation
)


def _load_nifti(path: Path):
    try:
        import nibabel as nib
    except Exception as e:
        raise RuntimeError("nibabel is required for PET NIfTI preprocessing.") from e
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    arr = img.get_fdata(dtype="float32")
    return img, arr


def _save_nifti_like(path: Path, arr, affine):
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def apply_spatial_alignment(pet_path: Path, t1_path: str | None, out_path: Path, mode: str, allow_identity_for_tests: bool, mni_template: str | None = None) -> dict:
    """Apply literature-aligned spatial preprocessing.

    Training-ready Method Y should use PET->T1->MNI where local toolchain permits.
    identity_for_tests is intentionally blocked unless explicitly allowed.
    """
    if mode == "identity_for_tests":
        if not allow_identity_for_tests:
            raise RuntimeError("identity_for_tests is only allowed with --allow_identity_for_tests.")
        img, arr = _load_nifti(pet_path)
        _save_nifti_like(out_path, arr, img.affine)
        return {"spatial_mode": mode, "status": "PASS_TEST_ONLY", "registration_performed": False, "training_ready_spatial": False}

    if mode == "precomputed_petprep_output":
        img, arr = _load_nifti(pet_path)
        _save_nifti_like(out_path, arr, img.affine)
        return {"spatial_mode": mode, "status": "PASS_PRECOMPUTED", "registration_performed": True, "training_ready_spatial": True}

    if not t1_path:
        raise RuntimeError("T1 path required for PET-to-T1 coregistration.")

    if mode in {"ants_rigid_pet_to_t1", "ants_pet_t1_mni"}:
        try:
            import ants
        except Exception as e:
            raise RuntimeError("antspyx is required for ANTs PET spatial preprocessing, or use precomputed PETPrep/petBrain outputs.") from e
        fixed_t1 = ants.image_read(str(t1_path))
        moving_pet = ants.image_read(str(pet_path))
        pet_to_t1 = ants.registration(fixed=fixed_t1, moving=moving_pet, type_of_transform="Rigid")
        if mode == "ants_rigid_pet_to_t1":
            ants.image_write(pet_to_t1["warpedmovout"], str(out_path))
            return {"spatial_mode": mode, "status": "PASS_ANTS_RIGID_TO_T1", "registration_performed": True, "training_ready_spatial": False, "space": "subject_t1"}
        if not mni_template:
            raise RuntimeError("ants_pet_t1_mni requires --mni_template for training-ready MNI outputs.")
        fixed_mni = ants.image_read(str(mni_template))
        t1_to_mni = ants.registration(fixed=fixed_mni, moving=fixed_t1, type_of_transform="SyN")
        # ANTsPy applies transforms from moving to fixed. Composite order may be backend-sensitive;
        # record transform names in QC and validate with visual overlay before training.
        warped_pet_mni = ants.apply_transforms(
            fixed=fixed_mni,
            moving=moving_pet,
            transformlist=t1_to_mni["fwdtransforms"] + pet_to_t1["fwdtransforms"],
            interpolator="linear",
        )
        ants.image_write(warped_pet_mni, str(out_path))
        return {
            "spatial_mode": mode,
            "status": "PASS_ANTS_PET_T1_MNI",
            "registration_performed": True,
            "training_ready_spatial": True,
            "space": "mni",
            "note": "Visual QC required before training because combined-transform order can be backend-sensitive.",
        }

    if mode == "spm_pet_t1_mni":
        raise RuntimeError("SPM PET->T1->MNI hook is declared but not implemented in this Python patch. Use local project SPM wrapper or precomputed outputs.")

    raise RuntimeError(f"Unknown spatial mode: {mode}")


def process_one(item, args, policy, out_dir: Path) -> dict:
    sample_hash = item.catalog.sample_hash or item.catalog.patient_hash or item.candidate.file_hash
    key = f"{sample_hash}_{item.candidate.file_hash}_{args.tracer}"
    item_dir = out_dir / args.tracer.lower()
    aligned_path = item_dir / "aligned" / f"{sample_hash}.nii.gz"
    norm_path = item_dir / "normalized" / f"{sample_hash}.nii.gz"
    qc_path = item_dir / "qc" / f"{sample_hash}.json"
    overlay_dir = Path("results/qc/method_y_pet/overlays") / args.tracer.lower()

    if norm_path.exists() and not args.overwrite:
        return {"key": key, "sample_hash": sample_hash, "status": "SKIPPED_EXISTS", "output_hash": sha256_file(norm_path)}

    spatial = apply_spatial_alignment(
        pet_path=Path(item.candidate.raw_path), t1_path=item.catalog.t1_path or item.candidate.t1_path or None,
        out_path=aligned_path, mode=args.spatial_mode, allow_identity_for_tests=args.allow_identity_for_tests,
        mni_template=args.mni_template,
    )
    img, arr = _load_nifti(aligned_path)
    qc_before = compute_volume_qc(arr, img.affine)

    # SUVR-image requires real atlas/reference masks; this patch blocks silent fake SUVR.
    if args.normalization == "suvr_image":
        if not args.reference_mask:
            raise RuntimeError("suvr_image normalization requires --reference_mask generated in the same aligned/MNI space.")
        _, ref_mask = _load_nifti(Path(args.reference_mask))
        norm_arr, norm_info = normalize_pet_array(arr, mode="suvr_image", reference_mask=ref_mask > 0)
    else:
        norm_arr, norm_info = normalize_pet_array(arr, mode=args.normalization)

    qc_after = compute_volume_qc(norm_arr, img.affine, centiloid=None)
    # Numeric registration QC placeholder: if T1 is same grid, compute NCC; otherwise visual QC remains required.
    if item.catalog.t1_path:
        try:
            _, t1_arr = _load_nifti(Path(item.catalog.t1_path))
            if t1_arr.shape == norm_arr.shape:
                qc_after["ncc_pet_t1"] = normalized_cross_correlation(norm_arr, t1_arr)
                overlay_status = write_overlay_png(overlay_dir / f"{sample_hash}_pet_on_t1.png", t1_arr, norm_arr, "PET on T1")
            else:
                overlay_status = {"status": "SKIPPED_SHAPE_MISMATCH"}
        except Exception as e:
            overlay_status = {"status": "FAILED", "error_message": scrub_sensitive_text(str(e))}
    else:
        overlay_status = {"status": "MISSING_T1"}

    gate = assess_qc_gates(qc_after, policy, spatial_status=spatial)
    qc_payload = {
        "sample_hash": sample_hash,
        "tracer": args.tracer,
        "tier": item.tier,
        "gap_days": item.catalog.abs_pet_t1_gap_days,
        "gate": gate,
        "qc_before": qc_before,
        "qc_after": qc_after,
        "normalization": norm_info,
        "spatial": spatial,
        "overlay": overlay_status,
        "raw_paths_exported": False,
        "visual_qc_required_for_training": True,
    }
    if gate["status"] != "PASS":
        qc_payload["status"] = "QC_FAIL"
        atomic_write_json(qc_path, qc_payload)
        return {"key": key, "sample_hash": sample_hash, "status": "QC_FAIL", "failures": gate["failures"]}
    _save_nifti_like(norm_path, norm_arr, img.affine)
    qc_payload["status"] = "PASS"
    qc_payload["output_hash"] = sha256_file(norm_path)
    atomic_write_json(qc_path, qc_payload)
    return {"key": key, "sample_hash": sample_hash, "status": "PASS", "output_hash": sha256_file(norm_path), "space": spatial.get("space")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--policy", default="configs/model/method_y_spatial_preprocess_policy.yaml")
    ap.add_argument("--out_root", default="data/preprocessed_pet")
    ap.add_argument("--tracer", default="fbp", help="fbp, fbb, pib, fmm, ftp")
    ap.add_argument("--tier", default="primary_strict", choices=["gold_standard_30", "primary_strict", "secondary_window", "relaxed_exploratory", "all"])
    ap.add_argument("--normalization", default="percentile_clip_foreground_zscore", choices=["foreground_zscore", "percentile_clip_foreground_zscore", "suvr_image"])
    ap.add_argument("--spatial_mode", default="ants_pet_t1_mni", choices=["ants_pet_t1_mni", "ants_rigid_pet_to_t1", "spm_pet_t1_mni", "precomputed_petprep_output", "identity_for_tests"])
    ap.add_argument("--mni_template", default=None)
    ap.add_argument("--reference_mask", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n_jobs", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_identity_for_tests", action="store_true")
    args = ap.parse_args()

    if args.execute == args.dry_run:
        raise RuntimeError("Choose exactly one of --dry_run or --execute.")
    policy = load_yaml(args.policy)
    validate_single_source_forbidden_terms(policy)
    out_root = assert_method_y_output_root(args.out_root, policy)
    (out_root / "_scratch_pet").mkdir(parents=True, exist_ok=True)
    if args.execute and args.spatial_mode == "identity_for_tests" and not args.allow_identity_for_tests:
        raise RuntimeError("identity_for_tests execute without real spatial alignment is blocked.")
    if args.execute and args.spatial_mode == "ants_pet_t1_mni" and not args.mni_template:
        raise RuntimeError("ants_pet_t1_mni execute requires --mni_template. Use dry_run or precomputed outputs if unavailable.")

    tracer = canonical_tracer(args.tracer)
    manifest_path = Path(args.manifest) if args.manifest else PROJECT_ROOT / "data" / "preprocessed_pet" / "_private" / "manifests" / f"{tracer}_private_manifest.csv"
    candidates = read_private_manifest(manifest_path, tracer=tracer)
    manifest_source = "private_manifest"
    if not candidates and args.data_root:
        candidates = enrich_candidates(discover_pet_files(args.data_root, tracer=tracer))
        manifest_source = "filename_regex_fallback"
    else:
        candidates = enrich_candidates(candidates)
    catalog = read_catalog_with_t1_paths(
        PROJECT_ROOT / args.catalog if not Path(args.catalog).is_absolute() else args.catalog,
        PROJECT_ROOT / "data/preprocessed_mri_96",
    )
    work, excluded, link_summary = link_candidates_to_catalog(catalog, candidates, policy, requested_tier=args.tier, one_per_patient=True)
    if args.limit:
        work = work[: args.limit]
    write_excluded_manifest(PROJECT_ROOT / "data" / "preprocessed_pet" / "_private" / "manifests" / f"{tracer}_excluded_manifest.csv", excluded)

    resume = PetResumeManifest(PROJECT_ROOT / "results" / "logs" / f"method_y_pet_preprocess_resume_{tracer}.json", stage="method_y_pet_preprocess", config={
        "tracer": tracer, "tier": args.tier, "normalization": args.normalization, "spatial_mode": args.spatial_mode,
        "policy_hash": sha256_text(Path(args.policy).read_text(encoding="utf-8")),
    })
    resume.load()
    max_jobs = min(max(args.n_jobs, 1), int(policy.get("parallel_and_resume", {}).get("max_n_jobs", 4)))
    summary = {
        "stage": "method_y_pet_preprocess", "version": "v3.8-literature-aligned", "mode": "dry_run" if args.dry_run else "execute",
        "tracer": tracer, "manifest_source": manifest_source, "candidate_count": len(candidates), "catalog_rows": len(catalog),
        "selected_work_items": len(work), "excluded_items": len(excluded), "link_summary": link_summary, "spatial_mode": args.spatial_mode,
        "normalization": args.normalization, "n_jobs": max_jobs, "raw_paths_exported": False, "train_ready": False,
        "requires_readiness_gate": True, "mni_template_provided": bool(args.mni_template), "reference_mask_provided": bool(args.reference_mask),
    }
    if args.dry_run:
        atomic_write_json(PROJECT_ROOT / "results" / "tables" / "method_y_pet_preprocess_summary.json", summary)
        print("Dry run complete.")
        print(f"Selected work items: {len(work)}")
        return 0

    status_counts = Counter()
    to_run = []
    for item in work:
        key = f"{item.catalog.sample_hash}_{item.candidate.file_hash}_{tracer}"
        if resume.is_done(key) and not args.overwrite:
            status_counts["RESUMED_SKIPPED"] += 1
            continue
        to_run.append(item)

    def safe(item):
        key = f"{item.catalog.sample_hash}_{item.candidate.file_hash}_{tracer}"
        try:
            result = process_one(item, args, policy, out_root)
        except Exception as e:
            result = {"key": key, "sample_hash": item.catalog.sample_hash, "status": "FAIL", "error_type": type(e).__name__, "error_message": scrub_sensitive_text(str(e))}
        return key, result

    if max_jobs == 1:
        results = [safe(item) for item in to_run]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) as ex:
            results = list(ex.map(safe, to_run))
    for key, result in results:
        status_counts[result.get("status", "UNKNOWN")] += 1
        resume.mark(key, result.get("status", "UNKNOWN"), result)
        resume.save()

    summary["status_counts"] = dict(status_counts)
    summary["processed_items"] = len(results)
    summary["verdict"] = "PASS_WITH_QC_OUTPUTS_NOT_YET_TRAINING_READY_UNTIL_READINESS_GATE" if status_counts.get("PASS", 0) > 0 else "NO_PASSING_OUTPUTS"
    atomic_write_json(PROJECT_ROOT / "results" / "tables" / "method_y_pet_preprocess_summary.json", summary)
    atomic_write_text(PROJECT_ROOT / "results" / "tables" / "method_y_pet_preprocess_summary.md", "# Method Y PET Preprocess Summary\n\n" + "\n".join([f"- {k}: `{v}`" for k,v in summary.items() if k != "link_summary"]) + "\n")
    print("Execute complete.")
    print(dict(status_counts))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
