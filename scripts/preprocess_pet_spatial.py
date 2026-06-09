#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections import Counter
from pathlib import Path
import sys

import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_spm_toolchain import check_spm_toolchain
from scripts.spm_pet_preprocess_backend import preprocess_pet_spm_t1_mni

from src.utils.pet_path_resolver import attach_native_t1_paths_to_candidates
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
from src.data.a4_schema import resolve_data_root
from src.utils.pet_reference_mask import image_derived_centiloid, infer_cerebellum_reference_mask

def _log(message: str) -> None:
    print(message, flush=True)


def _load_nifti(path: Path):
    try:
        import nibabel as nib
    except Exception as e:
        raise RuntimeError("nibabel is required for PET NIfTI preprocessing.") from e
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    arr = img.get_fdata(dtype="float32")
    arr = _collapse_pet_to_3d(arr)
    return img, arr


def _collapse_pet_to_3d(arr):
    import numpy as np

    if getattr(arr, "ndim", 0) == 4:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return np.nanmean(arr, axis=-1)
    return arr


def _prepare_pet_for_ants(pet_path: Path) -> tuple[Path, Path | None]:
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(pet_path)))
    arr = img.get_fdata(dtype="float32")
    if arr.ndim <= 3:
        return pet_path, None
    collapsed = _collapse_pet_to_3d(arr)
    tmp = pet_path.parent / f".tmp3d_{pet_path.stem}.nii.gz"
    _save_nifti_like(tmp, collapsed.astype("float32"), img.affine)
    return tmp, tmp


def _save_nifti_like(path: Path, arr, affine):
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def apply_spatial_alignment(
    pet_path: Path,
    t1_path: str | None,
    out_path: Path,
    mode: str,
    allow_identity_for_tests: bool,
    mni_template: str | None = None,
    *,
    sample_hash: str | None = None,
    matlab_cmd: str | None = None,
    spm_dir: str | None = None,
    scratch_root: Path | None = None,
    t1_deform_cache_root: Path | None = None,
) -> dict:
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
        pet_for_ants, tmp_pet = _prepare_pet_for_ants(pet_path)
        try:
            moving_pet = ants.image_read(str(pet_for_ants))
            pet_to_t1 = ants.registration(fixed=fixed_t1, moving=moving_pet, type_of_transform="Rigid")
        finally:
            if tmp_pet is not None and tmp_pet.exists():
                tmp_pet.unlink()
        collapsed_from_4d = tmp_pet is not None
        if mode == "ants_rigid_pet_to_t1":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ants.image_write(pet_to_t1["warpedmovout"], str(out_path))
            return {
                "spatial_mode": mode,
                "status": "PASS_ANTS_RIGID_TO_T1",
                "registration_performed": True,
                "training_ready_spatial": False,
                "space": "subject_t1",
                "input_collapsed_from_4d": collapsed_from_4d,
            }
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
        out_path.parent.mkdir(parents=True, exist_ok=True)
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
        if not sample_hash:
            raise RuntimeError("spm_pet_t1_mni requires sample_hash.")
        if not matlab_cmd or not spm_dir:
            raise RuntimeError("spm_pet_t1_mni requires --matlab and --spm_dir (or MATLAB_BIN / SPM_DIR env vars).")
        spm_path = Path(spm_dir)
        if not spm_path.exists():
            raise RuntimeError("spm_pet_t1_mni requires a valid --spm_dir.")
        native_t1 = t1_path
        if not native_t1:
            raise RuntimeError("Native T1 path required for SPM PET->T1->MNI preprocessing.")
        return preprocess_pet_spm_t1_mni(
            sample_hash=sample_hash,
            pet_path=pet_path,
            t1_path=Path(native_t1),
            out_path=out_path,
            spm_dir=spm_path,
            matlab_cmd=matlab_cmd,
            scratch_root=scratch_root or Path("data/preprocessed_pet/_scratch_spm_pet"),
            t1_deform_cache_root=t1_deform_cache_root or Path("data/preprocessed_pet/_private/t1_spm_deform_cache"),
        )

    raise RuntimeError(f"Unknown spatial mode: {mode}")


def _item_key(item, tracer: str) -> str:
    sample_hash = item.catalog.sample_hash or item.catalog.patient_hash or item.candidate.file_hash
    return f"{sample_hash}_{item.candidate.file_hash}_{tracer}"


def _output_already_done(norm_path: Path, qc_path: Path, spatial_mode: str, overwrite: bool) -> bool:
    if overwrite or not norm_path.exists():
        return False
    if not qc_path.exists():
        return False
    try:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return qc.get("status") == "PASS" and qc.get("spatial", {}).get("spatial_mode") == spatial_mode


def _pending_work_items(work, resume: PetResumeManifest, tracer: str, overwrite: bool) -> list:
    pending = []
    for item in work:
        key = _item_key(item, tracer)
        if resume.is_done(key) and not overwrite:
            continue
        pending.append(item)
    return pending


def process_one(item, args, policy, out_dir: Path, lookups: dict | None = None) -> dict:
    sample_hash = item.catalog.sample_hash or item.catalog.patient_hash or item.candidate.file_hash
    key = f"{sample_hash}_{item.candidate.file_hash}_{args.tracer}"
    item_dir = out_dir / args.tracer.lower()
    aligned_path = item_dir / "aligned" / f"{sample_hash}.nii.gz"
    norm_path = item_dir / "normalized" / f"{sample_hash}.nii.gz"
    qc_path = item_dir / "qc" / f"{sample_hash}.json"
    overlay_dir = Path("results/qc/method_y_pet/overlays") / args.tracer.lower()

    if _output_already_done(norm_path, qc_path, args.spatial_mode, args.overwrite):
        return {"key": key, "sample_hash": sample_hash, "status": "SKIPPED_EXISTS", "output_hash": sha256_file(norm_path)}

    native_t1_path = item.candidate.t1_path or None
    if args.spatial_mode == "spm_pet_t1_mni" and not native_t1_path:
        raise RuntimeError("Native T1 path missing; rerun with --data_root so native T1 paths can be attached.")

    spatial = apply_spatial_alignment(
        pet_path=Path(item.candidate.raw_path),
        t1_path=native_t1_path if args.spatial_mode == "spm_pet_t1_mni" else (native_t1_path or item.catalog.t1_path or None),
        out_path=aligned_path,
        mode=args.spatial_mode,
        allow_identity_for_tests=args.allow_identity_for_tests,
        mni_template=args.mni_template,
        sample_hash=sample_hash,
        matlab_cmd=args.matlab,
        spm_dir=args.spm_dir,
        scratch_root=Path(args.spm_scratch_root),
        t1_deform_cache_root=Path(args.t1_deform_cache_root),
    )
    img, arr = _load_nifti(aligned_path)
    qc_before = compute_volume_qc(arr, img.affine)
    lookups = lookups or {}
    suvr_policy = lookups.get("suvr_policy") or {}

    ref_mask_arr = None
    image_suvr = None
    image_centiloid = None
    if args.normalization == "suvr_image":
        if args.reference_mask:
            _, ref_mask_arr = _load_nifti(Path(args.reference_mask))
            ref_mask_arr = np.asarray(ref_mask_arr) > 0
        else:
            ref_mask_arr = infer_cerebellum_reference_mask(arr)
        norm_arr, norm_info = normalize_pet_array(arr, mode="suvr_image", reference_mask=ref_mask_arr)
        image_suvr, image_centiloid = image_derived_centiloid(
            arr, ref_mask_arr, tracer=args.tracer, suvr_policy=suvr_policy
        )
    else:
        norm_arr, norm_info = normalize_pet_array(arr, mode=args.normalization)

    qc_after = compute_volume_qc(norm_arr, img.affine, centiloid=None)
    substudy = (item.candidate.substudy or item.catalog.substudy or "").strip().upper()
    bid = (item.candidate.bid or "").strip().upper()
    attach_pet_official_qc_fields(
        qc_after,
        substudy=substudy,
        bid=bid,
        tracer=args.tracer,
        centiloid_lookup=lookups.get("centiloid"),
        suvr_lookup=lookups.get("suvr"),
        suvr_policy=suvr_policy,
    )
    if image_centiloid is not None:
        qc_after["image_composite_suvr"] = image_suvr
        qc_after["derived_centiloid"] = image_centiloid
        qc_after["derived_centiloid_source"] = "image_composite_suvr"
    # Numeric registration QC: use native T1 for rigid/subject-space modes, SPM MNI T1 for MNI PET.
    overlay_t1_path = item.candidate.t1_path or item.catalog.t1_path
    if args.spatial_mode == "spm_pet_t1_mni" and item.catalog.t1_path:
        overlay_t1_path = item.catalog.t1_path
    if overlay_t1_path:
        try:
            _, t1_arr = _load_nifti(Path(overlay_t1_path))
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
    ap.add_argument("--tier", default="primary_feasible_90", choices=["gold_standard_30", "primary_strict", "primary_feasible_90", "secondary_window", "relaxed_exploratory", "all"])
    ap.add_argument("--normalization", default=None, choices=["foreground_zscore", "percentile_clip_foreground_zscore", "suvr_image"])
    ap.add_argument("--spatial_mode", default="spm_pet_t1_mni", choices=["ants_pet_t1_mni", "ants_rigid_pet_to_t1", "spm_pet_t1_mni", "precomputed_petprep_output", "identity_for_tests"])
    ap.add_argument("--mni_template", default=None)
    ap.add_argument("--reference_mask", default=None)
    ap.add_argument("--matlab", default=os.environ.get("MATLAB_BIN"))
    ap.add_argument("--spm_dir", default=os.environ.get("SPM_DIR"))
    ap.add_argument("--spm_scratch_root", default="data/preprocessed_pet/_scratch_spm_pet")
    ap.add_argument("--t1_deform_cache_root", default="data/preprocessed_pet/_private/t1_spm_deform_cache")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=0, help="Process at most N pending items per batch (resume-friendly).")
    ap.add_argument("--auto_batches", action="store_true", help="Keep processing pending items in batches until all work items are completed.")
    ap.add_argument("--status_only", action="store_true", help="Print resume progress and exit without processing.")
    ap.add_argument("--n_jobs", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_identity_for_tests", action="store_true")
    args = ap.parse_args()

    if args.execute == args.dry_run and not args.status_only:
        raise RuntimeError("Choose exactly one of --dry_run or --execute.")
    if args.status_only and not args.execute:
        raise RuntimeError("--status_only requires --execute.")
    if args.limit and args.batch_size > 0:
        raise RuntimeError("Use either --limit for smoke tests or --batch_size for batches, not both.")
    if args.limit and args.auto_batches:
        raise RuntimeError("Use either --limit for smoke tests or --auto_batches for full resumable runs, not both.")
    if args.auto_batches and args.batch_size <= 0:
        args.batch_size = 10
    policy = load_yaml(args.policy)
    if args.normalization is None:
        args.normalization = policy.get("normalization", {}).get("default_mode", "percentile_clip_foreground_zscore")
    validate_single_source_forbidden_terms(policy)
    out_root = assert_method_y_output_root(args.out_root, policy)
    (out_root / "_scratch_pet").mkdir(parents=True, exist_ok=True)
    if args.execute and args.spatial_mode == "identity_for_tests" and not args.allow_identity_for_tests:
        raise RuntimeError("identity_for_tests execute without real spatial alignment is blocked.")
    if args.execute and args.spatial_mode == "ants_pet_t1_mni" and not args.mni_template:
        raise RuntimeError("ants_pet_t1_mni execute requires --mni_template. Use dry_run or precomputed outputs if unavailable.")
    if args.execute and args.spatial_mode == "spm_pet_t1_mni":
        toolchain = check_spm_toolchain(args.matlab, args.spm_dir)
        if toolchain["status"] != "READY":
            raise RuntimeError(
                "spm_pet_t1_mni execute blocked by SPM toolchain: "
                + ", ".join(str(x) for x in toolchain.get("missing", []))
            )
        if args.n_jobs != 1:
            _log("Note: forcing n_jobs=1 for SPM/MATLAB PET preprocessing.")
            args.n_jobs = 1

    _log("Loading PET manifest, catalog, and resume state...")
    tracer = canonical_tracer(args.tracer)
    manifest_path = Path(args.manifest) if args.manifest else PROJECT_ROOT / "data" / "preprocessed_pet" / "_private" / "manifests" / f"{tracer}_private_manifest.csv"
    candidates = read_private_manifest(manifest_path, tracer=tracer)
    manifest_source = "private_manifest"
    if not candidates and args.data_root:
        candidates = enrich_candidates(discover_pet_files(args.data_root, tracer=tracer))
        manifest_source = "filename_regex_fallback"
    else:
        candidates = enrich_candidates(candidates)
    if args.data_root and candidates:
        candidates = attach_native_t1_paths_to_candidates(candidates, args.data_root)
    catalog = read_catalog_with_t1_paths(
        PROJECT_ROOT / args.catalog if not Path(args.catalog).is_absolute() else args.catalog,
        PROJECT_ROOT / "data/preprocessed_mri_96",
    )
    work, excluded, link_summary = link_candidates_to_catalog(catalog, candidates, policy, requested_tier=args.tier, one_per_patient=True)
    if args.limit:
        work = work[: args.limit]
    write_excluded_manifest(PROJECT_ROOT / "data" / "preprocessed_pet" / "_private" / "manifests" / f"{tracer}_excluded_manifest.csv", excluded)

    resume_path = PROJECT_ROOT / "results" / "logs" / f"method_y_pet_preprocess_resume_{tracer}_{args.spatial_mode}.json"
    resume = PetResumeManifest(resume_path, stage="method_y_pet_preprocess", config={
        "tracer": tracer, "tier": args.tier, "normalization": args.normalization, "spatial_mode": args.spatial_mode,
        "policy_hash": sha256_text(Path(args.policy).read_text(encoding="utf-8")),
    })
    try:
        resume.load()
    except RuntimeError as e:
        raise RuntimeError(
            f"{e} Delete or rename {resume_path} only if you intentionally changed tier/spatial_mode/normalization."
        ) from e
    done_count = sum(1 for item in work if resume.is_done(_item_key(item, tracer)))
    pending_count = len(_pending_work_items(work, resume, tracer, args.overwrite))
    _log(
        f"Resume: {done_count}/{len(work)} complete, {pending_count} pending "
        f"(log: {resume_path.as_posix()})."
    )
    if args.status_only:
        _log("Status-only mode; exiting without processing.")
        return 0
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
        _log("Dry run complete.")
        _log(f"Selected work items: {len(work)}")
        return 0

    if pending_count == 0:
        _log("Nothing to do: all linked work items are already complete in the resume log.")
        summary["status_counts"] = {}
        summary["processed_items"] = 0
        summary["batches_completed"] = 0
        summary["initial_done"] = done_count
        summary["final_done"] = done_count
        summary["pending_after_run"] = 0
        summary["auto_batches"] = bool(args.auto_batches)
        summary["batch_size"] = args.batch_size
        summary["verdict"] = "ALL_WORK_ITEMS_ALREADY_COMPLETE"
        atomic_write_json(PROJECT_ROOT / "results" / "tables" / "method_y_pet_preprocess_summary.json", summary)
        return 0

    data_root = resolve_data_root(args.data_root) if args.data_root else None
    lookups: dict = {}
    if data_root:
        lookups["centiloid"] = load_official_centiloid_by_bid(data_root)
        lookups["suvr"] = load_official_suvr_by_bid(data_root)
        lookups["suvr_policy"] = load_suvr_reference_policy()

    def safe(item):
        key = _item_key(item, tracer)
        sample_hash = item.catalog.sample_hash or item.catalog.patient_hash or item.candidate.file_hash
        _log(f"Processing {sample_hash} ...")
        try:
            result = process_one(item, args, policy, out_root, lookups=lookups)
        except Exception as e:
            result = {
                "key": key,
                "sample_hash": item.catalog.sample_hash,
                "status": "FAIL",
                "error_type": type(e).__name__,
                "error_message": scrub_sensitive_text(str(e)),
            }
        _log(f"Finished {sample_hash}: {result.get('status', 'UNKNOWN')}")
        return key, result

    total_status = Counter()
    batches_completed = 0
    attempted = 0
    initial_done = sum(1 for item in work if resume.is_done(_item_key(item, tracer)))

    while True:
        pending = _pending_work_items(work, resume, tracer, args.overwrite)
        if not pending:
            break
        batch_items = pending[: args.batch_size] if args.batch_size > 0 else pending
        deferred = max(0, len(pending) - len(batch_items))
        if not batch_items:
            break
        batches_completed += 1
        done_now = sum(1 for item in work if resume.is_done(_item_key(item, tracer)))
        _log(
            f"Batch {batches_completed}: running {len(batch_items)} item(s); "
            f"{done_now}/{len(work)} done; {deferred} deferred to next batch."
        )

        if max_jobs == 1:
            results = [safe(item) for item in batch_items]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) as ex:
                results = list(ex.map(safe, batch_items))

        batch_status = Counter()
        for key, result in results:
            batch_status[result.get("status", "UNKNOWN")] += 1
            total_status[result.get("status", "UNKNOWN")] += 1
            resume.mark(key, result.get("status", "UNKNOWN"), result)
            resume.save()
        attempted += len(results)
        _log(f"Batch {batches_completed} complete: {dict(batch_status)}")
        if not args.auto_batches:
            break

    summary["status_counts"] = dict(total_status)
    summary["processed_items"] = attempted
    summary["batches_completed"] = batches_completed
    summary["initial_done"] = initial_done
    summary["final_done"] = sum(1 for item in work if resume.is_done(_item_key(item, tracer)))
    summary["pending_after_run"] = len(_pending_work_items(work, resume, tracer, args.overwrite))
    summary["auto_batches"] = bool(args.auto_batches)
    summary["batch_size"] = args.batch_size
    summary["verdict"] = (
        "PASS_WITH_QC_OUTPUTS_NOT_YET_TRAINING_READY_UNTIL_READINESS_GATE"
        if total_status.get("PASS", 0) > 0 or summary["final_done"] > 0
        else "NO_PASSING_OUTPUTS"
    )
    atomic_write_json(PROJECT_ROOT / "results" / "tables" / "method_y_pet_preprocess_summary.json", summary)
    atomic_write_text(
        PROJECT_ROOT / "results" / "tables" / "method_y_pet_preprocess_summary.md",
        "# Method Y PET Preprocess Summary\n\n"
        + "\n".join([f"- {k}: `{v}`" for k, v in summary.items() if k != "link_summary"])
        + "\n",
    )
    _log("Execute complete.")
    _log(
        {
            "status_counts": dict(total_status),
            "batches_completed": batches_completed,
            "final_done": summary["final_done"],
            "pending_after_run": summary["pending_after_run"],
        }
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
