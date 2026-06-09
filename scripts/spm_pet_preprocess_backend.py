from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.spm_preprocess_backend import (
    gzip_file,
    is_windows_matlab,
    prepare_input_nii,
    prepare_pet_input_nii,
    run_matlab_batch,
    write_spm_pet_mni_write_batch,
    write_spm_t1_deform_batch,
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def make_spm_pet_scratch(scratch_root: Path, sample_hash: str) -> Path:
    scratch_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(100):
        scratch = scratch_root / f"{sample_hash}_{utc_stamp()}_{attempt:02d}"
        try:
            scratch.mkdir(parents=True, exist_ok=False)
            return scratch
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate unique SPM PET scratch directory for {sample_hash}")


def coreg_pet_to_t1_ants(*, pet_nii: Path, t1_nii: Path, coreg_pet_nii: Path) -> None:
    try:
        import ants
        import nibabel as nib
        import numpy as np
    except Exception as e:
        raise RuntimeError("antspyx and nibabel are required for PET->T1 rigid coregistration before SPM MNI write.") from e

    fixed_t1 = ants.image_read(str(t1_nii))
    moving_pet = ants.image_read(str(pet_nii))
    pet_to_t1 = ants.registration(fixed=fixed_t1, moving=moving_pet, type_of_transform="Rigid")
    coreg_pet_nii.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(pet_to_t1["warpedmovout"], str(coreg_pet_nii))
    arr = nib.load(str(coreg_pet_nii)).get_fdata(dtype=np.float32)
    mask = np.isfinite(arr) & (arr != 0)
    if mask.sum() == 0 or float(arr[mask].std()) <= 0.0:
        raise RuntimeError("ANTs PET->T1 coregistration produced empty or invalid output.")


def preprocess_pet_spm_t1_mni(
    *,
    sample_hash: str,
    pet_path: Path,
    t1_path: Path,
    out_path: Path,
    spm_dir: Path,
    matlab_cmd: str,
    scratch_root: Path,
    t1_deform_cache_root: Path,
) -> dict:
    """SPM/MNI PET preprocessing: ANTs rigid PET->T1, SPM T1 deform cache, SPM normalise.write."""
    scratch = make_spm_pet_scratch(scratch_root, sample_hash)
    t1_nii = scratch / "t1.nii"
    pet_nii = scratch / "pet.nii"
    coreg_pet_nii = scratch / "pet_coreg_t1.nii"
    output_nii = scratch / "pet_mni.nii"
    deform_batch = scratch / "run_spm_t1_deform.m"
    mni_batch = scratch / "run_spm_pet_mni_write.m"
    qc_json = scratch / "spm_pet_qc.json"
    deform_log = scratch / "matlab_spm_t1_deform.log"
    mni_log = scratch / "matlab_spm_pet_mni.log"
    deform_cache = t1_deform_cache_root / sample_hash / "y_deform.nii"
    deform_cache.parent.mkdir(parents=True, exist_ok=True)
    used_cached_t1_deform = deform_cache.exists()

    prepare_input_nii(t1_path, t1_nii)
    prepare_pet_input_nii(pet_path, pet_nii)
    coreg_pet_to_t1_ants(pet_nii=pet_nii, t1_nii=t1_nii, coreg_pet_nii=coreg_pet_nii)

    write_spm_t1_deform_batch(
        batch_path=deform_batch,
        spm_dir=spm_dir,
        t1_nii=t1_nii,
        deform_cache=deform_cache,
        windows_matlab=is_windows_matlab(matlab_cmd),
        sample_hash=sample_hash,
    )
    code = run_matlab_batch(matlab_cmd, deform_batch, log_path=deform_log)
    if code != 0:
        raise RuntimeError(f"SPM T1 deformation extraction failed with matlab_returncode_{code}")
    if not deform_cache.exists():
        raise RuntimeError("SPM T1 deformation cache missing after MATLAB batch.")

    write_spm_pet_mni_write_batch(
        batch_path=mni_batch,
        spm_dir=spm_dir,
        coreg_pet_nii=coreg_pet_nii,
        output_nii=output_nii,
        deform_cache=deform_cache,
        windows_matlab=is_windows_matlab(matlab_cmd),
        sample_hash=sample_hash,
        qc_json=qc_json,
    )
    code = run_matlab_batch(matlab_cmd, mni_batch, log_path=mni_log)
    if code != 0:
        raise RuntimeError(f"SPM PET MNI write failed with matlab_returncode_{code}")
    if not output_nii.exists():
        raise RuntimeError("SPM PET MNI output missing after MATLAB batch.")
    if not qc_json.exists():
        raise RuntimeError("SPM PET QC JSON missing after MATLAB batch.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    gzip_file(output_nii, tmp_out)
    tmp_out.replace(out_path)
    spm_qc = json.loads(qc_json.read_text(encoding="utf-8"))
    finite_fraction = float(spm_qc.get("finite_fraction") or 0.0)
    pet_mni_std = float(spm_qc.get("pet_mni_std") or 0.0)
    if finite_fraction < 0.01 or pet_mni_std <= 0.0:
        raise RuntimeError(
            "SPM PET MNI output empty or invalid after coregistration/normalization "
            f"(finite_fraction={finite_fraction:.4f}, pet_mni_std={pet_mni_std:.6f})."
        )
    return {
        "spatial_mode": "spm_pet_t1_mni",
        "status": "PASS_SPM_PET_T1_MNI",
        "registration_performed": True,
        "training_ready_spatial": True,
        "space": "mni",
        "mni_voxel_mm": [2, 2, 2],
        "pet_coreg_backend": "ants_rigid_pet_to_t1",
        "used_cached_t1_deform": used_cached_t1_deform,
        "spm_pet_qc": {
            "qc_status": spm_qc.get("qc_status"),
            "finite_fraction": spm_qc.get("finite_fraction"),
            "pet_mni_std": spm_qc.get("pet_mni_std"),
        },
    }
