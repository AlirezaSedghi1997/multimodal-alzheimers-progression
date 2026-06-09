from __future__ import annotations

import gzip
import os
import shutil
import subprocess
from pathlib import Path


def resolve_command(value: str | None, candidates: list[str]) -> str | None:
    if value:
        p = Path(value)
        if p.exists():
            return str(p)
        found = shutil.which(value)
        if found:
            return found
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def is_windows_matlab(matlab_cmd: str) -> bool:
    return Path(matlab_cmd).name.lower().endswith(".exe")


def wslpath_windows(path: Path) -> str:
    proc = subprocess.run(["wslpath", "-w", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"wslpath failed for {path}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def matlab_path(path: Path, *, windows_matlab: bool) -> str:
    path = path.resolve()
    if windows_matlab and os.name != "nt":
        return wslpath_windows(path)
    return str(path)


def matlab_quote(value: str) -> str:
    return value.replace("'", "''")


def prepare_input_nii(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.name.lower().endswith(".nii.gz"):
        with gzip.open(source, "rb") as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(source, dest)


def prepare_pet_input_nii(source: Path, dest: Path) -> None:
    """Collapse 4D PET to 3D and write uncompressed NIfTI for SPM.

    Preserve native orientation/affine so SPM coregistration matches the subject T1 grid.
    Reorienting PET alone (e.g. as_closest_canonical) breaks SPM estwrite reslicing.
    """
    import nibabel as nib
    import numpy as np

    dest.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(source))
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    if arr.ndim == 4:
        arr = arr[..., 0] if arr.shape[-1] == 1 else np.nanmean(arr, axis=-1)
    out = nib.Nifti1Image(arr.astype(np.float32), img.affine, img.header.copy())
    out.set_data_dtype(np.float32)
    nib.save(out, str(dest))


def gzip_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with source.open("rb") as src, gzip.open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(tmp, dest)


def write_spm_batch(
    *,
    batch_path: Path,
    spm_dir: Path,
    input_nii: Path,
    output_nii: Path,
    windows_matlab: bool,
    sample_hash: str = "unknown_sample",
    qc_json: Path | None = None,
) -> None:
    if qc_json is None:
        qc_json = batch_path.with_suffix(".qc.json")
    spm_dir_m = matlab_quote(matlab_path(spm_dir, windows_matlab=windows_matlab))
    input_m = matlab_quote(matlab_path(input_nii, windows_matlab=windows_matlab))
    output_m = matlab_quote(matlab_path(output_nii, windows_matlab=windows_matlab))
    qc_json_m = matlab_quote(matlab_path(qc_json, windows_matlab=windows_matlab))
    sample_hash_m = matlab_quote(sample_hash)
    tpm_m = matlab_quote(matlab_path(spm_dir / "tpm" / "TPM.nii", windows_matlab=windows_matlab))
    batch_path.write_text(
        f"""
addpath('{spm_dir_m}');
spm('Defaults', 'fmri');
spm_jobman('initcfg');
input_nii = '{input_m}';
output_nii = '{output_m}';
qc_json = '{qc_json_m}';
sample_hash = '{sample_hash_m}';
[work_dir, name, ext] = fileparts(input_nii);
tpm = '{tpm_m}';

matlabbatch = {{}};
matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{[input_nii ',1']}};
matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{{1}}.spm.spatial.preproc.channel.write = [0 1];
for k = 1:6
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).tpm = {{[tpm ',' num2str(k)]}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).ngaus = max(k, 2);
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).native = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).warped = [0 0];
end
matlabbatch{{1}}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{{1}}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{{1}}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{{1}}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{{1}}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{{1}}.spm.spatial.preproc.warp.write = [0 1];
spm_jobman('run', matlabbatch);

bias_path = fullfile(work_dir, ['m' name ext]);
c1_path = fullfile(work_dir, ['c1' name ext]);
c2_path = fullfile(work_dir, ['c2' name ext]);
c3_path = fullfile(work_dir, ['c3' name ext]);
y_path = fullfile(work_dir, ['y_' name ext]);
brain_path = fullfile(work_dir, ['brain_' name ext]);

V = spm_vol(bias_path);
Y = spm_read_vols(V);
c1 = spm_read_vols(spm_vol(c1_path));
c2 = spm_read_vols(spm_vol(c2_path));
c3 = spm_read_vols(spm_vol(c3_path));
tissue_sum = c1 + c2 + c3;
native_brain_mask = tissue_sum > 0.20;
Y(~isfinite(Y)) = 0;
Y(~native_brain_mask) = 0;
V.fname = brain_path;
spm_write_vol(V, Y);

clear matlabbatch;
matlabbatch{{1}}.spm.spatial.normalise.write.subj.def = {{y_path}};
matlabbatch{{1}}.spm.spatial.normalise.write.subj.resample = {{[brain_path ',1']}};
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70; 78 76 85];
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.interp = 4;
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.prefix = 'w';
spm_jobman('run', matlabbatch);

warped_path = fullfile(work_dir, ['wbrain_' name ext]);
Vw = spm_vol(warped_path);
Yw = spm_read_vols(Vw);
Yw(~isfinite(Yw)) = 0;
brain_mask = abs(Yw) > 0;
if any(brain_mask(:))
    clipped = double(Yw);
    vals = sort(clipped(brain_mask));
    nvals = numel(vals);
    lo = vals(max(1, round(0.005 * nvals)));
    hi = vals(min(nvals, max(1, round(0.995 * nvals))));
    if hi > lo
        clipped(clipped < lo & brain_mask) = lo;
        clipped(clipped > hi & brain_mask) = hi;
    end
    norm_vals = clipped(brain_mask);
    mu = mean(norm_vals(:));
    sigma = std(norm_vals(:));
    if ~isfinite(sigma) || sigma < 1e-6
        sigma = 1;
    end
    Yw = zeros(size(Yw), 'single');
    Yw(brain_mask) = single((clipped(brain_mask) - mu) ./ sigma);
else
    Yw = single(Yw);
end
Vw.dt = [16 0];
Vw.pinfo = [1; 0; 0];
spm_write_vol(Vw, Yw);

voxel_volume_ml = abs(det(V.mat(1:3, 1:3))) / 1000;
tissue_total = sum(tissue_sum(native_brain_mask));
if tissue_total > 0
    tissue_fraction_gm = sum(c1(native_brain_mask)) / tissue_total;
    tissue_fraction_wm = sum(c2(native_brain_mask)) / tissue_total;
    tissue_fraction_csf = sum(c3(native_brain_mask)) / tissue_total;
else
    tissue_fraction_gm = 0;
    tissue_fraction_wm = 0;
    tissue_fraction_csf = 0;
end
native_values = Y(native_brain_mask);
if isempty(native_values)
    native_std = 1;
else
    native_std = std(native_values(:));
    if ~isfinite(native_std) || native_std < 1e-6
        native_std = 1;
    end
end
gm_values = Y(c1 > 0.50);
wm_values = Y(c2 > 0.50);
if isempty(gm_values) || isempty(wm_values)
    gm_wm_cnr = 0;
else
    gm_wm_cnr = abs(mean(wm_values(:)) - mean(gm_values(:))) / native_std;
end
norm_values = Yw(brain_mask);
if isempty(norm_values)
    normalized_brain_mean = 0;
    normalized_brain_std = 0;
else
    normalized_brain_mean = mean(norm_values(:));
    normalized_brain_std = std(norm_values(:));
end
brain_volume_ml = nnz(native_brain_mask) * voxel_volume_ml;
warnings = {{}};
if brain_volume_ml < 800 || brain_volume_ml > 1800
    warnings{{end + 1}} = 'brain_volume_outside_expected_range';
end
if tissue_fraction_gm < 0.25 || tissue_fraction_gm > 0.70
    warnings{{end + 1}} = 'gray_matter_fraction_outside_expected_range';
end
if gm_wm_cnr < 0.25
    warnings{{end + 1}} = 'low_gm_wm_contrast_to_noise';
end
if abs(normalized_brain_mean) > 0.15
    warnings{{end + 1}} = 'normalized_brain_mean_not_centered';
end
if normalized_brain_std < 0.75 || normalized_brain_std > 1.25
    warnings{{end + 1}} = 'normalized_brain_std_outside_expected_range';
end
qc = struct();
qc.sample_hash = sample_hash;
if isempty(warnings)
    qc.qc_status = 'PASS';
else
    qc.qc_status = 'WARNING';
end
qc.warnings = warnings;
qc.brain_volume_ml = brain_volume_ml;
qc.brain_mask_voxels = nnz(native_brain_mask);
qc.tissue_fraction_gm = tissue_fraction_gm;
qc.tissue_fraction_wm = tissue_fraction_wm;
qc.tissue_fraction_csf = tissue_fraction_csf;
qc.gm_wm_cnr = gm_wm_cnr;
qc.normalized_brain_mean = normalized_brain_mean;
qc.normalized_brain_std = normalized_brain_std;
qc.normalized_nonzero_fraction = nnz(brain_mask) / numel(brain_mask);
qc.contains_pet_input = false;
qc.contains_raw_identifiers = false;
fid = fopen(qc_json, 'w');
if fid == -1
    error('Could not write QC JSON: %s', qc_json);
end
fprintf(fid, '%s', jsonencode(qc));
fclose(fid);
copyfile(warped_path, output_nii);
exit(0);
""".strip()
        + "\n",
        encoding="utf-8",
    )


def write_spm_pet_t1_mni_batch(
    *,
    batch_path: Path,
    spm_dir: Path,
    t1_nii: Path,
    pet_nii: Path,
    output_nii: Path,
    deform_cache: Path,
    windows_matlab: bool,
    sample_hash: str = "unknown_sample",
    qc_json: Path | None = None,
) -> None:
    if qc_json is None:
        qc_json = batch_path.with_suffix(".qc.json")
    spm_dir_m = matlab_quote(matlab_path(spm_dir, windows_matlab=windows_matlab))
    t1_m = matlab_quote(matlab_path(t1_nii, windows_matlab=windows_matlab))
    pet_m = matlab_quote(matlab_path(pet_nii, windows_matlab=windows_matlab))
    output_m = matlab_quote(matlab_path(output_nii, windows_matlab=windows_matlab))
    deform_cache_m = matlab_quote(matlab_path(deform_cache, windows_matlab=windows_matlab))
    qc_json_m = matlab_quote(matlab_path(qc_json, windows_matlab=windows_matlab))
    sample_hash_m = matlab_quote(sample_hash)
    tpm_m = matlab_quote(matlab_path(spm_dir / "tpm" / "TPM.nii", windows_matlab=windows_matlab))
    batch_path.write_text(
        f"""
addpath('{spm_dir_m}');
spm('Defaults', 'fmri');
spm_jobman('initcfg');
t1_nii = '{t1_m}';
pet_nii = '{pet_m}';
output_nii = '{output_m}';
deform_cache = '{deform_cache_m}';
qc_json = '{qc_json_m}';
sample_hash = '{sample_hash_m}';
[work_dir, t1_name, t1_ext] = fileparts(t1_nii);
[~, pet_name, pet_ext] = fileparts(pet_nii);
tpm = '{tpm_m}';

if exist(deform_cache, 'file')
    y_path = deform_cache;
    used_cached_t1_deform = true;
else
    used_cached_t1_deform = false;
    matlabbatch = {{}};
    matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{[t1_nii ',1']}};
    matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{{1}}.spm.spatial.preproc.channel.write = [0 1];
    for k = 1:6
        matlabbatch{{1}}.spm.spatial.preproc.tissue(k).tpm = {{[tpm ',' num2str(k)]}};
        matlabbatch{{1}}.spm.spatial.preproc.tissue(k).ngaus = max(k, 2);
        matlabbatch{{1}}.spm.spatial.preproc.tissue(k).native = [0 0];
        matlabbatch{{1}}.spm.spatial.preproc.tissue(k).warped = [0 0];
    end
    matlabbatch{{1}}.spm.spatial.preproc.tissue(1).native = [1 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(2).native = [1 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(3).native = [1 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{{1}}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{{1}}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{{1}}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{{1}}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{{1}}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{{1}}.spm.spatial.preproc.warp.write = [0 1];
    spm_jobman('run', matlabbatch);
    y_path = fullfile(work_dir, ['y_' t1_name t1_ext]);
    if ~exist(y_path, 'file')
        error('T1 segmentation did not produce deformation field: %s', y_path);
    end
    [cache_dir, ~, ~] = fileparts(deform_cache);
    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
    copyfile(y_path, deform_cache);
    y_path = deform_cache;
end

matlabbatch = {{}};
matlabbatch{{1}}.spm.spatial.coreg.estwrite.ref = {{[t1_nii ',1']}};
matlabbatch{{1}}.spm.spatial.coreg.estwrite.source = {{[pet_nii ',1']}};
matlabbatch{{1}}.spm.spatial.coreg.estwrite.other = {{''}};
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.interp = 1;
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';
spm_jobman('run', matlabbatch);

pet_coreg = fullfile(work_dir, ['r' pet_name pet_ext]);
if ~exist(pet_coreg, 'file')
    error('PET coregistration output missing: %s', pet_coreg);
end

matlabbatch = {{}};
matlabbatch{{1}}.spm.spatial.normalise.write.subj.def = {{y_path}};
matlabbatch{{1}}.spm.spatial.normalise.write.subj.resample = {{[pet_coreg ',1']}};
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70; 78 76 85];
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.interp = 1;
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.prefix = 'w';
spm_jobman('run', matlabbatch);

[~, coreg_name, coreg_ext] = fileparts(pet_coreg);
pet_mni = fullfile(work_dir, ['w' coreg_name coreg_ext]);
if ~exist(pet_mni, 'file')
    error('PET MNI normalization output missing: %s', pet_mni);
end

V = spm_vol(pet_mni);
Y = spm_read_vols(V);
mask = isfinite(Y) & Y ~= 0;
if any(mask(:))
    vals = Y(mask);
    pet_mean = mean(vals(:));
    pet_std = std(vals(:));
    finite_fraction = nnz(mask) / numel(Y);
else
    pet_mean = 0;
    pet_std = 0;
    finite_fraction = 0;
end
qc = struct();
qc.sample_hash = sample_hash;
qc.qc_status = 'PASS';
if finite_fraction < 0.95 || pet_std < 1e-6
    qc.qc_status = 'WARNING';
end
qc.used_cached_t1_deform = used_cached_t1_deform;
qc.finite_fraction = finite_fraction;
qc.pet_mni_mean = pet_mean;
qc.pet_mni_std = pet_std;
qc.space = 'mni';
qc.mni_voxel_mm = [2 2 2];
qc.contains_pet_input = true;
qc.contains_raw_identifiers = false;
fid = fopen(qc_json, 'w');
if fid == -1
    error('Could not write QC JSON: %s', qc_json);
end
fprintf(fid, '%s', jsonencode(qc));
fclose(fid);
copyfile(pet_mni, output_nii);
exit(0);
""".strip()
        + "\n",
        encoding="utf-8",
    )


def write_spm_t1_deform_batch(
    *,
    batch_path: Path,
    spm_dir: Path,
    t1_nii: Path,
    deform_cache: Path,
    windows_matlab: bool,
    sample_hash: str = "unknown_sample",
) -> None:
    spm_dir_m = matlab_quote(matlab_path(spm_dir, windows_matlab=windows_matlab))
    t1_m = matlab_quote(matlab_path(t1_nii, windows_matlab=windows_matlab))
    deform_cache_m = matlab_quote(matlab_path(deform_cache, windows_matlab=windows_matlab))
    sample_hash_m = matlab_quote(sample_hash)
    tpm_m = matlab_quote(matlab_path(spm_dir / "tpm" / "TPM.nii", windows_matlab=windows_matlab))
    batch_path.write_text(
        f"""
addpath('{spm_dir_m}');
spm('Defaults', 'fmri');
spm_jobman('initcfg');
t1_nii = '{t1_m}';
deform_cache = '{deform_cache_m}';
sample_hash = '{sample_hash_m}';
[work_dir, t1_name, t1_ext] = fileparts(t1_nii);
tpm = '{tpm_m}';

if exist(deform_cache, 'file')
    exit(0);
end

matlabbatch = {{}};
matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{[t1_nii ',1']}};
matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{{1}}.spm.spatial.preproc.channel.write = [0 1];
for k = 1:6
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).tpm = {{[tpm ',' num2str(k)]}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).ngaus = max(k, 2);
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).native = [0 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).warped = [0 0];
end
matlabbatch{{1}}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{{1}}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{{1}}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{{1}}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{{1}}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{{1}}.spm.spatial.preproc.warp.write = [0 1];
spm_jobman('run', matlabbatch);

y_path = fullfile(work_dir, ['y_' t1_name t1_ext]);
if ~exist(y_path, 'file')
    error('T1 segmentation did not produce deformation field: %s', y_path);
end
[cache_dir, ~, ~] = fileparts(deform_cache);
if ~exist(cache_dir, 'dir')
    mkdir(cache_dir);
end
copyfile(y_path, deform_cache);
exit(0);
""".strip()
        + "\n",
        encoding="utf-8",
    )


def write_spm_pet_mni_write_batch(
    *,
    batch_path: Path,
    spm_dir: Path,
    coreg_pet_nii: Path,
    output_nii: Path,
    deform_cache: Path,
    windows_matlab: bool,
    sample_hash: str = "unknown_sample",
    qc_json: Path | None = None,
) -> None:
    if qc_json is None:
        qc_json = batch_path.with_suffix(".qc.json")
    spm_dir_m = matlab_quote(matlab_path(spm_dir, windows_matlab=windows_matlab))
    coreg_pet_m = matlab_quote(matlab_path(coreg_pet_nii, windows_matlab=windows_matlab))
    output_m = matlab_quote(matlab_path(output_nii, windows_matlab=windows_matlab))
    deform_cache_m = matlab_quote(matlab_path(deform_cache, windows_matlab=windows_matlab))
    qc_json_m = matlab_quote(matlab_path(qc_json, windows_matlab=windows_matlab))
    sample_hash_m = matlab_quote(sample_hash)
    batch_path.write_text(
        f"""
addpath('{spm_dir_m}');
spm('Defaults', 'fmri');
spm_jobman('initcfg');
coreg_pet_nii = '{coreg_pet_m}';
output_nii = '{output_m}';
deform_cache = '{deform_cache_m}';
qc_json = '{qc_json_m}';
sample_hash = '{sample_hash_m}';
[work_dir, pet_name, pet_ext] = fileparts(coreg_pet_nii);

if ~exist(deform_cache, 'file')
    error('Missing T1 deformation cache: %s', deform_cache);
end
y_path = deform_cache;

matlabbatch = {{}};
matlabbatch{{1}}.spm.spatial.normalise.write.subj.def = {{y_path}};
matlabbatch{{1}}.spm.spatial.normalise.write.subj.resample = {{[coreg_pet_nii ',1']}};
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70; 78 76 85];
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.interp = 1;
matlabbatch{{1}}.spm.spatial.normalise.write.woptions.prefix = 'w';
spm_jobman('run', matlabbatch);

pet_mni = fullfile(work_dir, ['w' pet_name pet_ext]);
if ~exist(pet_mni, 'file')
    error('PET MNI normalization output missing: %s', pet_mni);
end

V = spm_vol(pet_mni);
Y = spm_read_vols(V);
mask = isfinite(Y) & Y ~= 0;
if any(mask(:))
    vals = Y(mask);
    pet_mean = mean(vals(:));
    pet_std = std(vals(:));
    finite_fraction = nnz(mask) / numel(Y);
else
    pet_mean = 0;
    pet_std = 0;
    finite_fraction = 0;
end
qc = struct();
qc.sample_hash = sample_hash;
qc.qc_status = 'PASS';
if finite_fraction < 0.95 || pet_std < 1e-6
    qc.qc_status = 'WARNING';
end
qc.used_cached_t1_deform = true;
qc.pet_coreg_backend = 'ants_rigid_pet_to_t1';
qc.finite_fraction = finite_fraction;
qc.pet_mni_mean = pet_mean;
qc.pet_mni_std = pet_std;
qc.space = 'mni';
qc.mni_voxel_mm = [2 2 2];
qc.contains_pet_input = true;
qc.contains_raw_identifiers = false;
fid = fopen(qc_json, 'w');
if fid == -1
    error('Could not write QC JSON: %s', qc_json);
end
fprintf(fid, '%s', jsonencode(qc));
fclose(fid);
copyfile(pet_mni, output_nii);
exit(0);
""".strip()
        + "\n",
        encoding="utf-8",
    )


def run_matlab_batch(matlab_cmd: str, batch_path: Path, log_path: Path | None = None) -> int:
    batch_arg = str(batch_path)
    if is_windows_matlab(matlab_cmd) and os.name != "nt":
        batch_arg = wslpath_windows(batch_path)
    batch_expr = f"restoredefaultpath; matlabrc; run('{matlab_quote(batch_arg)}')"
    if log_path is None:
        proc = subprocess.run(
            [matlab_cmd, "-batch", batch_expr],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(
                [matlab_cmd, "-batch", batch_expr],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    return int(proc.returncode)
