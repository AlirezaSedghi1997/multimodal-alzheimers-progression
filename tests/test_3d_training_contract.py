import json
from pathlib import Path

import pytest


def test_method_x_3d_forward_contract():
    torch = pytest.importorskip("torch")
    from src.models.backbones.lightweight_3d import MethodX3DClassifier

    model = MethodX3DClassifier(
        base_channels=2,
        latent_dim=16,
        tab_hidden_dim=8,
        fusion="concat",
        backbone="tiny",
    )
    image = torch.zeros(2, 1, 16, 16, 16)
    tabular = torch.zeros(2, 5)
    logits = model(image, tabular)
    assert logits.shape == (2,)


def test_method_x_3d_cross_attention_forward_contract():
    torch = pytest.importorskip("torch")
    from src.models.backbones.lightweight_3d import MethodX3DClassifier

    model = MethodX3DClassifier(
        base_channels=2,
        latent_dim=16,
        tab_hidden_dim=8,
        fusion="cross_attention",
        dropout=0.0,
        backbone="tiny",
    )
    image = torch.zeros(2, 1, 16, 16, 16)
    tabular = torch.zeros(2, 5)
    logits = model(image, tabular)
    assert logits.shape == (2,)


def test_method_x_3d_latent_alignment_contract():
    torch = pytest.importorskip("torch")
    from src.models.backbones.lightweight_3d import MethodX3DClassifier

    model = MethodX3DClassifier(
        base_channels=2,
        latent_dim=16,
        tab_hidden_dim=8,
        fusion="cross_attention",
        dropout=0.0,
        backbone="tiny",
    )
    image = torch.zeros(2, 1, 16, 16, 16)
    tabular = torch.zeros(2, 5)
    logits, mri_latent, tab_latent = model.forward_with_latents(image, tabular)
    assert logits.shape == (2,)
    assert mri_latent.shape == tab_latent.shape == (2, 16)


def test_method_x_3d_synib_loss_smoke():
    torch = pytest.importorskip("torch")
    from src.models.backbones.lightweight_3d import MethodX3DClassifier
    from src.models.losses.synib_synergy_loss import SynIBConfig, synib_multimodal_loss

    model = MethodX3DClassifier(
        base_channels=2,
        latent_dim=16,
        tab_hidden_dim=8,
        fusion="cross_attention",
        dropout=0.0,
        backbone="tiny",
    )
    image = torch.zeros(2, 1, 16, 16, 16)
    tabular = torch.zeros(2, 5)
    targets = torch.tensor([0.0, 1.0])
    logits, mri_latent, tab_latent = model.forward_with_latents(image, tabular)
    mri_latent, tab_raw = model.encode(image, tabular)
    zeros_mri = torch.zeros_like(mri_latent)
    zeros_tab = torch.zeros_like(tab_raw)
    loss, parts = synib_multimodal_loss(
        logits,
        model.logits_from_latents(zeros_mri, tab_raw),
        model.logits_from_latents(mri_latent, zeros_tab),
        targets,
        config=SynIBConfig(lambda_syn=0.5, tabular_dim=5, mri_dim=0),
    )
    assert loss.ndim == 0
    assert parts["total"] >= parts["task_loss"]


def test_3d_metrics_contract_if_generated():
    p = Path("results/final_results/three_d/three_d_metrics.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    agg = obj["aggregate"]
    assert agg["model"] == "MethodX3DClassifier"
    assert agg["fusion"] in {"cross_attention", "concat"}
    assert agg["uses_pet_as_input"] is False
    assert agg["uses_raw_identifiers"] is False
    assert agg["mode"] in {"FULL_3D_TRAINING", "CPU_DEBUG_ONLY"}
    if agg.get("external_like") is not None:
        assert agg["external_like"]["external_used_for_fit"] is False
    if agg["mode"] == "CPU_DEBUG_ONLY":
        assert agg["final_claim_allowed"] is False


def test_corrupt_3d_cache_is_not_current(monkeypatch, tmp_path):
    pytest.importorskip("h5py")

    from scripts import run_project_stage

    bad_cache = tmp_path / "t1_volume_cache_64.h5"
    bad_cache.write_bytes(b"not an hdf5 cache")
    monkeypatch.setitem(run_project_stage.STAGE_ARTIFACTS, "mri-volume-cache", bad_cache)

    assert run_project_stage.artifact_is_current("mri-volume-cache") is False


def test_t1_index_accepts_uppercase_nifti_extensions(tmp_path):
    from scripts.build_mri_volume_cache import build_t1_hash_index

    t1_dir = tmp_path / "A4 Learn" / "T1" / "ImageData"
    t1_dir.mkdir(parents=True)
    (t1_dir / "A4_MR_T1_B000001_001.NII.GZ").write_bytes(b"synthetic")

    assert len(build_t1_hash_index(tmp_path / "A4 Learn")) == 1


def test_preprocessed_index_uses_sample_hash_nifti_names(tmp_path):
    from scripts.build_mri_volume_cache import build_preprocessed_hash_index

    out_dir = tmp_path / "preprocessed"
    out_dir.mkdir()
    (out_dir / "hash_abc.nii.gz").write_bytes(b"synthetic")
    scratch = out_dir / "_scratch" / "hash_abc"
    scratch.mkdir(parents=True)
    (scratch / "n4.nii.gz").write_bytes(b"intermediate")

    index = build_preprocessed_hash_index(out_dir)
    assert index["hash_abc"].name == "hash_abc.nii.gz"
    assert "n4" not in index


def test_volume_normalization_uses_nonzero_brain_mask(monkeypatch, tmp_path):
    import sys
    import types

    import numpy as np

    pytest.importorskip("torch")
    from scripts.build_mri_volume_cache import load_volume

    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[1:3, 1:3, 1:3] = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)

    class FakeImage:
        def get_fdata(self, dtype=np.float32):
            return data.astype(dtype)

    fake_nib = types.SimpleNamespace(load=lambda _: FakeImage())
    monkeypatch.setitem(sys.modules, "nibabel", fake_nib)

    vol = load_volume(tmp_path / "sample.nii.gz", resolution=4, device="cpu")
    assert vol.shape == (1, 4, 4, 4)
    assert np.all(vol[0, data == 0] == 0)
    assert abs(float(vol[0, data != 0].mean())) < 1e-3
