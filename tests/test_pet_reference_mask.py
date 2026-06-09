import numpy as np

from src.utils.pet_reference_mask import composite_foreground_suvr, infer_cerebellum_reference_mask, image_derived_centiloid


def test_infer_cerebellum_reference_mask_non_empty():
    arr = np.zeros((32, 32, 32), dtype=np.float32)
    arr[8:24, 8:24, 4:16] = 2.0
    mask = infer_cerebellum_reference_mask(arr)
    assert mask.sum() > 0


def test_image_derived_centiloid_fbp():
    arr = np.zeros((32, 32, 32), dtype=np.float32)
    arr[8:24, 8:24, 4:20] = 1.0
    arr[8:24, 20:24, 4:20] = 2.0
    mask = infer_cerebellum_reference_mask(arr)
    suvr = composite_foreground_suvr(arr, mask)
    assert suvr is not None and suvr > 0
    policy = {"amyloid_pet": {"tracers": {"fbp": {"centiloid_slope": 188.22, "centiloid_intercept": -189.16}}}}
    _, cl = image_derived_centiloid(arr, mask, tracer="fbp", suvr_policy=policy)
    assert cl is not None
