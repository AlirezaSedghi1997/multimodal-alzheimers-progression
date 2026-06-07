from pathlib import Path

from src.utils.pet_path_resolver import discover_pet_from_project_inventory, parse_structured_pet_path


def test_parse_structured_pet_path():
    candidate = parse_structured_pet_path("H:/A4 Learn/FBP/ImageData/A4_PET_FBP_B10081264_002.nii.gz", tracer="fbp")
    assert candidate is not None
    assert candidate.substudy == "A4"
    assert candidate.bid == "B10081264"
    assert candidate.visit_code == "002"
    assert candidate.source == "project_inventory"


def test_discover_pet_from_project_inventory_when_data_present():
    root = Path(r"H:\A4 Learn")
    image_dir = root / "FBP" / "ImageData"
    if not image_dir.is_dir():
        return
    sample = next(image_dir.glob("A4_PET_FBP_*.nii.gz"), None)
    assert sample is not None
    parsed = parse_structured_pet_path(sample, tracer="fbp")
    assert parsed is not None
    assert parsed.source == "project_inventory"
    assert Path(parsed.raw_path).exists()
