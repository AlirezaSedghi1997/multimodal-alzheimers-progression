from pathlib import Path


def test_no_raw_identifier_columns_in_catalog_if_generated():
    catalog = Path("data/processed/master_catalog.csv")
    if not catalog.exists():
        return
    header = catalog.read_text(encoding="utf-8").splitlines()[0].lower()
    forbidden = ["bid", "prevbid", "patient_id", "sample_id"]
    assert not any(col in header for col in forbidden)


def test_pet_is_not_declared_as_input():
    config = Path("configs/config.yaml").read_text(encoding="utf-8").lower()
    assert "pet: label_or_target_only" in config
    assert "pet as model input" not in config
