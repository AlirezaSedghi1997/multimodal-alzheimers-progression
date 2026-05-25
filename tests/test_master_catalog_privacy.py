from pathlib import Path


def test_catalog_does_not_store_raw_bid_if_generated():
    p = Path("data/processed/master_catalog.csv")
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8")
    assert "BID" not in text
    assert "PREVBID" not in text

