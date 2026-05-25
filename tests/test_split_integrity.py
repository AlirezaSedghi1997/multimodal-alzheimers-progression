import csv
from pathlib import Path


def _read(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_split_files_have_no_patient_overlap_if_generated():
    split_dir = Path("splits")
    if not split_dir.exists():
        return
    external = _read(split_dir / "external_holdout.csv")
    external_ids = {r["patient_hash"] for r in external}
    assert external_ids
    for test_path in split_dir.glob("fold_*_test.csv"):
        fold = test_path.name.split("_")[1]
        train = _read(split_dir / f"fold_{fold}_train.csv")
        test = _read(test_path)
        train_ids = {r["patient_hash"] for r in train}
        test_ids = {r["patient_hash"] for r in test}
        assert train_ids.isdisjoint(test_ids)
        assert external_ids.isdisjoint(train_ids)
        assert external_ids.isdisjoint(test_ids)


def test_split_summary_marks_pseudo_external_if_generated():
    summary = Path("splits/split_summary.md")
    if not summary.exists():
        return
    text = summary.read_text(encoding="utf-8")
    assert "pseudo-external" in text.lower()
    assert "cohort_label_confound" in text

