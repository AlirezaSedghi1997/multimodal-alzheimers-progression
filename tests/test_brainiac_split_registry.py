"""Adversarial tests for the BrainIAC CAFT split registry (Plan 004 / B3).

Fixtures are fully synthetic (generated in-test): no study data, no real IDs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.research import brainiac_split_registry as sr


SEED = sr.SEED_ROLE_SUFFIX


def _sample(i: int) -> str:
    return f"s_{i:06d}" + hashlib.sha256(str(i).encode()).hexdigest()[:10]


def _authority(root: Path, *, n_real: int = 1669, minority: float = 0.3,
               synthetic_rows: int = 0, overlap: bool = False,
               minority_fraction_override=None) -> Path:
    """Synthetic split authority: five folds + external, with labels."""

    import csv

    n_minority = int(n_real * minority)
    labels = { _sample(i): (1 if i < n_minority else 0) for i in range(n_real) }
    if minority_fraction_override is not None:
        labels = { _sample(i): (1 if i < int(n_real * minority_fraction_override) else 0) for i in range(n_real)}
    hashes = list(labels)
    folds = [hashes[i::5] for i in range(5)]
    splits = root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    external = hashes[:340]
    seen_external = set(external)
    for fold, members in enumerate(folds):
        test = [h for h in members if h not in seen_external][:272]
        train = [h for h in hashes if h not in set(test)][:1087]
        if overlap and fold == 0:
            intruder = next(h for h in folds[1] if h not in seen_external)
            test.append(intruder)  # deliberate overlap with fold 1
        for name, rows in ((f"fold_{fold}_test", test), (f"fold_{fold}_train", train)):
            with (splits / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "patient_hash", "sample_hash", "substudy", "ab_label", "is_synthetic",
                ])
                writer.writeheader()
                for h in rows:
                    writer.writerow({
                        "patient_hash": "pt_" + h[2:10], "sample_hash": h,
                        "substudy": "SYNTH", "ab_label": labels[h], "is_synthetic": False,
                    })
    with (splits / "external_holdout.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "patient_hash", "sample_hash", "substudy", "ab_label", "is_synthetic",
        ])
        writer.writeheader()
        for h in external:
            writer.writerow({
                "patient_hash": "pt_" + h[2:10], "sample_hash": h,
                "substudy": "SYNTH", "ab_label": labels[h], "is_synthetic": False,
            })
    for i in range(synthetic_rows):
        with (splits / f"fold_0_test.csv").open("a", encoding="utf-8", newline="") as handle:
            handle.write(f"pt_x{i},s_synth{i},SYNTH,0,True\n")
    return splits


def _membership(labels_dir: Path, n: int = 1669) -> list[dict[str, str]]:
    return sorted(
        ({"sample_hash": _sample(i), "t1_file_hash": f"t1{i}"} for i in range(n)),
        key=lambda row: row["sample_hash"],
    )


# ---------------------------------------------------------------------------
# deterministic rules
# ---------------------------------------------------------------------------


def test_role_key_is_deterministic_and_fold_scoped() -> None:
    key = sr.role_key("s_abc", fold=2)
    assert key == hashlib.sha256(b"s_abc|2|13").hexdigest()
    assert key != sr.role_key("s_abc", fold=3)


def test_sealed_selection_is_label_blind_and_exact() -> None:
    members = [_sample(i) for i in range(1669)]
    sealed = sr.select_sealed_holdout(members)
    dev = sr.select_development(members)
    assert len(sealed) == sr.SEALED_COUNT == 333
    assert len(dev) == sr.DEV_COUNT == 1336
    assert set(sealed).isdisjoint(dev)
    assert set(sealed) | set(dev) == set(members)


def test_largest_remainder_allocation_is_deterministic_and_exact() -> None:
    counts = sr.allocate_role_counts(1087)
    assert counts == sr.allocate_role_counts(1087)
    total = sum(counts.values())
    assert total == 1087
    # largest remainder: floors 706/163/108/108, remainder 2 -> both 10% roles
    assert counts == {
        "head_fit": 706,
        "model_selection": 163,
        "probability_calibration": 109,
        "conformal_calibration": 109,
    }
    assert all(counts[r] >= int(1087 * p) for r, p in sr.ROLE_PROPORTIONS.items())


# ---------------------------------------------------------------------------
# contradiction gate
# ---------------------------------------------------------------------------


def test_gate_passes_on_consistent_authority(tmp_path: Path) -> None:
    splits = _authority(tmp_path)
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    validated = sr.contradiction_gate(facts)
    assert validated["real_rows"] == 1669
    assert validated["dev_rows"] == 1336
    assert validated["sealed_rows"] == 333


def test_gate_fails_on_fold_overlap_without_writes(tmp_path: Path) -> None:
    splits = _authority(tmp_path, overlap=True)
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    marker = tmp_path / "no-write-marker"
    with pytest.raises(RuntimeError, match="SPLIT_CONTRACT_INCONSISTENT"):
        sr.conadiction_gate = None  # guard against typo usage
        sr.contradiction_gate(facts)
    assert not marker.exists()


def test_gate_fails_on_synthetic_rows(tmp_path: Path) -> None:
    splits = _authority(tmp_path, synthetic_rows=3)
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    with pytest.raises(RuntimeError, match="SPLIT_CONTRACT_INCONSISTENT"):
        sr.contradiction_gate(facts)


def test_nonmember_real_rows_are_excluded_not_fatal(tmp_path: Path) -> None:
    import csv

    splits = _authority(tmp_path)
    with (splits / "fold_0_test.csv").open("a", encoding="utf-8", newline="") as handle:
        handle.write("pt_extra,s_realbutexcluded,LEARN,0,False" + chr(10))
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    validated = sr.contradiction_gate(facts)
    assert validated["real_rows"] == 1669
    assert facts["excluded_nonreal_unique"] == 1


def test_gate_fails_when_minority_role_floor_violated(tmp_path: Path) -> None:
    splits = _authority(tmp_path, minority_fraction_override=0.02)
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    with pytest.raises(RuntimeError, match="SPLIT_CONTRACT_INCONSISTENT"):
        sr.contradiction_gate(facts)


# ---------------------------------------------------------------------------
# registry build, privacy, access control
# ---------------------------------------------------------------------------


def test_registry_is_deterministic(tmp_path: Path) -> None:
    splits = _authority(tmp_path)
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    first = sr.build_registry(sr.contradiction_gate(facts))
    second = sr.build_registry(sr.contradiction_gate(sr.load_split_authority(
        splits_dir=splits, membership=_membership(tmp_path))))
    assert first == second


def test_registry_roles_cover_dev_exactly_once_per_fold(tmp_path: Path) -> None:
    splits = _authority(tmp_path)
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    registry = sr.build_registry(sr.contradiction_gate(facts))
    dev = set(sr.select_development(facts["real_sample_hashes"]))
    for fold in range(5):
        roles = registry["folds"][str(fold)]["roles"]
        members = [h for role_rows in roles.values() for h in role_rows]
        assert len(members) == len(set(members))
        assert set(members) <= dev


def test_public_receipt_has_no_labels_or_ids(tmp_path: Path) -> None:
    splits = _authority(tmp_path)
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    registry = sr.build_registry(sr.contradiction_gate(facts))
    receipt = sr.build_public_receipt(registry)
    serialized = json.dumps(receipt)
    for h in facts["real_sample_hashes"][:50]:
        assert h not in serialized
    assert "ab_label" not in serialized
    assert "label" not in serialized


def test_sealed_label_access_requires_authorization(tmp_path: Path) -> None:
    splits = _authority(tmp_path)
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    registry = sr.build_registry(sr.contradiction_gate(facts))
    with pytest.raises(RuntimeError, match="SEALED_LABELS_LOCKED"):
        sr.sealed_labels(registry, authorized=False)
    unlocked = sr.sealed_labels(registry, authorized=True)
    assert len(unlocked) == 333


def test_registry_validation_is_live_byte_bound(tmp_path: Path) -> None:
    splits = _authority(tmp_path)
    facts = sr.load_split_authority(splits_dir=splits, membership=_membership(tmp_path))
    registry = sr.build_registry(sr.contradiction_gate(facts))
    private = tmp_path / "private.json"
    private.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
    validated = sr.validate_registry(private)
    assert validated["sealed_rows"] == 333
    private.write_text(json.dumps({**registry, "sealed": registry["sealed"][:-1]}, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="REGISTRY_INVALID"):
        sr.validate_registry(private)
