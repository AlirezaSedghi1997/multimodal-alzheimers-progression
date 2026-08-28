"""Private CAFT split registry for BrainIAC R1 (Plan 004 / B3).

This is the trusted governance builder: it is the only component in the
feature-preparation stack allowed to read labels, and only from the current
split authority CSVs.  Feature extractors receive role-specific views and
never label access.  The sealed internal holdout is label-blind by
construction (separate hash domain) and its labels stay locked until the
evaluation-stage authorization.

Deterministic rules (card Plan 004 Step 3):
    role key  = SHA256(f"{sample_hash}|{fold}|13"), sorted ascending per class
    sealed    = 333 lowest SHA256(f"{sample_hash}|caft_r1_sealed_holdout|13")
    roles     = 65/15/10/10 by largest remainder, tie break by role name
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.research.brainiac_preprocessing_v2 import load_locked_membership

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "splits"
PRIVATE_REGISTRY_PATH = (
    ROOT / "results" / "method_x_registry" / "brainiac_r1"
    / "brainiac_split_registry_private_v1.json"
)
PUBLIC_RECEIPT_PATH = (
    ROOT / "results" / "method_x_registry" / "brainiac_r1"
    / "brainiac_split_registry_public_receipt_v1.json"
)

SEED_ROLE_SUFFIX = 13
FOLD_COUNT = 5
SEALED_COUNT = 333
DEV_COUNT = 1336
SEALED_DOMAIN = "caft_r1_sealed_holdout"
MINORITY_ROLE_FLOOR = 33
ROLE_PROPORTIONS: dict[str, float] = {
    "head_fit": 0.65,
    "model_selection": 0.15,
    "probability_calibration": 0.10,
    "conformal_calibration": 0.10,
}
TEN_PERCENT_ROLES = ("probability_calibration", "conformal_calibration")
REGISTRY_ARTIFACT = "BRAINIAC_CAFT_SPLIT_REGISTRY_PRIVATE_V1"
PUBLIC_ARTIFACT = "BRAINIAC_CAFT_SPLIT_REGISTRY_PUBLIC_RECEIPT_V1"
_BLOCKER_CODE = "BRAINIAC_SPLIT_CONTRACT_INCONSISTENT"


def _inconsistent(detail: str) -> RuntimeError:
    return RuntimeError(f"{_BLOCKER_CODE}:{detail}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def role_key(sample_hash: str, *, fold: int) -> str:
    return _sha256_text(f"{sample_hash}|{fold}|{SEED_ROLE_SUFFIX}")


def sealed_key(sample_hash: str) -> str:
    return _sha256_text(f"{sample_hash}|{SEALED_DOMAIN}|{SEED_ROLE_SUFFIX}")


def select_sealed_holdout(members: Sequence[str]) -> list[str]:
    chosen = sorted(set(members), key=lambda h: (sealed_key(h), h))
    if len(chosen) < SEALED_COUNT:
        raise _inconsistent(f"membership_too_small:{len(chosen)}")
    return chosen[:SEALED_COUNT]


def select_development(members: Sequence[str]) -> list[str]:
    sealed = set(select_sealed_holdout(members))
    return sorted((h for h in set(members) if h not in sealed))


def allocate_role_counts(total: int) -> dict[str, int]:
    """Largest-remainder allocation with deterministic tie break by role name."""

    if total < len(ROLE_PROPORTIONS):
        raise _inconsistent(f"outer_train_too_small:{total}")
    roles = sorted(ROLE_PROPORTIONS)
    floors = {role: int(math.floor(total * ROLE_PROPORTIONS[role])) for role in roles}
    remainder = total - sum(floors.values())
    order = sorted(
        roles,
        key=lambda role: (-(total * ROLE_PROPORTIONS[role] - floors[role]), role),
    )
    for role in order[:remainder]:
        floors[role] += 1
    return floors


def allocate_stratified_roles(
    members_by_class: Mapping[int, Sequence[str]], *, fold: int
) -> dict[str, list[str]]:
    """Sort each class by role key and allocate proportional role counts."""

    roles: dict[str, list[str]] = {role: [] for role in ROLE_PROPORTIONS}
    for class_value in sorted(members_by_class):
        members = sorted(set(members_by_class[class_value]), key=lambda h: (role_key(h, fold=fold), h))
        counts = allocate_role_counts(len(members))
        cursor = 0
        for role in sorted(counts):
            take = counts[role]
            roles[role].extend(members[cursor : cursor + take])
            cursor += take
    return {role: sorted(hashes) for role, hashes in roles.items()}


def _read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_split_authority(
    *,
    splits_dir: Path = SPLITS_DIR,
    membership: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Read the current fold/external authority and join the locked roster."""

    members = (
        list(membership)
        if membership is not None
        else load_locked_membership()
    )
    member_hashes = {str(row["sample_hash"]) for row in members}
    if len(member_hashes) != len(members):
        raise _inconsistent("membership_not_unique")
    splits = Path(splits_dir)
    fold_test: dict[int, set[str]] = {}
    fold_train: dict[int, set[str]] = {}
    labels: dict[str, int] = {}
    synthetic_rows = 0
    duplicate_rows = 0
    excluded_nonreal_rows = 0
    excluded_nonreal_unique: set[str] = set()
    for fold in range(FOLD_COUNT):
        for target, name in ((fold_test, f"fold_{fold}_test"), (fold_train, f"fold_{fold}_train")):
            path = splits / f"{name}.csv"
            if not path.is_file():
                raise _inconsistent(f"authority_missing:{name}")
            current: set[str] = target.setdefault(fold, set())
            for row in _read_split_rows(path):
                sample_hash = str(row.get("sample_hash") or "")
                flag_synthetic = str(row.get("is_synthetic", "")).casefold() in {
                    "true", "1", "yes"
                } or str(row.get("substudy", "")).casefold() in {"parent", "synthetic"}
                if flag_synthetic:
                    synthetic_rows += 1
                    continue
                if sample_hash not in member_hashes:
                    # Real-looking but outside the locked 1669 roster
                    # (e.g. second scans / QC-excluded history): excluded from
                    # the CAFT registry by construction, never fatal.
                    excluded_nonreal_rows += 1
                    excluded_nonreal_unique.add(sample_hash)
                    continue
                if sample_hash in current:
                    duplicate_rows += 1
                    continue
                current.add(sample_hash)
                try:
                    labels[sample_hash] = int(row.get("ab_label"))
                except (TypeError, ValueError) as exc:
                    raise _inconsistent("label_unparseable") from exc
    for row in _read_split_rows(splits / "external_holdout.csv"):
        sample_hash = str(row.get("sample_hash") or "")
        if sample_hash in member_hashes:
            try:
                labels.setdefault(sample_hash, int(row.get("ab_label")))
            except (TypeError, ValueError) as exc:
                raise _inconsistent("label_unparseable") from exc
    return {
        "real_sample_hashes": sorted(member_hashes),
        "labels": labels,
        "fold_test": {fold: fold_test[fold] for fold in range(FOLD_COUNT)},
        "fold_train": {fold: fold_train[fold] for fold in range(FOLD_COUNT)},
        "synthetic_rows": synthetic_rows,
        "duplicate_rows": duplicate_rows,
        "excluded_nonreal_rows": excluded_nonreal_rows,
        "excluded_nonreal_unique": len(excluded_nonreal_unique),
    }


def contradiction_gate(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Machine-check every contract fact; no split is written on failure."""

    real = list(facts["real_sample_hashes"])
    labels: Mapping[str, int] = facts["labels"]
    if len(real) != DEV_COUNT + SEALED_COUNT or len(set(real)) != len(real):
        raise _inconsistent(f"real_rows:{len(real)}")
    if int(facts["synthetic_rows"]) > 0:
        raise _inconsistent(f"synthetic_rows:{facts['synthetic_rows']}")
    if int(facts["duplicate_rows"]) > 0:
        raise _inconsistent(f"duplicate_rows:{facts['duplicate_rows']}")
    missing_labels = [h for h in real if h not in labels]
    if missing_labels:
        raise _inconsistent(f"labels_missing:{len(missing_labels)}")

    fold_test: Mapping[int, set] = facts["fold_test"]
    fold_train: Mapping[int, set] = facts["fold_train"]
    for fold in range(FOLD_COUNT):
        for other in range(fold + 1, FOLD_COUNT):
            overlap = fold_test[fold] & fold_test[other]
            if overlap:
                raise _inconsistent(
                    f"fold_test_overlap:{fold}:{other}:{len(overlap)}"
                )

    sealed = select_sealed_holdout(real)
    dev = select_development(real)
    if len(sealed) != SEALED_COUNT or len(dev) != DEV_COUNT or set(sealed) & set(dev):
        raise _inconsistent("partition_invalid")
    dev_set = set(dev)

    positive = sum(1 for h in real if labels[h] == 1)
    minority_value = 1 if positive <= len(real) - positive else 0
    registry_folds: dict[str, Any] = {}
    for fold in range(FOLD_COUNT):
        test_dev = sorted(fold_test[fold] & dev_set)
        values = {labels[h] for h in test_dev}
        if len(values) < 2:
            raise _inconsistent(f"fold_test_single_class:{fold}")
        outer_train = sorted(fold_train[fold] & dev_set)
        by_class: dict[int, list[str]] = {0: [], 1: []}
        for h in outer_train:
            by_class[labels[h]].append(h)
        for value, rows in by_class.items():
            if not rows:
                raise _inconsistent(f"outer_train_class_empty:{fold}:{value}")
        roles = allocate_stratified_roles(by_class, fold=fold)
        for role, rows in roles.items():
            if {labels[h] for h in rows} != {0, 1}:
                raise _inconsistent(f"role_single_class:{fold}:{role}")
        for role in TEN_PERCENT_ROLES:
            minority_in_role = sum(1 for h in roles[role] if labels[h] == minority_value)
            if minority_in_role < MINORITY_ROLE_FLOOR:
                raise _inconsistent(
                    f"minority_floor:{fold}:{role}:{minority_in_role}"
                )
        registry_folds[str(fold)] = {
            "outer_test": test_dev,
            "roles": roles,
        }
    return {
        "real_rows": len(real),
        "dev_rows": len(dev),
        "sealed_rows": len(sealed),
        "minority_value": minority_value,
        "real_sample_hashes": real,
        "labels": dict(labels),
        "sealed": sealed,
        "dev": dev,
        "folds": registry_folds,
    }


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_registry(validated: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "artifact": REGISTRY_ARTIFACT,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "sealed": sorted(validated["sealed"]),
        "sealed_class_values": {
            h: int(validated["labels"][h]) for h in validated["sealed"]
        },
        "dev_rows": validated["dev_rows"],
        "sealed_rows": validated["sealed_rows"],
        "minority_value": validated["minority_value"],
        "role_proportions": dict(ROLE_PROPORTIONS),
        "role_key_rule": f"sha256(sample_hash|fold|{SEED_ROLE_SUFFIX})_ascending_within_class",
        "sealed_key_rule": f"sha256(sample_hash|{SEALED_DOMAIN}|{SEED_ROLE_SUFFIX})_lowest_{SEALED_COUNT}",
        "folds": validated["folds"],
    }
    body["integrity_sha256"] = _canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    return body


def build_public_receipt(registry: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "artifact": PUBLIC_ARTIFACT,
        "dev_rows": registry["dev_rows"],
        "sealed_rows": registry["sealed_rows"],
        "fold_role_counts": {
            fold: {
                role: len(rows)
                for role, rows in row["roles"].items()
            }
            for fold, row in registry["folds"].items()
        },
        "fold_outer_test_counts": {
            fold: len(row["outer_test"]) for fold, row in registry["folds"].items()
        },
        "sealed_membership_sha256": _canonical_sha(registry["sealed"]),
        "registry_integrity_sha256": registry["integrity_sha256"],
        "sealed_class_values_locked": True,
    }
    body["integrity_sha256"] = _canonical_sha(
        {k: v for k, v in body.items() if k != "integrity_sha256"}
    )
    return body


def sealed_labels(registry: Mapping[str, Any], *, authorized: bool) -> dict[str, int]:
    if not authorized:
        raise RuntimeError("BRAINIAC_SEALED_LABELS_LOCKED")
    return dict(registry["sealed_class_values"])


def validate_registry(path: Path = PRIVATE_REGISTRY_PATH) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    registry = json.loads(raw)
    if not isinstance(registry, dict) or registry.get("artifact") != REGISTRY_ARTIFACT:
        raise RuntimeError("BRAINIAC_SPLIT_REGISTRY_INVALID")
    expected = _canonical_sha(
        {k: v for k, v in registry.items() if k != "integrity_sha256"}
    )
    if registry.get("integrity_sha256") != expected:
        raise RuntimeError("BRAINIAC_SPLIT_REGISTRY_INVALID")
    if len(registry.get("sealed", ())) != SEALED_COUNT:
        raise RuntimeError("BRAINIAC_SPLIT_REGISTRY_INVALID")
    if int(registry.get("dev_rows", -1)) != DEV_COUNT:
        raise RuntimeError("BRAINIAC_SPLIT_REGISTRY_INVALID")
    sealed = set(registry["sealed"])
    for fold, row in registry.get("folds", {}).items():
        for role, rows in row.get("roles", {}).items():
            if sealed & set(rows):
                raise RuntimeError("BRAINIAC_SPLIT_REGISTRY_INVALID")
    return {
        "dev_rows": registry["dev_rows"],
        "sealed_rows": len(sealed),
        "registry_integrity_sha256": registry["integrity_sha256"],
    }


def validate_public_receipt(path: Path = PUBLIC_RECEIPT_PATH) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt.get("artifact") != PUBLIC_ARTIFACT:
        raise RuntimeError("BRAINIAC_SPLIT_PUBLIC_RECEIPT_INVALID")
    expected = _canonical_sha(
        {k: v for k, v in receipt.items() if k != "integrity_sha256"}
    )
    if receipt.get("integrity_sha256") != expected:
        raise RuntimeError("BRAINIAC_SPLIT_PUBLIC_RECEIPT_INVALID")
    if receipt.get("sealed_rows") != SEALED_COUNT or receipt.get("dev_rows") != DEV_COUNT:
        raise RuntimeError("BRAINIAC_SPLIT_PUBLIC_RECEIPT_INVALID")
    return dict(receipt)
