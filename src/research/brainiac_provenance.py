"""BrainIAC R0 provenance, evidence registry, and claim-language boundaries.

This module encodes the honesty rules for the post-hoc BrainIAC retrofit. Three
of them are load-bearing:

* the upstream boundary is *pinned but unresolved*. The commit and published
  input contract are known from the paper and repository tree; weight filenames,
  hashes, sizes and import paths are not, and are never guessed. They stay
  ``UNRESOLVED_PENDING_VENDOR`` until the pinned commit is vendored on H.
* pretraining overlap is ``UNKNOWN`` by default. BrainIAC pretrained on cohorts
  that include ADNI/OASIS, so absence of subject-level overlap with A4/LEARN
  cannot be inferred from silence -- it has to be shown.
* evidence never rounds up. A partial ``citeprobe`` result does not become
  "verified", preprints never merge into the peer-reviewed tier, and a review
  below the target range reports as incomplete rather than complete.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_WEIGHT_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors"}
_OFFICIAL_ENCODER_NAME = "brainiac.ckpt"
# Dropbox folder listing advertised BrainIAC.ckpt as 345.34 MB. A smaller
# file is an in-flight download, not a hashed encoder.
_MIN_OFFICIAL_ENCODER_BYTES = 300_000_000
_WEIGHT_SOURCE_UNRESOLVED = (
    "README documents a Dropbox folder of BrainIAC plus downstream "
    "checkpoints; argparse default is checkpoints/BrainIAC.ckpt. The "
    "pinned GitHub snapshot has no weight file. A hashed copy is "
    "accepted only from vendor/src/checkpoints or "
    "models/pretrained/brainiac. Unofficial Hugging Face conversions "
    "are not official provenance."
)

# --- Upstream boundary constants ----------------------------------------------

UPSTREAM_PINNED_COMMIT = "ba60f45bed832ee0e5683c678caeaeefe2072f0d"
UPSTREAM_REPOSITORY = "https://github.com/AIM-KannLab/BrainIAC"
UPSTREAM_PAPER = "https://www.nature.com/articles/s41593-026-02202-6"

#: Sentinel for any fact that must come from the vendored repository itself.
UNRESOLVED = "UNRESOLVED_PENDING_VENDOR"

#: Fields that may never be populated by inference, only by inspecting the repo.
VENDOR_ONLY_FIELDS = (
    "weight_filename",
    "weight_sha256",
    "weight_size_bytes",
    "encoder_module_path",
    "encoder_class_name",
    "loader_symbol",
)

_MODULE_PATH_FIELDS = ("encoder_module_path", "encoder_class_name", "loader_symbol")
_WEIGHT_FIELDS = ("weight_filename", "weight_sha256", "weight_size_bytes")

# --- Review constants ---------------------------------------------------------

R0_TARGET_SOURCE_RANGE = (24, 30)
R0_MIN_PEER_REVIEWED = 10
R0_REVIEW_WINDOW_START_YEAR = 2025
R0_REVIEW_WINDOW_END_YEAR = 2026

EVIDENCE_TIERS = frozenset({"peer_reviewed", "preprint"})

CITATION_STATUSES = frozenset(
    {"IDENTIFIER_VERIFIED", "INCONCLUSIVE", "UNKNOWN", "FAILED"}
)

REQUIRED_EXTRACTION_FIELDS = (
    "id",
    "title",
    "year",
    "tier",
    "url",
    "endpoint",
    "cohort",
    "modality",
    "preprocessing",
    "split",
    "external_validation",
    "confound_audit",
    "uncertainty",
    "code_weights",
    "citation_status",
)

#: PRISMA-style screening depth. A source recorded from a search summary has been
#: screened, not read; treating the two as equivalent would inflate the review.
SCREENING_STAGES = frozenset(
    {
        "FULL_TEXT_REVIEWED",
        "FULL_TEXT_AVAILABLE_PENDING_REVIEW",
        "TITLE_ABSTRACT_SCREENED",
        "SEARCH_SUMMARY_ONLY",
    }
)
FULL_TEXT_STAGE = "FULL_TEXT_REVIEWED"
FULL_TEXT_AVAILABLE_STAGE = "FULL_TEXT_AVAILABLE_PENDING_REVIEW"

NOVELTY_CEILING = "NOVELTY_CANDIDATE"

#: Phrases that overstate what an internal, MRI-only, single-cohort result shows.
FORBIDDEN_CLAIM_PHRASES = frozenset(
    {
        "alzheimer diagnosis",
        "diagnoses alzheimer",
        "diagnosis of alzheimer",
        "external validation",
        "externally validated",
        "causal region",
        "causally responsible region",
        "diseased region",
        "affected brain region",
        "proves brain damage",
        "clinical triage",
        "regional quantitative biomarker",
    }
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vendor_snapshot_root(project_root: Path | None = None) -> Path:
    root = project_root or _project_root()
    return root / "vendor" / "BrainIAC" / UPSTREAM_PINNED_COMMIT


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _weight_candidates(project_root: Path) -> list[Path]:
    """Official encoder locations only — never a third-party conversion tree."""

    vendor = vendor_snapshot_root(project_root)
    locations = (
        vendor / "src" / "checkpoints",
        project_root / "models" / "pretrained" / "brainiac",
    )
    files: list[Path] = []
    seen: set[Path] = set()
    for location in locations:
        if not location.is_dir():
            continue
        for path in sorted(location.iterdir()):
            resolved = path.resolve()
            if (
                path.is_file()
                and path.suffix.lower() in _WEIGHT_SUFFIXES
                and resolved not in seen
            ):
                seen.add(resolved)
                files.append(path)
    return files


def _select_official_encoder(weight_files: list[Path]) -> Path | None:
    named = [path for path in weight_files if path.name.lower() == _OFFICIAL_ENCODER_NAME]
    if len(named) == 1:
        return named[0]
    if len(weight_files) == 1:
        return weight_files[0]
    return None


def inspect_vendored_tree(project_root: Path | None = None) -> dict[str, Any]:
    """Read encoder/loader identity from the pinned snapshot; never guess weights."""

    root = project_root or _project_root()
    vendor = vendor_snapshot_root(root)
    model_py = vendor / "src" / "model.py"
    load_py = vendor / "src" / "load_brainiac.py"
    license_file = vendor / "LICENSE"
    if not (model_py.is_file() and load_py.is_file() and license_file.is_file()):
        return {"vendored": False}
    model_text = model_py.read_text(encoding="utf-8", errors="replace")
    load_text = load_py.read_text(encoding="utf-8", errors="replace")
    if "class ViTBackboneNet" not in model_text:
        raise RuntimeError("VENDOR_ENCODER_CLASS_MISSING")
    if "def load_brainiac" not in load_text:
        raise RuntimeError("VENDOR_LOADER_MISSING")
    weight_files = _weight_candidates(root)
    inspected: dict[str, Any] = {
        "vendored": True,
        "encoder_module_path": "src/model.py",
        "encoder_class_name": "ViTBackboneNet",
        "loader_symbol": "load_brainiac",
        "weight_filename": UNRESOLVED,
        "weight_sha256": UNRESOLVED,
        "weight_size_bytes": UNRESOLVED,
        "weight_source": _WEIGHT_SOURCE_UNRESOLVED,
        "weight_local_path": UNRESOLVED,
    }
    selected = _select_official_encoder(weight_files)
    if selected is not None and selected.stat().st_size < _MIN_OFFICIAL_ENCODER_BYTES:
        inspected["weight_source"] = (
            "An incomplete BrainIAC.ckpt download is present; filename/hash "
            f"stay {UNRESOLVED} until the file reaches the advertised "
            "345.34 MB encoder size."
        )
        selected = None
    if selected is not None:
        digest, size = _sha256_file(selected)
        inspected["weight_filename"] = selected.name
        inspected["weight_size_bytes"] = size
        inspected["weight_sha256"] = digest
        inspected["weight_local_path"] = str(selected)
        inspected["weight_source"] = (
            "Hashed official encoder file on H. Source of the bytes must still "
            "be the AIM-KannLab Dropbox folder, not an unofficial conversion."
        )
    elif weight_files:
        inspected["weight_source"] = (
            "Multiple checkpoint files are present and none is uniquely "
            f"named BrainIAC.ckpt; filename/hash stay {UNRESOLVED} until the "
            "SimCLR ViT-B encoder file is identified."
        )
    return inspected


def build_upstream_manifest() -> dict[str, Any]:
    """Describe the upstream boundary with unresolved facts left unresolved."""

    inspected = inspect_vendored_tree()
    vendored = bool(inspected.get("vendored"))
    return {
        "schema_version": 1,
        "component": "BrainIAC",
        "family": "method_x_r1_post_hoc_foundation_retrofit",
        "source_url": f"{UPSTREAM_REPOSITORY}/tree/{UPSTREAM_PINNED_COMMIT}",
        "paper_url": UPSTREAM_PAPER,
        "commit": UPSTREAM_PINNED_COMMIT,
        "ref_type": "PINNED_COMMIT",
        # Published in the paper / repository tree -- safe to record.
        "backbone": "simclr_vit_b_3d",
        "input_shape": [96, 96, 96],
        "embedding_dim": 768,
        "input_modality": "single_sequence_mri",
        "expected_preprocessing": [
            "n4_bias_field_correction",
            "resample_1mm_isotropic",
            "rigid_mni_registration",
            "brain_extraction",
            "standardize_96_cubed",
        ],
        "pretraining_scan_count": 32015,
        "evaluation_scan_count": 48965,
        "strict_load_required": True,
        "required_key_coverage": 1.0,
        # Licence boundary.
        "license_boundary": "NON_COMMERCIAL_ACADEMIC_RESEARCH_ONLY",
        "license_url": f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_PINNED_COMMIT}/LICENSE",
        "commercial_use_permitted": False,
        # Encoder identity may be filled only from the vendored tree.
        "weight_filename": inspected.get("weight_filename", UNRESOLVED),
        "weight_sha256": inspected.get("weight_sha256", UNRESOLVED),
        "weight_size_bytes": inspected.get("weight_size_bytes", UNRESOLVED),
        "encoder_module_path": inspected.get("encoder_module_path", UNRESOLVED),
        "encoder_class_name": inspected.get("encoder_class_name", UNRESOLVED),
        "loader_symbol": inspected.get("loader_symbol", UNRESOLVED),
        "weight_source": inspected.get("weight_source", UNRESOLVED),
        "weight_local_path": inspected.get("weight_local_path", UNRESOLVED),
        # Overlap cannot be inferred from silence.
        "pretraining_overlap_status": "UNKNOWN_PRETRAINING_OVERLAP",
        "overlap_evidence": UNRESOLVED,
        "overlap_note": (
            "BrainIAC pretraining cohorts include ADNI/OASIS. Subject-level "
            "disjointness from A4/LEARN must be demonstrated, not assumed."
        ),
        "pet_inputs_permitted": False,
        "raw_identifiers_permitted": False,
        "vendored": vendored,
        "vendor_target_root": (
            f"H:/New project/vendor/BrainIAC/{UPSTREAM_PINNED_COMMIT}"
        ),
    }


def validate_upstream_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on fabricated provenance; report the honest pending state."""

    if manifest.get("commit") != UPSTREAM_PINNED_COMMIT:
        raise RuntimeError("UPSTREAM_COMMIT_NOT_PINNED")
    if manifest.get("ref_type") != "PINNED_COMMIT":
        raise RuntimeError("UPSTREAM_COMMIT_NOT_PINNED")
    if manifest.get("strict_load_required") is not True:
        raise RuntimeError("STRICT_LOAD_REQUIRED")
    if manifest.get("commercial_use_permitted") is not False:
        raise RuntimeError("LICENSE_BOUNDARY_VIOLATION")
    if manifest.get("pet_inputs_permitted") is not False:
        raise RuntimeError("PET_INPUT_FORBIDDEN_IN_METHOD_X")

    vendored = bool(manifest.get("vendored"))
    live = inspect_vendored_tree()

    # Weight and module facts may only be concrete once the repo is on disk,
    # and only if they match the live snapshot — never a guessed hash or path.
    for field in _WEIGHT_FIELDS:
        value = manifest.get(field)
        if not vendored and value != UNRESOLVED:
            raise RuntimeError(f"UNVERIFIED_WEIGHT_PROVENANCE:{field}")
        if vendored and live.get(field, UNRESOLVED) == UNRESOLVED and value != UNRESOLVED:
            raise RuntimeError(f"UNVERIFIED_WEIGHT_PROVENANCE:{field}")
        if (
            vendored
            and live.get(field, UNRESOLVED) != UNRESOLVED
            and value not in (UNRESOLVED, live.get(field))
        ):
            raise RuntimeError(f"UNVERIFIED_WEIGHT_PROVENANCE:{field}")
    for field in _MODULE_PATH_FIELDS:
        value = manifest.get(field)
        if not vendored and value != UNRESOLVED:
            raise RuntimeError(f"UNVERIFIED_MODULE_PATH:{field}")
        if vendored and value not in (UNRESOLVED, live.get(field)):
            raise RuntimeError(f"UNVERIFIED_MODULE_PATH:{field}")

    overlap = manifest.get("pretraining_overlap_status")
    if overlap != "UNKNOWN_PRETRAINING_OVERLAP" and manifest.get(
        "overlap_evidence"
    ) in (None, "", UNRESOLVED):
        raise RuntimeError("OVERLAP_CLAIM_WITHOUT_EVIDENCE")

    return {
        "status": "VENDORED" if vendored else "PENDING_VENDOR",
        "vendored": vendored,
        "commit": UPSTREAM_PINNED_COMMIT,
        "unresolved_fields": [
            field for field in VENDOR_ONLY_FIELDS if manifest.get(field) == UNRESOLVED
        ],
        "pretraining_overlap_status": overlap,
        "license_boundary": manifest.get("license_boundary"),
    }


def validate_evidence_registry(registry: Mapping[str, Any]) -> None:
    """Structural validation of the R0 source table."""

    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("EMPTY_EVIDENCE_REGISTRY")
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise RuntimeError("INVALID_SOURCE_RECORD")
        for field in REQUIRED_EXTRACTION_FIELDS:
            if field not in source:
                raise RuntimeError(f"MISSING_EXTRACTION_FIELD:{field}")
        identifier = str(source["id"])
        if identifier in seen:
            raise RuntimeError(f"DUPLICATE_SOURCE_ID:{identifier}")
        seen.add(identifier)
        if source["tier"] not in EVIDENCE_TIERS:
            raise RuntimeError(f"UNKNOWN_EVIDENCE_TIER:{source['tier']}")
        if source["citation_status"] not in CITATION_STATUSES:
            raise RuntimeError(f"UNKNOWN_CITATION_STATUS:{source['citation_status']}")
        stage = source.get("screening_stage", "SEARCH_SUMMARY_ONLY")
        if stage not in SCREENING_STAGES:
            raise RuntimeError(f"UNKNOWN_SCREENING_STAGE:{stage}")
        year = int(source["year"])
        if not R0_REVIEW_WINDOW_START_YEAR <= year <= R0_REVIEW_WINDOW_END_YEAR:
            raise RuntimeError(f"SOURCE_OUTSIDE_REVIEW_WINDOW:{year}")


def assert_current_r0_write_target(
    target_path: str | Path,
    *,
    current_registry_path: str | Path | None = None,
) -> None:
    """Prevent historical generators from replacing the corrected R0 registry.

    Before v2 exists, the original scaffold may still be created.  Once the
    append-only corrected registry exists, only that exact file is a permissible
    registry write target; v1 remains immutable historical evidence.
    """

    current = Path(current_registry_path) if current_registry_path is not None else (
        PROJECT_ROOT
        / "research"
        / "brainiac_caft_20260808"
        / "evidence_registry_v2.json"
    )
    target = Path(target_path)
    if current.is_file() and target.resolve() != current.resolve():
        raise RuntimeError(
            "BRAINIAC_R0_V1_WRITER_SUPERSEDED:"
            f"current={current}:requested={target}"
        )


def _bound_text_evidence_valid(
    evidence: Any, *, reviewed: bool
) -> bool:
    """Return whether a review/retrieval claim is bound to immutable evidence."""

    if not isinstance(evidence, Mapping):
        return False
    path = str(evidence.get("source_path") or "")
    digest = str(evidence.get("source_sha256") or "").lower()
    try:
        size = int(evidence.get("source_bytes", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or not _is_sha256(digest)
        or size <= 0
    ):
        return False
    if reviewed and (
        evidence.get("review_basis") != "FULL_TEXT_READ_AND_STRUCTURED_EXTRACTION"
        or not str(evidence.get("reviewed_utc") or "")
    ):
        return False
    return True


def evidence_registry_status(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize coverage without rounding partial evidence up to complete."""

    validate_evidence_registry(registry)
    sources = list(registry["sources"])
    peer_reviewed = [s for s in sources if s["tier"] == "peer_reviewed"]
    preprints = [s for s in sources if s["tier"] == "preprint"]
    inconclusive = [s for s in sources if s["citation_status"] != "IDENTIFIER_VERIFIED"]

    low, high = R0_TARGET_SOURCE_RANGE
    total = len(sources)
    meets_range = total >= low
    # A source recorded from a search summary has been screened, not read. Absent
    # an explicit stage, assume the weaker one — never the stronger.
    schema_version = int(registry.get("schema_version", 1))
    stamped_reviewed = [
        s
        for s in sources
        if s.get("screening_stage", "SEARCH_SUMMARY_ONLY") == FULL_TEXT_STAGE
    ]
    if schema_version >= 2:
        full_text = [
            s
            for s in stamped_reviewed
            if _bound_text_evidence_valid(s.get("review_evidence"), reviewed=True)
        ]
        invalid_review_evidence = len(stamped_reviewed) - len(full_text)
    else:
        # Historical v1 registries are still readable, but v2 is required for
        # an auditable completion claim.
        full_text = stamped_reviewed
        invalid_review_evidence = 0
    available_pending = [
        s
        for s in sources
        if s.get("screening_stage", "SEARCH_SUMMARY_ONLY")
        == FULL_TEXT_AVAILABLE_STAGE
        and _bound_text_evidence_valid(s.get("full_text_evidence"), reviewed=False)
    ]
    blockers: list[str] = []
    if not meets_range:
        blockers.append("SOURCE_COUNT_BELOW_TARGET")
    if len(peer_reviewed) < R0_MIN_PEER_REVIEWED:
        blockers.append("PEER_REVIEWED_FLOOR")
    if len(full_text) < low:
        blockers.append("FULL_TEXT_REVIEW_BELOW_TARGET")
    if invalid_review_evidence:
        blockers.append("FULL_TEXT_REVIEW_EVIDENCE_INVALID")

    return {
        "total_count": total,
        "peer_reviewed_count": len(peer_reviewed),
        "preprint_count": len(preprints),
        "inconclusive_count": len(inconclusive),
        "full_text_reviewed_count": len(full_text),
        "full_text_available_pending_review_count": len(available_pending),
        "invalid_full_text_review_evidence_count": invalid_review_evidence,
        "screened_not_read_count": total - len(full_text),
        "fully_verified": not inconclusive,
        "target_range": [low, high],
        "meets_target_range": meets_range,
        "blockers": blockers,
        "status": "COMPLETE" if not blockers else "INCOMPLETE_PENDING_R0_COMPLETION",
        "claim_ceiling": NOVELTY_CEILING,
    }


def assert_claim_language(text: str) -> None:
    """Reject prose that overstates an internal, MRI-only, single-cohort result."""

    normalized = re.sub(r"\s+", " ", str(text)).lower()
    for phrase in sorted(FORBIDDEN_CLAIM_PHRASES):
        if phrase in normalized:
            raise RuntimeError(f"FORBIDDEN_CLAIM_LANGUAGE:{phrase}")
