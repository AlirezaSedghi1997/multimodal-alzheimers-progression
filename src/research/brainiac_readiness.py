"""Byte-bound BrainIAC R0 audit and action-level R1 readiness decisions.

The module deliberately separates *preparation* from training/evaluation.  It
does not perform model, image, label, or outer-test I/O.  Every unknown or
ambiguous authority is represented as a blocker rather than rounded up.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from src.research.brainiac_provenance import (
    FULL_TEXT_STAGE,
    NOVELTY_CEILING,
    R0_REVIEW_WINDOW_END_YEAR,
    R0_REVIEW_WINDOW_START_YEAR,
    validate_evidence_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_POLICY = PROJECT_ROOT / "governance" / "brainiac_r1_execution_policy_v1.json"
DEFAULT_SCOPE_AMENDMENT = (
    PROJECT_ROOT
    / "governance"
    / "brainiac_r1_preterminal_preparation_scope_amendment_v1.json"
)
DEFAULT_PREDECESSOR_SCOPE = (
    PROJECT_ROOT / "governance" / "method_x_current_completion_scope_v1.json"
)
DEFAULT_R0_REGISTRY = (
    PROJECT_ROOT / "research" / "brainiac_caft_20260808" / "evidence_registry_v3.json"
)
DEFAULT_SCHEMA3_READINESS = (
    PROJECT_ROOT
    / "research"
    / "brainiac_caft_20260808"
    / "r1_readiness_status_20260822.json"
)
DEFAULT_BATTERY_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "final_results"
    / "method_x_model_battery_v3_depth_and_diversity_20260729"
    / "battery_manifest.json"
)
DEFAULT_RUNTIME_REPORT = (
    PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "brainiac_official_runtime_lock.json"
)
DEFAULT_STRICT_LOAD_REPORT = (
    PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "brainiac_official_strict_load_receipt.json"
)
DEFAULT_FORWARD_REPORT = (
    PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "brainiac_official_forward_receipt.json"
)
DEFAULT_UPSTREAM_MANIFEST = (
    PROJECT_ROOT
    / "research"
    / "brainiac_caft_20260808"
    / "upstream_boundary_manifest.json"
)
DEFAULT_OPTIONAL_ARTIFACTS = {
    "preprocessing_v2": PROJECT_ROOT
    / "data"
    / "derived"
    / "method_x_brainiac"
    / "r1_primary_v2"
    / "aggregate_manifest.json",
    "brainiac_embeddings": PROJECT_ROOT
    / "data"
    / "derived"
    / "method_x_brainiac"
    / "r1_primary_v2"
    / "brainiac_embeddings_manifest.json",
    "split_registry": PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "split_registry_v1.json",
    "resnet34_representation_contract": PROJECT_ROOT
    / "governance"
    / "brainiac_resnet34_representation_contract_v1.json",
    "resnet34_inner_features": PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "resnet34_inner_features_manifest.json",
    "brainiac_extractor_contract": PROJECT_ROOT
    / "governance"
    / "brainiac_embedding_extractor_contract_v1.json",
    "caft_trainer_contract": PROJECT_ROOT
    / "governance"
    / "brainiac_caft_training_contract_v1.json",
    "outer_head_calibration_lock": PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "outer_head_calibration_lock.json",
    "promotion_evidence": PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "promotion_evidence.json",
    "external_authorization": PROJECT_ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "external_evaluation_authorization.json",
}

PINNED_POLICY_SHA256 = "9ef85415fcc1f7608a060ff2c6072841adb8b9e91be97764b70446dc42d46dfc"
PINNED_SCOPE_AMENDMENT_SHA256 = "6b313303ff16c16c160838c4022801a5a2fd3de0aed2084ac7289cf8ecfae5fc"

EFFECTIVE_FIVE = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)
EXCLUDED_BACKBONES = ("resnet50_3d", "efficientnet3d_b0")

ACTIONS = (
    "PREPARE_RAW_T1",
    "EXTRACT_FROZEN_BRAINIAC_EMBEDDINGS",
    "EXTRACT_RESNET34_INNER_FEATURES",
    "CAFT_TRAINING",
    "OUTER_FEATURE_SNAPSHOT",
    "PROMOTION",
    "EXTERNAL_EVALUATION",
)
PREPARATION_ACTIONS = ACTIONS[:3]
BATTERY_TERMINAL_ACTIONS = ACTIONS[3:6]

CLAIM_CEILING = "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"
UNKNOWN_OVERLAP = "UNKNOWN_PRETRAINING_OVERLAP"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"NONFINITE:{value}")


def _load_json_strict(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"NONFINITE_OR_INVALID_JSON:{path.name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"MALFORMED_JSON_ROOT:{path.name}")
    return payload, raw


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return True
    attributes = int(getattr(info, "st_file_attributes", 0))
    reparse_flag = int(getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return path.is_symlink() or bool(attributes & reparse_flag)


def _beneath(path: Path, root: Path, *, must_exist: bool = True) -> Path:
    base = root.resolve(strict=True)
    lexical = path if path.is_absolute() else base / path
    lexical = Path(os.path.abspath(str(lexical)))
    try:
        lexical.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"AUTHORITY_OUTSIDE_PROJECT_ROOT:{path}") from exc

    # Resolve only after checking the lexical chain.  Otherwise an in-root
    # junction/symlink disappears from ``Path.resolve`` and is never inspected.
    current = lexical
    while current != base:
        if os.path.lexists(current) and _is_reparse(current):
            raise RuntimeError(f"REPARSE_AUTHORITY_FORBIDDEN:{path}")
        current = current.parent

    candidate = lexical.resolve(strict=must_exist)
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"AUTHORITY_OUTSIDE_PROJECT_ROOT:{path}") from exc
    current = candidate
    while current != base:
        if _is_reparse(current):
            raise RuntimeError(f"REPARSE_AUTHORITY_FORBIDDEN:{path}")
        current = current.parent
    return candidate


def bind_file(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Bind an existing regular file by resolved path, exact bytes and SHA-256."""

    resolved = _beneath(Path(path), Path(project_root), must_exist=True)
    if not resolved.is_file() or _is_reparse(resolved):
        raise RuntimeError(f"AUTHORITY_FILE_INVALID:{path}")
    raw = resolved.read_bytes()
    return {"path": str(resolved), "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _normalize_identifier(kind: str, value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if kind == "doi":
        text = text.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        if text.startswith("arxiv:"):
            return "arxiv:" + text.split(":", 1)[1].strip()
        return "doi:" + text
    if kind in {"pubmed", "pmid"}:
        return "pmid:" + text.removeprefix("pmid:")
    if kind == "pmc":
        return "pmc:pmc" + text.removeprefix("pmc")
    if kind == "arxiv":
        return "arxiv:" + text.removeprefix("arxiv:")
    return None


def _source_identifiers(source: Mapping[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for key in ("doi", "pubmed", "pmid", "pmc", "arxiv"):
        normalized = _normalize_identifier(key, source.get(key))
        if normalized:
            identifiers.add(normalized)
    url = str(source.get("url") or "").strip().casefold().rstrip("/")
    url_identifier: str | None = None
    if url:
        parsed = urlparse(url)
        host = parsed.netloc.removeprefix("www.")
        path = unquote(parsed.path).strip("/")
        if host in {"doi.org", "dx.doi.org"} and path:
            url_identifier = _normalize_identifier("doi", path)
        elif host == "pubmed.ncbi.nlm.nih.gov" and path:
            pmid = path.split("/", 1)[0]
            if pmid.isdigit():
                url_identifier = _normalize_identifier("pmid", pmid)
        elif host == "pmc.ncbi.nlm.nih.gov" and path.casefold().startswith("articles/"):
            pmc = path.split("/", 2)[1]
            url_identifier = _normalize_identifier("pmc", pmc)
        elif "arxiv.org" in host and path:
            tail = path.removeprefix("abs/").removeprefix("html/")
            tail = re.sub(r"v\d+$", "", tail)
            if tail:
                url_identifier = _normalize_identifier("arxiv", tail)
    if url_identifier:
        identifiers.add(url_identifier)
    elif not identifiers and url:
        identifiers.add("url:" + url)
    return identifiers


def _validate_cross_identifier_consistency(
    source: Mapping[str, Any], *, project_root: Path
) -> None:
    """Require byte-bound evidence before treating distinct identifiers as aliases."""

    explicit = _source_identifiers(source)
    if len(explicit) <= 1:
        return

    evidence = source.get("identifier_crosswalk_evidence")
    if isinstance(evidence, Mapping):
        relative = Path(str(evidence.get("source_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"R0_CROSS_IDENTIFIER_EVIDENCE_PATH_INVALID:{source.get('id')}"
            )
        binding = bind_file(project_root / relative, project_root=project_root)
        declared = {
            str(value).casefold()
            for value in (evidence.get("canonical_identifiers") or ())
        }
        evidence_text = Path(binding["path"]).read_text(
            encoding="utf-8", errors="strict"
        ).casefold()
        identifier_values = [token.split(":", 1)[1] for token in explicit]
        if (
            declared != explicit
            or int(evidence.get("source_bytes", -1)) != binding["bytes"]
            or str(evidence.get("source_sha256") or "").casefold()
            != binding["sha256"]
            or str(source.get("id") or "").casefold() not in evidence_text
            or not all(value in evidence_text for value in identifier_values)
        ):
            raise RuntimeError(
                f"R0_CROSS_IDENTIFIER_EVIDENCE_MISMATCH:{source.get('id')}"
            )
        return

    review = source.get("review_evidence")
    if isinstance(review, Mapping):
        relative = Path(str(review.get("source_path") or ""))
        if not relative.is_absolute() and ".." not in relative.parts:
            binding = bind_file(project_root / relative, project_root=project_root)
            raw = Path(binding["path"]).read_text(encoding="utf-8", errors="ignore").casefold()
            values = [token.split(":", 1)[1].casefold() for token in explicit]
            if all(value in raw for value in values):
                return
    raise RuntimeError(f"R0_CROSS_IDENTIFIER_EVIDENCE_MISSING:{source.get('id')}")


def _validate_review_evidence(
    source: Mapping[str, Any], *, project_root: Path
) -> bool:
    if source.get("screening_stage") != FULL_TEXT_STAGE:
        return False
    evidence = source.get("review_evidence")
    if not isinstance(evidence, Mapping):
        raise RuntimeError(f"R0_REVIEW_EVIDENCE_MISSING:{source.get('id')}")
    if evidence.get("review_basis") != "FULL_TEXT_READ_AND_STRUCTURED_EXTRACTION":
        raise RuntimeError(f"R0_REVIEW_BASIS_INVALID:{source.get('id')}")
    relative = Path(str(evidence.get("source_path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"R0_REVIEW_EVIDENCE_PATH_INVALID:{source.get('id')}")
    path = _beneath(project_root / relative, project_root, must_exist=True)
    binding = bind_file(path, project_root=project_root)
    try:
        expected_bytes = int(evidence.get("source_bytes", -1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"R0_REVIEW_EVIDENCE_SIZE_INVALID:{source.get('id')}") from exc
    if expected_bytes != binding["bytes"]:
        raise RuntimeError(f"R0_REVIEW_EVIDENCE_SIZE_MISMATCH:{source.get('id')}")
    if str(evidence.get("source_sha256") or "").casefold() != binding["sha256"]:
        raise RuntimeError(f"R0_REVIEW_EVIDENCE_HASH_MISMATCH:{source.get('id')}")
    if not str(evidence.get("reviewed_utc") or ""):
        raise RuntimeError(f"R0_REVIEW_TIMESTAMP_MISSING:{source.get('id')}")
    return True


def audit_r0_registry(
    registry_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Recompute R0 counts and verify identifiers plus full-text evidence bytes."""

    root = Path(project_root)
    path = _beneath(Path(registry_path), root, must_exist=True)
    registry, raw = _load_json_strict(path)
    validate_evidence_registry(registry)
    sources = registry["sources"]
    if len(sources) != 29:
        raise RuntimeError(f"R0_SOURCE_COUNT_DRIFT:{len(sources)}")

    seen: dict[str, str] = {}
    full_text_reviewed = 0
    peer_reviewed = 0
    for source in sources:
        source_id = str(source.get("id") or "")
        year = int(source.get("year"))
        if not R0_REVIEW_WINDOW_START_YEAR <= year <= R0_REVIEW_WINDOW_END_YEAR:
            raise RuntimeError(f"R0_YEAR_WINDOW_DRIFT:{source_id}:{year}")
        identifiers = _source_identifiers(source)
        if not identifiers:
            raise RuntimeError(f"R0_CANONICAL_IDENTIFIER_MISSING:{source_id}")
        for identifier in sorted(identifiers):
            previous = seen.get(identifier)
            if previous is not None and previous != source_id:
                raise RuntimeError(
                    f"R0_CANONICAL_IDENTIFIER_DUPLICATE:{identifier}:{previous}:{source_id}"
                )
            seen[identifier] = source_id
        _validate_cross_identifier_consistency(source, project_root=root)
        if source.get("tier") == "peer_reviewed":
            peer_reviewed += 1
        if _validate_review_evidence(source, project_root=root):
            full_text_reviewed += 1

    if peer_reviewed != 13:
        raise RuntimeError(f"R0_PEER_REVIEWED_COUNT_DRIFT:{peer_reviewed}")
    if full_text_reviewed != 28:
        raise RuntimeError(f"R0_FULL_TEXT_COUNT_DRIFT:{full_text_reviewed}")
    corroboration = str(registry.get("citation_corroboration") or "UNKNOWN")
    partial = corroboration != "CONCLUSIVE" or any(
        source.get("citation_status") != "IDENTIFIER_VERIFIED" for source in sources
    )
    return {
        "r0_status": "R0_COMPLETE",
        "citation_corroboration": (
            "CITATION_CORROBORATION_PARTIAL"
            if partial
            else "CITATION_CORROBORATION_COMPLETE"
        ),
        "novelty_status": NOVELTY_CEILING,
        "total_in_window": len(sources),
        "peer_reviewed": peer_reviewed,
        "full_text_reviewed": full_text_reviewed,
        "non_full_text": len(sources) - full_text_reviewed,
        "fully_verified": not partial,
        "registry_binding": {
            "path": str(path),
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
        },
    }


def _require_sha_pin(value: str, *, code: str) -> str:
    normalized = str(value or "").casefold()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise RuntimeError(f"{code}_SHA256_PIN_INVALID")
    return normalized


def validate_policy_and_scope(
    *,
    policy_path: str | Path = DEFAULT_POLICY,
    scope_amendment_path: str | Path = DEFAULT_SCOPE_AMENDMENT,
    expected_policy_sha256: str = PINNED_POLICY_SHA256,
    expected_scope_amendment_sha256: str = PINNED_SCOPE_AMENDMENT_SHA256,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate the immutable policy and append-only scope transition."""

    root = Path(project_root)
    policy_binding = bind_file(policy_path, project_root=root)
    amendment_binding = bind_file(scope_amendment_path, project_root=root)
    if policy_binding["sha256"] != _require_sha_pin(
        expected_policy_sha256, code="BRAINIAC_POLICY"
    ):
        raise RuntimeError("BRAINIAC_POLICY_SHA256_MISMATCH")
    if amendment_binding["sha256"] != _require_sha_pin(
        expected_scope_amendment_sha256, code="BRAINIAC_SCOPE_AMENDMENT"
    ):
        raise RuntimeError("BRAINIAC_SCOPE_AMENDMENT_SHA256_MISMATCH")
    policy, _ = _load_json_strict(Path(policy_binding["path"]))
    amendment, _ = _load_json_strict(Path(amendment_binding["path"]))

    action_rows = policy.get("actions")
    if not isinstance(action_rows, Mapping) or tuple(action_rows) != ACTIONS:
        raise RuntimeError("BRAINIAC_POLICY_ACTION_SET_INVALID")
    if tuple(policy.get("effective_five_model_battery") or ()) != EFFECTIVE_FIVE:
        raise RuntimeError("BRAINIAC_POLICY_EFFECTIVE_BATTERY_INVALID")
    if tuple(policy.get("protocol_exclusions") or ()) != EXCLUDED_BACKBONES:
        raise RuntimeError("BRAINIAC_POLICY_EXCLUSIONS_INVALID")
    supervision = policy.get("supervision")
    if not isinstance(supervision, Mapping) or any(
        (
            supervision.get("human_approval_required") is not False,
            supervision.get("manual_override_allowed") is not False,
            supervision.get("unknown_is_pass") is not False,
        )
    ):
        raise RuntimeError("BRAINIAC_POLICY_SUPERVISION_INVALID")
    if policy.get("lora_unlocked") is not False:
        raise RuntimeError("BRAINIAC_POLICY_LORA_INVALID")

    predecessor = amendment.get("predecessor_scope")
    if not isinstance(predecessor, Mapping):
        raise RuntimeError("BRAINIAC_SCOPE_PREDECESSOR_BINDING_MISSING")
    relative = Path(str(predecessor.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("BRAINIAC_SCOPE_PREDECESSOR_PATH_INVALID")
    predecessor_binding = bind_file(root / relative, project_root=root)
    if (
        int(predecessor.get("bytes", -1)) != predecessor_binding["bytes"]
        or str(predecessor.get("sha256") or "").casefold()
        != predecessor_binding["sha256"]
    ):
        raise RuntimeError("BRAINIAC_SCOPE_PREDECESSOR_BINDING_MISMATCH")
    preserved = amendment.get("preserved")
    changed = amendment.get("changed_rule_only")
    if (
        not isinstance(preserved, Mapping)
        or preserved.get("brainiac_role") != "POST_HOC_SIXTH_MODEL_FAMILY"
        or preserved.get("official_battery_member") is not False
        or preserved.get("training_before_terminal_five_cnn_battery")
        != "HARD_DISABLED"
        or not isinstance(changed, Mapping)
        or tuple(changed.get("preterminal_actions") or ()) != PREPARATION_ACTIONS
    ):
        raise RuntimeError("BRAINIAC_SCOPE_AMENDMENT_SEMANTICS_INVALID")
    return {
        "policy": policy_binding,
        "scope_amendment": amendment_binding,
        "predecessor_scope": predecessor_binding,
    }


def _integrity_matches_compact(payload: Mapping[str, Any]) -> bool:
    expected = str(payload.get("integrity_sha256") or "").casefold()
    if len(expected) != 64:
        return False
    body = dict(payload)
    body.pop("integrity_sha256", None)
    raw = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(raw) == expected


def _validate_official_runtime_evidence(
    *,
    runtime: Mapping[str, Any],
    strict: Mapping[str, Any],
    forward: Mapping[str, Any],
    upstream: Mapping[str, Any],
    project_root: Path,
) -> dict[str, bool]:
    if not all(_integrity_matches_compact(row) for row in (runtime, strict, forward)):
        raise RuntimeError("OFFICIAL_RUNTIME_EVIDENCE_INTEGRITY_MISMATCH")
    from src.research.brainiac_adapter import stable_source_tree_sha256
    from src.research.brainiac_model_loader import (
        DEFAULT_CKPT,
        PINNED_CHECKPOINT_SHA256,
        PINNED_LICENSE_PATH,
        PINNED_LICENSE_SHA256,
        PINNED_MODEL_PATH,
        PINNED_VENDOR_COMMIT,
        PINNED_VENDOR_MODEL_SHA256,
        PINNED_VENDOR_ROOT,
    )

    def bound_runtime_file(path_value: Any, digest_value: Any) -> bool:
        try:
            path = _beneath(Path(str(path_value or "")), project_root, must_exist=True)
            return (
                path.is_file()
                and not _is_reparse(path)
                and _sha256_file(path) == str(digest_value or "").casefold()
            )
        except (OSError, RuntimeError, ValueError):
            return False

    runtime_identity = runtime.get("runtime_identity")
    pip_freeze = runtime.get("pip_freeze")
    runtime_files_ok = (
        bound_runtime_file(runtime.get("interpreter_path"), runtime.get("interpreter_sha256"))
        and bound_runtime_file(
            runtime.get("torch_overlay_path"), runtime.get("torch_overlay_sha256")
        )
        and isinstance(pip_freeze, list)
        and _sha256_bytes(
            json.dumps(pip_freeze, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        == runtime.get("pip_freeze_sha256")
    )
    runtime_ok = (
        runtime.get("status") == "PASS"
        and runtime.get("cuda_available") is True
        and isinstance(runtime_identity, Mapping)
        and runtime_identity.get("status") == "PASS"
        and runtime_identity.get("official_vendor_runtime") is True
        and runtime_identity.get("class_name") == "ViTBackboneNet"
        and runtime_identity.get("loader_symbol") == "load_brainiac"
        and runtime_identity.get("module_path") == "src/model.py"
        and runtime_files_ok
    )
    checkpoint_path = Path(str(strict.get("checkpoint_path") or ""))
    loader_path = PINNED_VENDOR_ROOT / "src" / "load_brainiac.py"
    strict_assets_ok = (
        checkpoint_path.resolve(strict=False) == DEFAULT_CKPT.resolve(strict=False)
        and bound_runtime_file(checkpoint_path, PINNED_CHECKPOINT_SHA256)
        and int(strict.get("checkpoint_bytes", -1)) == DEFAULT_CKPT.stat().st_size
        and strict.get("checkpoint_sha256") == PINNED_CHECKPOINT_SHA256
        and bound_runtime_file(PINNED_MODEL_PATH, PINNED_VENDOR_MODEL_SHA256)
        and strict.get("model_file_sha256") == PINNED_VENDOR_MODEL_SHA256
        and bound_runtime_file(loader_path, strict.get("loader_file_sha256"))
        and bound_runtime_file(PINNED_LICENSE_PATH, PINNED_LICENSE_SHA256)
        and strict.get("license_sha256") == PINNED_LICENSE_SHA256
        and upstream.get("commit") == PINNED_VENDOR_COMMIT
        and Path(str(upstream.get("vendor_target_root") or "")).resolve(strict=False)
        == PINNED_VENDOR_ROOT.resolve(strict=False)
        and upstream.get("weight_sha256") == PINNED_CHECKPOINT_SHA256
        and Path(str(upstream.get("weight_local_path") or "")).resolve(strict=False)
        == DEFAULT_CKPT.resolve(strict=False)
        and int(upstream.get("weight_size_bytes", -1)) == DEFAULT_CKPT.stat().st_size
        and stable_source_tree_sha256(PINNED_VENDOR_ROOT)
        == strict.get("source_tree_sha256")
    )
    strict_ok = (
        strict.get("status") == "PASS"
        and strict.get("receipt_type") == "ACTUAL_STRICT_LOAD"
        and strict.get("strict") is True
        and int(strict.get("matched_keys", 0)) > 0
        and int(strict.get("missing_keys", -1)) == 0
        and int(strict.get("unexpected_keys", -1)) == 0
        and int(strict.get("matched_keys", 0)) == int(strict.get("shape_key_count", -1))
        and strict_assets_ok
        and Path(str(strict.get("environment_lock_path") or "")).resolve(strict=False)
        == DEFAULT_RUNTIME_REPORT.resolve(strict=False)
    )
    forward_ok = (
        forward.get("status") == "PASS"
        and forward.get("receipt_type") == "ACTUAL_FORWARD"
        and forward.get("device") == "cuda"
        and forward.get("finite") is True
        and forward.get("input_shape") == [1, 1, 96, 96, 96]
        and forward.get("output_shape") == [1, 768]
        and int(forward.get("study_images_read", -1)) == 0
        and int(forward.get("labels_read", -1)) == 0
        and forward.get("outer_test_touched") is False
        and int(forward.get("peak_vram_bytes", 1 << 63)) < int(7.5 * (1024**3))
        and forward.get("checkpoint_sha256") == PINNED_CHECKPOINT_SHA256
        and forward.get("source_tree_sha256") == strict.get("source_tree_sha256")
        and forward.get("module_identity_sha256") == strict.get("module_identity_sha256")
        and Path(str(forward.get("strict_load_receipt_path") or "")).resolve(strict=False)
        == DEFAULT_STRICT_LOAD_REPORT.resolve(strict=False)
    )
    return {
        "official_runtime": runtime_ok,
        "strict_load": strict_ok,
        "official_forward": forward_ok,
    }


_WRITER_MARKERS = (
    "scripts\\train_3d_mri.py",
    "scripts/train_3d_mri.py",
    "scripts\\run_method_x_claim_benchmark.py",
    "scripts/run_method_x_claim_benchmark.py",
    "scripts\\research\\preprocess_brainiac_raw_t1.py",
    "scripts/research/preprocess_brainiac_raw_t1.py",
    "scripts\\research\\extract_brainiac_embeddings.py",
    "scripts/research/extract_brainiac_embeddings.py",
    "scripts\\research\\run_brainiac_caft_pipeline.py",
    "scripts/research/run_brainiac_caft_pipeline.py",
)


def _default_process_provider() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,Name,CommandLine,CreationDate | "
        "ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError("PROCESS_INSPECTION_FAILED")
    raw = completed.stdout.strip()
    if not raw:
        return []
    try:
        rows = json.loads(raw, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("PROCESS_INSPECTION_INVALID") from exc
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError("PROCESS_INSPECTION_INVALID")
    return [dict(row) for row in rows]


def inspect_writer_state(
    battery: Mapping[str, Any],
    *,
    process_provider: Any | None = None,
) -> dict[str, Any]:
    """Inspect live processes and reconcile manifest-recorded owners.

    The provider is injectable only for deterministic tests.  Production callers
    omit it and the process table is obtained directly at the decision boundary.
    """

    provider = process_provider or _default_process_provider
    try:
        supplied = provider() if callable(provider) else provider
        if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
            raise RuntimeError("PROCESS_INSPECTION_INVALID")
        processes: dict[int, dict[str, Any]] = {}
        for item in supplied:
            if not isinstance(item, Mapping):
                raise RuntimeError("PROCESS_INSPECTION_INVALID")
            pid = int(item.get("ProcessId", item.get("pid", 0)))
            if pid <= 0:
                continue
            name = str(item.get("Name", item.get("name", "")))
            command_value = item.get("CommandLine", item.get("command_line"))
            if (
                pid != os.getpid()
                and name.casefold().startswith("python")
                and (command_value is None or not str(command_value).strip())
            ):
                raise RuntimeError("PROCESS_COMMAND_LINE_UNKNOWN")
            processes[pid] = {
                "pid": pid,
                "name": name,
                "command_line": str(command_value or ""),
                "creation_time": str(
                    item.get("CreationDate", item.get("creation_time", "")) or ""
                ),
            }
    except (OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError):
        return {
            "inspection_status": "UNKNOWN",
            "active_writers": [],
            "recorded_owners": [],
        }

    active: list[dict[str, Any]] = []
    for process in processes.values():
        lowered = process["command_line"].casefold()
        executable = process["name"].casefold()
        if (
            process["pid"] != os.getpid()
            and executable.startswith("python")
            and any(marker in lowered for marker in _WRITER_MARKERS)
        ):
            active.append(
                {
                    "pid": process["pid"],
                    "name": process["name"],
                    "creation_time": process["creation_time"],
                }
            )

    owners: list[dict[str, Any]] = []
    rows = battery.get("models")
    if not isinstance(rows, list):
        return {
            "inspection_status": "UNKNOWN",
            "active_writers": active,
            "recorded_owners": [],
        }
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        training = row.get("training")
        if not isinstance(training, Mapping):
            continue
        status = str(training.get("status") or row.get("status") or "").upper()
        if "RUNNING" not in status:
            continue
        try:
            pid = int(training.get("pid"))
        except (TypeError, ValueError):
            owners.append(
                {"pid": None, "liveness": "UNKNOWN", "creation_time_matches": None}
            )
            continue
        process = processes.get(pid)
        if process is None:
            owners.append({"pid": pid, "liveness": "DEAD", "creation_time_matches": True})
            continue
        expected_creation = str(training.get("process_creation_time") or "")
        actual_creation = process["creation_time"]
        owners.append(
            {
                "pid": pid,
                "liveness": "LIVE",
                "creation_time_matches": bool(expected_creation)
                and expected_creation == actual_creation,
            }
        )
    return {
        "inspection_status": "PASS",
        "active_writers": sorted(active, key=lambda row: row["pid"]),
        "recorded_owners": owners,
    }


def _load_bound_json(
    role: str,
    path: Path,
    *,
    project_root: Path,
    bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    binding = bind_file(path, project_root=project_root)
    payload, _ = _load_json_strict(Path(binding["path"]))
    bindings[role] = binding
    return payload


_FUTURE_AUTHORITY_ROLES = frozenset(DEFAULT_OPTIONAL_ARTIFACTS)


def validate_future_authority(
    role: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed until the owning B2-B5 card supplies a real byte verifier.

    Binding a JSON file proves only which bytes were read; it does not prove the
    images, receipts, features, predictions, DUA, or exactly-once events that
    the JSON describes.  Each later card must replace its role with a dedicated
    validator and RED tests before the role can ever become ready.
    """

    if role not in _FUTURE_AUTHORITY_ROLES:
        raise RuntimeError(f"UNKNOWN_OPTIONAL_AUTHORITY_ROLE:{role}")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"OPTIONAL_AUTHORITY_SCHEMA_INVALID:{role}")
    return {
        "role": role,
        "ready": False,
        "status": "BLOCKED_UNTIL_ROLE_SPECIFIC_VALIDATOR",
        "blockers": [f"ROLE_SPECIFIC_VALIDATOR_NOT_IMPLEMENTED:{role}"],
    }


def build_live_readiness(
    *,
    policy_path: str | Path = DEFAULT_POLICY,
    scope_amendment_path: str | Path = DEFAULT_SCOPE_AMENDMENT,
    registry_path: str | Path = DEFAULT_R0_REGISTRY,
    schema3_readiness_path: str | Path = DEFAULT_SCHEMA3_READINESS,
    battery_manifest_path: str | Path = DEFAULT_BATTERY_MANIFEST,
    runtime_report_path: str | Path = DEFAULT_RUNTIME_REPORT,
    strict_load_report_path: str | Path = DEFAULT_STRICT_LOAD_REPORT,
    forward_report_path: str | Path = DEFAULT_FORWARD_REPORT,
    upstream_manifest_path: str | Path = DEFAULT_UPSTREAM_MANIFEST,
    optional_artifact_paths: Mapping[str, str | Path] | None = None,
    process_provider: Any | None = None,
    expected_policy_sha256: str = PINNED_POLICY_SHA256,
    expected_scope_amendment_sha256: str = PINNED_SCOPE_AMENDMENT_SHA256,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Recompute schema-4 readiness from byte-bound authorities and live writers."""

    root = Path(project_root)
    governance = validate_policy_and_scope(
        policy_path=policy_path,
        scope_amendment_path=scope_amendment_path,
        expected_policy_sha256=expected_policy_sha256,
        expected_scope_amendment_sha256=expected_scope_amendment_sha256,
        project_root=root,
    )
    bindings: dict[str, dict[str, Any]] = {
        "policy": governance["policy"],
        "scope_amendment": governance["scope_amendment"],
        "predecessor_scope": governance["predecessor_scope"],
    }
    registry_binding = bind_file(registry_path, project_root=root)
    bindings["r0_registry"] = registry_binding
    r0 = audit_r0_registry(registry_path, project_root=root)
    battery = _load_bound_json(
        "battery_manifest", Path(battery_manifest_path), project_root=root, bindings=bindings
    )
    runtime = _load_bound_json(
        "official_runtime", Path(runtime_report_path), project_root=root, bindings=bindings
    )
    strict = _load_bound_json(
        "official_strict_load", Path(strict_load_report_path), project_root=root, bindings=bindings
    )
    forward = _load_bound_json(
        "official_forward", Path(forward_report_path), project_root=root, bindings=bindings
    )
    upstream = _load_bound_json(
        "upstream_manifest", Path(upstream_manifest_path), project_root=root, bindings=bindings
    )
    schema3_binding = bind_file(schema3_readiness_path, project_root=root)
    schema3, _ = _load_json_strict(Path(schema3_binding["path"]))
    if schema3.get("schema_version") != 3:
        raise RuntimeError("READINESS_PREDECESSOR_SCHEMA_INVALID")
    bindings["schema3_readiness_predecessor"] = schema3_binding

    if str(upstream.get("pretraining_overlap_status")) != UNKNOWN_OVERLAP:
        overlap = str(upstream.get("pretraining_overlap_status") or "UNKNOWN")
    else:
        overlap = UNKNOWN_OVERLAP
    evidence: dict[str, Any] = {
        "r0_complete": r0["r0_status"] == "R0_COMPLETE",
        **_validate_official_runtime_evidence(
            runtime=runtime,
            strict=strict,
            forward=forward,
            upstream=upstream,
            project_root=root,
        ),
        "preprocessing_v2_complete": False,
        "brainiac_embeddings_complete": False,
        "split_registry_locked": False,
        "resnet34_representation_contract_ready": False,
        "resnet34_inner_features_complete": False,
        "brainiac_extractor_contract_ready": False,
        "caft_trainer_contract_ready": False,
        "outer_head_calibration_locked": False,
        "promotion_evidence_ready": False,
        "external_dua": False,
        "external_lineage": False,
        "external_no_refit": False,
        "external_exactly_once": False,
        "pretraining_overlap_status": overlap,
    }
    optional = dict(DEFAULT_OPTIONAL_ARTIFACTS)
    if optional_artifact_paths:
        unknown_roles = set(optional_artifact_paths) - set(optional)
        if unknown_roles:
            raise RuntimeError("UNKNOWN_OPTIONAL_AUTHORITY_ROLE:" + ",".join(sorted(unknown_roles)))
        optional.update({key: Path(value) for key, value in optional_artifact_paths.items()})
    role_to_field = {
        "preprocessing_v2": "preprocessing_v2_complete",
        "brainiac_embeddings": "brainiac_embeddings_complete",
        "split_registry": "split_registry_locked",
        "resnet34_representation_contract": "resnet34_representation_contract_ready",
        "resnet34_inner_features": "resnet34_inner_features_complete",
        "brainiac_extractor_contract": "brainiac_extractor_contract_ready",
        "caft_trainer_contract": "caft_trainer_contract_ready",
        "outer_head_calibration_lock": "outer_head_calibration_locked",
        "promotion_evidence": "promotion_evidence_ready",
    }
    for role, candidate in optional.items():
        path = Path(candidate)
        if not path.exists():
            continue
        payload = _load_bound_json(role, path, project_root=root, bindings=bindings)
        authority = validate_future_authority(role, payload)
        ready = authority["ready"] is True
        if role != "external_authorization":
            evidence[role_to_field[role]] = ready

    if strict.get("environment_lock_sha256") != bindings["official_runtime"]["sha256"]:
        raise RuntimeError("STRICT_LOAD_RUNTIME_BINDING_MISMATCH")
    if forward.get("strict_load_receipt_file_sha256") != bindings["official_strict_load"]["sha256"]:
        raise RuntimeError("FORWARD_STRICT_LOAD_BINDING_MISMATCH")

    from src.research.brainiac_battery_terminal import verify_battery_terminal

    status_terminal, _ = _battery_state(battery)
    battery_terminal_report = verify_battery_terminal(
        manifest_path=battery_manifest_path,
        battery_root=Path(battery_manifest_path).parent,
        project_root=root,
        hash_checkpoints=status_terminal,
    )
    writer_state = inspect_writer_state(battery, process_provider=process_provider)
    supersedes = [
        {
            **schema3_binding,
            "status": "SUPERSEDED_BY_DERIVED_SCHEMA4",
        }
    ]
    payload = derive_readiness_payload(
        battery=battery,
        evidence=evidence,
        writer_state=writer_state,
        input_bindings=bindings,
        r0_report=r0,
        supersedes=supersedes,
        battery_terminal_report=battery_terminal_report,
    )
    payload["derived_evidence"] = evidence
    payload["writer_state"] = writer_state
    payload["battery_terminal_report"] = battery_terminal_report
    payload["main_method_x_battery_affected"] = False
    return payload


def _battery_state(battery: Mapping[str, Any]) -> tuple[bool, list[str]]:
    rows = battery.get("models")
    if not isinstance(rows, list) or len(rows) != 7:
        return False, ["FIVE_CNN_BATTERY_IDENTITY_INVALID"]
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return False, ["FIVE_CNN_BATTERY_IDENTITY_INVALID"]
        name = str(row.get("backbone") or "")
        if not name or name in by_name:
            return False, ["FIVE_CNN_BATTERY_IDENTITY_INVALID"]
        by_name[name] = row
    if set(by_name) != set(EFFECTIVE_FIVE) | set(EXCLUDED_BACKBONES):
        return False, ["FIVE_CNN_BATTERY_IDENTITY_INVALID"]
    if any(
        str(by_name[name].get("status")) != "PROTOCOL_EXCLUDED_NOT_STARTED"
        for name in EXCLUDED_BACKBONES
    ):
        return False, ["FIVE_CNN_BATTERY_EXCLUSION_INVALID"]
    terminal = all(str(by_name[name].get("status")) == "COMPLETED" for name in EFFECTIVE_FIVE)
    return terminal, []


def _writer_blockers(writer_state: Mapping[str, Any]) -> list[str]:
    if writer_state.get("inspection_status") != "PASS":
        return ["WRITER_LIVENESS_UNKNOWN"]
    active = writer_state.get("active_writers")
    owners = writer_state.get("recorded_owners")
    if not isinstance(active, list) or not isinstance(owners, list):
        return ["WRITER_LIVENESS_UNKNOWN"]
    if active:
        return ["ACTIVE_BRAINIAC_OR_CNN_WRITER"]
    for owner in owners:
        if not isinstance(owner, Mapping):
            return ["WRITER_LIVENESS_UNKNOWN"]
        liveness = str(owner.get("liveness") or "UNKNOWN").upper()
        if liveness in {"UNKNOWN", "NOT_MEASURABLE"}:
            return ["WRITER_LIVENESS_UNKNOWN"]
        if liveness == "LIVE":
            if owner.get("creation_time_matches") is not True:
                return ["RECORDED_OWNER_PID_REUSE_OR_CREATION_MISMATCH"]
            return ["UNLISTED_LIVE_RECORDED_OWNER"]
    return []


def _missing(evidence: Mapping[str, Any], field: str, blocker: str) -> list[str]:
    value = evidence.get(field)
    if value is True:
        return []
    if value in {"UNKNOWN", "NOT_MEASURABLE", None}:
        return [blocker]
    return [blocker]


def derive_action_decisions(
    *,
    battery: Mapping[str, Any],
    evidence: Mapping[str, Any],
    writer_state: Mapping[str, Any],
    battery_terminal_report: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return one fail-closed decision for each immutable B1 action."""

    _status_terminal, identity_blockers = _battery_state(battery)
    terminal = bool(
        isinstance(battery_terminal_report, Mapping)
        and battery_terminal_report.get("identity_valid") is True
        and battery_terminal_report.get("terminal") is True
        and not battery_terminal_report.get("blockers")
    )
    global_blockers = identity_blockers + _writer_blockers(writer_state)
    prereqs = {
        "PREPARE_RAW_T1": (
            ("r0_complete", "R0_NOT_COMPLETE"),
            ("official_runtime", "OFFICIAL_RUNTIME_NOT_READY"),
            ("strict_load", "OFFICIAL_STRICT_LOAD_NOT_READY"),
            ("official_forward", "OFFICIAL_FORWARD_NOT_READY"),
        ),
        "EXTRACT_FROZEN_BRAINIAC_EMBEDDINGS": (
            ("preprocessing_v2_complete", "BRAINIAC_PREPROCESSING_V2_INCOMPLETE"),
            (
                "brainiac_extractor_contract_ready",
                "BRAINIAC_EXTRACTOR_CONTRACT_NOT_READY",
            ),
        ),
        "EXTRACT_RESNET34_INNER_FEATURES": (
            ("split_registry_locked", "PRIMARY_SPLIT_REGISTRY_NOT_LOCKED"),
            (
                "resnet34_representation_contract_ready",
                "RESNET34_REPRESENTATION_CONTRACT_NOT_READY",
            ),
        ),
        "CAFT_TRAINING": (
            ("preprocessing_v2_complete", "BRAINIAC_PREPROCESSING_V2_INCOMPLETE"),
            ("brainiac_embeddings_complete", "BRAINIAC_EMBEDDINGS_INCOMPLETE"),
            ("split_registry_locked", "PRIMARY_SPLIT_REGISTRY_NOT_LOCKED"),
            (
                "resnet34_representation_contract_ready",
                "RESNET34_REPRESENTATION_CONTRACT_NOT_READY",
            ),
            ("resnet34_inner_features_complete", "RESNET34_INNER_FEATURES_INCOMPLETE"),
            ("caft_trainer_contract_ready", "CAFT_TRAINER_CONTRACT_NOT_READY"),
        ),
        "OUTER_FEATURE_SNAPSHOT": (
            ("outer_head_calibration_locked", "HEAD_AND_CALIBRATION_NOT_LOCKED"),
        ),
        "PROMOTION": (("promotion_evidence_ready", "PROMOTION_EVIDENCE_NOT_READY"),),
        "EXTERNAL_EVALUATION": (
            ("external_dua", "EXTERNAL_DUA_MISSING"),
            ("external_lineage", "EXTERNAL_LINEAGE_MISSING"),
            ("external_no_refit", "EXTERNAL_NO_REFIT_AUTHORIZATION_MISSING"),
            ("external_exactly_once", "EXTERNAL_EXACTLY_ONCE_AUTHORIZATION_MISSING"),
        ),
    }
    base_preparation = list(prereqs["PREPARE_RAW_T1"])
    decisions: dict[str, dict[str, Any]] = {}
    for action in ACTIONS:
        blockers = list(global_blockers)
        requirements = list(prereqs[action])
        if action in {
            "EXTRACT_FROZEN_BRAINIAC_EMBEDDINGS",
            "EXTRACT_RESNET34_INNER_FEATURES",
        }:
            requirements = base_preparation + requirements
        if action in BATTERY_TERMINAL_ACTIONS or action == "EXTERNAL_EVALUATION":
            if not terminal and not identity_blockers:
                blockers.append("FIVE_CNN_BATTERY_NOT_TERMINAL")
        for field, blocker in requirements:
            blockers.extend(_missing(evidence, field, blocker))
        blockers = list(dict.fromkeys(blockers))
        decisions[action] = {
            "allowed": not blockers,
            "status": "PASS" if not blockers else "BLOCKED",
            "blockers": blockers,
        }
    return decisions


def derive_readiness_payload(
    *,
    battery: Mapping[str, Any],
    evidence: Mapping[str, Any],
    writer_state: Mapping[str, Any],
    input_bindings: Mapping[str, Mapping[str, Any]],
    battery_terminal_report: Mapping[str, Any] | None = None,
    r0_report: Mapping[str, Any] | None = None,
    supersedes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    actions = derive_action_decisions(
        battery=battery,
        evidence=evidence,
        writer_state=writer_state,
        battery_terminal_report=battery_terminal_report,
    )
    authorized = [action for action in ACTIONS if actions[action]["allowed"]]
    preparation_ready = all(actions[action]["allowed"] for action in PREPARATION_ACTIONS)
    training_allowed = actions["CAFT_TRAINING"]["allowed"]
    training_blockers = actions["CAFT_TRAINING"]["blockers"]
    sole = (
        "FIVE_CNN_BATTERY_NOT_TERMINAL"
        if training_blockers == ["FIVE_CNN_BATTERY_NOT_TERMINAL"]
        else None
    )
    overlap = str(evidence.get("pretraining_overlap_status") or "UNKNOWN")
    return {
        "schema_version": 4,
        "artifact_type": "BRAINIAC_R1_DERIVED_READINESS",
        "status": "READY" if training_allowed else "BLOCKED",
        "ready": training_allowed,
        "preparation_ready": preparation_ready,
        "training_allowed": training_allowed,
        "authorized_actions": authorized,
        "actions": actions,
        "sole_runtime_blocker": sole,
        "r0": dict(r0_report or {}),
        "input_bindings": {key: dict(value) for key, value in sorted(input_bindings.items())},
        "pretraining_overlap_status": overlap,
        "claim_limits": {
            "claim_ceiling": CLAIM_CEILING,
            "novelty_status": NOVELTY_CEILING,
            "confirmatory_pretraining_independence": overlap == "VERIFIED_DISJOINT",
            "external_generalization": False,
            "diagnostic_claim_allowed": False,
            "causal_claim_allowed": False,
        },
        "external_evaluation_status": (
            "EXTERNAL_ELIGIBLE_LOCKED_NOT_RUN"
            if actions["EXTERNAL_EVALUATION"]["allowed"]
            else "EXTERNAL_UNAVAILABLE_NONBLOCKING"
        ),
        "lora_unlocked": False,
        "outer_test_touched": False,
        "training_started": False,
        "human_approval_required": False,
        "manual_override_allowed": False,
        "unknown_is_pass": False,
        "supersedes": [dict(row) for row in supersedes],
    }


_FORBIDDEN_READINESS_KEYS = frozenset(
    {
        "subject_id",
        "patient_id",
        "raw_id",
        "raw_identifier",
        "raw_path",
        "ab_label",
        "label",
        "labels",
        "pet_input",
        "pet_path",
        "age",
        "sex",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
        "tabular",
        "predictors",
        "subject_rows",
        "subjects",
    }
)


def _assert_readiness_privacy(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_READINESS_KEYS:
                raise RuntimeError(f"READINESS_PRIVACY_FIELD_FORBIDDEN:{key}")
            _assert_readiness_privacy(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_readiness_privacy(nested)


def write_readiness_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    append_only: bool = False,
) -> dict[str, Any]:
    """Atomically materialize schema 4 with a canonical self-integrity digest."""

    target = Path(path)
    body = dict(payload)
    _assert_readiness_privacy(body)
    body.pop("integrity_sha256", None)
    body["integrity_sha256"] = _sha256_bytes(_canonical_json_bytes(body))
    raw = _canonical_json_bytes(body)
    target.parent.mkdir(parents=True, exist_ok=True)
    if append_only and target.exists():
        existing = target.read_bytes()
        if existing != raw:
            raise RuntimeError(f"APPEND_ONLY_ARTIFACT_DRIFT:{target}")
        return {
            "path": str(target.resolve()),
            "bytes": len(existing),
            "sha256": _sha256_bytes(existing),
        }
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        temp = Path(temp_name)
        if temp.exists():
            temp.unlink()
    return {"path": str(target.resolve()), "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _binding_matches(binding: Mapping[str, Any], *, project_root: Path) -> bool:
    try:
        path = _beneath(Path(str(binding.get("path") or "")), project_root, must_exist=True)
        actual = bind_file(path, project_root=project_root)
        return (
            int(binding.get("bytes", -1)) == actual["bytes"]
            and str(binding.get("sha256") or "").casefold() == actual["sha256"]
            and Path(str(binding.get("path"))).resolve() == path
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _assert_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("NONFINITE_OR_INVALID_JSON")
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_finite(nested)


def validate_materialized_readiness(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    resolved = _beneath(Path(path), root, must_exist=True)
    payload, raw = _load_json_strict(resolved)
    _assert_finite(payload)
    _assert_readiness_privacy(payload)
    if expected_sha256 is not None and _sha256_bytes(raw) != expected_sha256.casefold():
        raise RuntimeError("READINESS_EXTERNAL_SHA256_MISMATCH")
    if payload.get("schema_version") != 4:
        raise RuntimeError("READINESS_SCHEMA_UNSUPPORTED")
    expected_integrity = str(payload.get("integrity_sha256") or "")
    body = dict(payload)
    body.pop("integrity_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(body)) != expected_integrity:
        raise RuntimeError("READINESS_INTEGRITY_SHA256_MISMATCH")
    bindings = payload.get("input_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise RuntimeError("READINESS_INPUT_BINDING_MISSING")
    for binding in bindings.values():
        if not isinstance(binding, Mapping) or not _binding_matches(binding, project_root=root):
            raise RuntimeError("READINESS_INPUT_BINDING_MISMATCH")
        bound_path = Path(str(binding["path"]))
        if bound_path.suffix.casefold() == ".json":
            _load_json_strict(bound_path)
    if payload.get("outer_test_touched") is not False or payload.get("lora_unlocked") is not False:
        raise RuntimeError("READINESS_CLAIM_BOUNDARY_INVALID")
    if payload.get("human_approval_required") is not False:
        raise RuntimeError("READINESS_SUPERVISION_INVALID")
    if payload.get("manual_override_allowed") is not False:
        raise RuntimeError("READINESS_SUPERVISION_INVALID")
    if payload.get("unknown_is_pass") is not False:
        raise RuntimeError("READINESS_SUPERVISION_INVALID")
    return payload
