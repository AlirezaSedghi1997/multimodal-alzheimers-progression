"""Versioned checkpoint staging for Method X (Task 4.2).

Why this module exists
----------------------
The trainer used to write a complete resume payload to the authoritative H
volume after *every* optimizer step: 533 MB for ResNet18 and 1.02 GB for
ResNet34, on a USB HDD, roughly 228 times per epoch. That is the dominant cost
of the current run and it buys nothing that a small journal plus a periodic full
snapshot cannot provide.

The split enforced here:

* full payloads land on the NVMe staging root allowlisted by the storage policy,
  keeping only ``resume_current.pt`` and ``resume_previous.pt``;
* one small fsynced journal record per optimizer step goes to H, so evidence of
  every step is durable on the authoritative volume;
* full payloads are mirrored to H at epoch end, fold end, clean stop, and
  terminal state, and a C generation is pruned only after a hash-verified commit;
* ``best_model_state`` leaves the repeating payload entirely and is written to a
  hash-locked H file only when validation improves.

This module changes *where and how often* state is written. It never changes what
is trained: no scientific parameter, data order, or optimizer behaviour is
touched here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.training.storage_locality import DiskUsage, StoragePolicy, _disk_usage, verify_path

# Mirror/replay vocabularies are owned by the resume contract so the flag values
# and the digested runtime contract can never drift apart.
from src.training.method_x_resume import (
    _ALLOWED_H_MIRROR_POLICIES,
    _ALLOWED_REPLAY_POLICIES,
)

CHECKPOINT_STAGING_SCHEMA_VERSION = 2
CURRENT_BASENAME = "resume_current.pt"
PREVIOUS_BASENAME = "resume_previous.pt"
MIRROR_STATE_BASENAME = "checkpoint_mirror_state.json"
_STAGING_TEMP_NAME = re.compile(r"^resume_(?:current|previous)\.pt\.\d+\.tmp$")
EXACT_BINDING_MODE = "EXACT_COMMIT_V2"
_ZERO_SHA256 = "0" * 64

MIRROR_REASONS = frozenset({"EPOCH_END", "FOLD_END", "CLEAN_STOP", "TERMINAL"})

# argparse destinations added by Task 4. Every one of these is runtime-only and
# must be excluded from the training contract, otherwise supplying a staging
# root would change the contract digest and break resume for the live
# ResNet18/ResNet34 checkpoints.
STAGING_RUNTIME_FLAGS = (
    "checkpoint_staging_root",
    "checkpoint_full_cadence",
    "checkpoint_h_mirror",
    "resume_replay_policy",
)

DEFAULT_STORAGE_POLICY_PATH = "configs/policy/method_x_storage_locality_v1.yaml"

# A hard kill can land at most one cadence-worth of steps past the last full
# checkpoint. Anything beyond that means the journal and the payload disagree,
# which is a corruption signal rather than something to replay through.
DEFAULT_FULL_CADENCE = 32
MAX_DETERMINISTIC_REPLAY_STEPS = DEFAULT_FULL_CADENCE - 1

# Exit code the battery runner reads as "paused, resumable" rather than failed.
RESUMABLE_EXIT_CODE = 75

# Keys that must never appear in a staged payload. Presence is enough to
# fail: a None value still serializes the key, and that is what crashed the
# live ResNet18 resume (BEST_STATE_MUST_NOT_BE_STAGED:best_model_state_dict).
# best_averaged_state_dict is the same second copy under the trainer's name.
FORBIDDEN_STAGED_PAYLOAD_KEYS = frozenset(
    {
        "best_model_state",
        "best_model_state_dict",
        "best_averaged_state",
        "best_averaged_state_dict",
    }
)
_FORBIDDEN_PAYLOAD_KEYS = FORBIDDEN_STAGED_PAYLOAD_KEYS

GIB = 1024**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    """Force one file's bytes to the platform's stable storage.

    Windows ``_commit`` requires a writable descriptor, so the file is opened
    read-write rather than read-only.
    """

    try:
        handle = os.open(path, os.O_RDWR)
    except OSError as exc:
        raise RuntimeError(f"FSYNC_FAILED:OPEN:{path}") from exc
    try:
        os.fsync(handle)
    except OSError as exc:
        raise RuntimeError(f"FSYNC_FAILED:COMMIT:{path}") from exc
    finally:
        os.close(handle)


def _atomic_replace_with_retry(
    src: Path | str,
    dst: Path | str,
    *,
    attempts: int = 20,
    base_seconds: float = 0.05,
) -> None:
    """Atomic file replacement with exponential backoff retry for Windows sharing contention."""

    src_p = Path(src)
    dst_p = Path(dst)
    last_err: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            os.replace(src_p, dst_p)
            return
        except (PermissionError, FileExistsError, OSError) as exc:
            winerror = getattr(exc, "winerror", None)
            if winerror in (5, 32, 183) or isinstance(exc, PermissionError) or "access is denied" in str(exc).lower():
                last_err = exc
                time.sleep(base_seconds * (1.3 ** attempt))
                if attempt > 3 and dst_p.exists():
                    try:
                        dst_p.unlink(missing_ok=True)
                    except Exception:
                        pass
                continue
            raise
    if last_err is not None:
        raise last_err


def _canonical_sha256(payload: Mapping[str, Any], *, exclude: tuple[str, ...] = ()) -> str:
    canonical = {key: value for key, value in dict(payload).items() if key not in exclude}
    raw = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    if len(text) != 64:
        return False
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def _atomic_write_json_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _fsync_path(temp)
        _atomic_replace_with_retry(temp, path)
        _fsync_path(path)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def rng_state_digest(state: Mapping[str, Any] | None = None) -> str:
    """Stable digest of the captured RNG state.

    The journal stores this rather than the state itself: it is small, carries no
    identifiers, and is enough to prove on recovery that a replayed step landed
    on the same random stream.
    """

    if state is None:
        from src.utils.reproducibility import capture_rng_state

        state = capture_rng_state()
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode("utf-8"))
        value = state[key]
        if value is None:
            digest.update(b"\x00")
            continue
        raw = getattr(value, "numpy", None)
        if callable(raw):  # torch.ByteTensor RNG states
            digest.update(bytes(value.reshape(-1).tolist()))
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


# --- Journal ------------------------------------------------------------------


def build_journal_record(
    *,
    optimizer_step: int,
    epoch: int,
    epoch_batch_cursor: int,
    sampler_cursor: int,
    generation: int,
    checkpoint_committed: bool,
    payload_sha256: str | None,
    loss_sum: float,
    loss_count: int,
    rng_digest: str,
    payload_size_bytes: int | None = None,
    run_id: str | None = None,
    backbone: str | None = None,
    fold: int | None = None,
    event: str = "OPTIMIZER_STEP",
    scientific_contract_digest: str | None = None,
    runtime_resume_digest: str | None = None,
    storage_policy_digest: str | None = None,
    binding_mode: str | None = None,
) -> dict[str, Any]:
    """One durable, identifier-free record describing a committed step."""

    exact_binding = binding_mode == EXACT_BINDING_MODE or (
        binding_mode is None
        and run_id is not None
        and backbone is not None
        and fold is not None
        and (not checkpoint_committed or payload_size_bytes is not None)
    )
    return {
        "schema_version": CHECKPOINT_STAGING_SCHEMA_VERSION,
        "binding_mode": EXACT_BINDING_MODE if exact_binding else "LEGACY_COMPAT",
        "event": str(event),
        "run_id": str(run_id) if run_id is not None else None,
        "backbone": str(backbone) if backbone is not None else None,
        "fold": int(fold) if fold is not None else None,
        "optimizer_step": int(optimizer_step),
        "epoch": int(epoch),
        "epoch_batch_cursor": int(epoch_batch_cursor),
        "sampler_cursor": int(sampler_cursor),
        "generation": int(generation),
        "checkpoint_committed": bool(checkpoint_committed),
        "payload_sha256": str(payload_sha256) if payload_sha256 else None,
        "payload_size_bytes": (
            int(payload_size_bytes) if payload_size_bytes is not None else None
        ),
        "loss_sum": float(loss_sum),
        "loss_count": int(loss_count),
        "rng_digest": str(rng_digest),
        "scientific_contract_digest": (
            str(scientific_contract_digest) if scientific_contract_digest else None
        ),
        "runtime_resume_digest": (
            str(runtime_resume_digest) if runtime_resume_digest else None
        ),
        "storage_policy_digest": (
            str(storage_policy_digest) if storage_policy_digest else None
        ),
        "recorded_at_utc": _utc_now(),
    }


def append_journal_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one JSONL record and fsync it.

    Append-and-fsync is used rather than atomic replace: the journal must survive
    a kill at any instant, and rewriting a whole file per step would reintroduce
    the very cost this module removes.
    """

    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    prior_records = _read_journal_records(journal_path, require_chain=False)
    previous_sha256 = (
        str(prior_records[-1].get("record_sha256"))
        if prior_records and _is_sha256(prior_records[-1].get("record_sha256"))
        else _ZERO_SHA256
    )
    entry = dict(record)
    entry["previous_record_sha256"] = previous_sha256
    entry["record_sha256"] = _canonical_sha256(entry, exclude=("record_sha256",))
    line = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    try:
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RuntimeError(f"FSYNC_FAILED:JOURNAL:{journal_path}") from exc
    if isinstance(record, dict):
        record.clear()
        record.update(entry)
    return journal_path


def read_journal_tail(path: str | Path) -> dict[str, Any] | None:
    """Return the last intact record, ignoring a torn final line.

    A hard kill mid-append can leave a partial trailing line. That line is
    discarded rather than treated as corruption, because the previous record is
    still a valid recovery point.
    """

    records = _read_journal_records(Path(path), require_chain=False)
    return records[-1] if records else None


def _is_staging_run_id(value: Any) -> bool:
    """Accept the C-layout identity token; reject empty or path-like values."""

    text = str(value or "")
    if not text or len(text) > 64:
        return False
    return not any(token in text for token in ("/", "\\", ":", ".."))


def fold_journal_path(h_output_dir: Path, fold: int) -> Path:
    return Path(h_output_dir) / "journals" / f"fold_{int(fold)}_optimizer_steps.jsonl"


def resolve_staging_run_id(*, live_run_id: str, journal_path: str | Path) -> str:
    """Keep C staging under the journal identity across source-only digest hops.

    ``run_contract["digest"][:16]`` changes whenever trainer source is edited.
    The NVMe payloads, journal ``run_id``, and best-state provenance do not.
    Looking under the live prefix after a source-only migration is how
    ``NO_VALID_EXACT_RESUME_SOURCE_FOR_LATEST_COMMIT`` was raised on a still-
    valid ResNet18 C payload.
    """

    live = str(live_run_id or "").strip()
    if not _is_staging_run_id(live):
        raise RuntimeError(f"INVALID_STAGING_RUN_ID:{live_run_id}")
    path = Path(journal_path)
    if not path.exists():
        return live
    records = _read_journal_records(path, require_chain=True)
    if not records:
        return live
    present = [str(record.get("run_id")) for record in records if record.get("run_id")]
    if present and (
        len(present) != len(records) or len(set(present)) != 1
    ):
        raise RuntimeError("JOURNAL_STAGING_RUN_ID_INCONSISTENT")
    if not present:
        return live
    resolved = present[0]
    if not _is_staging_run_id(resolved):
        raise RuntimeError("JOURNAL_STAGING_RUN_ID_INVALID")
    return resolved


# --- Layout -------------------------------------------------------------------


@dataclass(frozen=True)
class StagingLayout:
    """Absolute paths for one (run, backbone, fold) triple.

    Payloads live under the policy's C staging root; journal, mirrors, and the
    best-state sidecar live on H, which stays the only authoritative volume.
    """

    run_id: str
    backbone: str
    fold: int
    staging_dir: Path
    mirror_dir: Path
    journal_path: Path
    best_state_path: Path
    best_state_sidecar_path: Path

    @property
    def mirror_state_path(self) -> Path:
        return self.mirror_dir / MIRROR_STATE_BASENAME

    @property
    def current_path(self) -> Path:
        return self.staging_dir / CURRENT_BASENAME

    @property
    def previous_path(self) -> Path:
        return self.staging_dir / PREVIOUS_BASENAME

    @classmethod
    def build(
        cls,
        *,
        policy: StoragePolicy,
        run_id: str,
        backbone: str,
        fold: int,
        h_output_dir: Path,
        staging_root_override: Path | None = None,
    ) -> "StagingLayout":
        root = Path(staging_root_override or policy.checkpoint_staging_root)
        staging_dir = root / str(run_id) / str(backbone) / str(int(fold))
        h_dir = Path(h_output_dir)
        return cls(
            run_id=str(run_id),
            backbone=str(backbone),
            fold=int(fold),
            staging_dir=staging_dir,
            mirror_dir=h_dir / "checkpoint_mirror" / f"fold_{int(fold)}",
            journal_path=fold_journal_path(h_dir, fold),
            best_state_path=h_dir / f"method_x_3d_fold_{int(fold)}_best_state.pt",
            best_state_sidecar_path=(
                h_dir / f"method_x_3d_fold_{int(fold)}_best_state.provenance.json"
            ),
        )


# --- Stager -------------------------------------------------------------------


class CheckpointStager:
    """Cadence-based staging with a per-step journal and hash-verified mirrors."""

    def __init__(
        self,
        *,
        policy: StoragePolicy,
        layout: StagingLayout,
        full_cadence: int = 32,
        disk_usage: Callable[[Path], DiskUsage] = _disk_usage,
        save_fn: Callable[[Any, Path], None] | None = None,
        load_fn: Callable[[Path], Any] | None = None,
        copy_fn: Callable[[Path, Path], None] | None = None,
    ) -> None:
        if int(full_cadence) < 1:
            raise ValueError(f"INVALID_CHECKPOINT_FULL_CADENCE:{full_cadence}")
        self.policy = policy
        self.layout = layout
        self.full_cadence = int(full_cadence)
        self._disk_usage = disk_usage
        self._save_fn = save_fn or self._default_save
        self._load_fn = load_fn or self._default_load
        self._copy_fn = copy_fn or (lambda src, dst: shutil.copy2(src, dst))
        self._generation = 0
        self._last_payload_sha256: str | None = None
        self._last_payload_size_bytes: int | None = None
        self._best_metric: float | None = None
        self._restore_runtime_metadata()

    @property
    def generation(self) -> int:
        """Monotonic count of committed staged generations for this fold."""

        return self._generation

    def _restore_runtime_metadata(self) -> None:
        """Restore generation metadata from the authoritative H journal only.

        A readable C payload can exist before its journal append commits.  It is
        therefore recovery evidence, never generation authority.
        """

        generations: list[int] = []
        records = _read_journal_records(self.layout.journal_path, require_chain=True)
        for record in records:
            try:
                generations.append(int(record.get("generation", 0)))
            except (TypeError, ValueError):
                raise RuntimeError("JOURNAL_GENERATION_INVALID")
        self._generation = max(generations, default=0)
        committed = [
            record
            for record in records
            if record.get("checkpoint_committed")
            and _is_sha256(record.get("payload_sha256"))
            and isinstance(record.get("payload_size_bytes"), int)
        ]
        if committed:
            newest = max(committed, key=lambda row: int(row.get("generation", 0)))
            self._last_payload_sha256 = str(newest["payload_sha256"])
            self._last_payload_size_bytes = int(newest["payload_size_bytes"])

    def _payload_is_journal_committed(self, path: Path) -> bool:
        """Return true only when ``path`` exactly matches a committed record."""

        if not path.is_file() or path.stat().st_size == 0:
            return False
        digest = _sha256_file(path)
        size = path.stat().st_size
        return any(
            record.get("checkpoint_committed")
            and record.get("payload_sha256") == digest
            and int(record.get("payload_size_bytes", -1)) == size
            for record in _read_journal_records(
                self.layout.journal_path,
                require_chain=True,
            )
        )

    # -- defaults kept lazy so the module imports without torch present -------

    @staticmethod
    def _default_save(payload: Any, path: Path) -> None:
        import torch

        torch.save(payload, path)

    @staticmethod
    def _default_load(path: Path) -> Any:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)

    # -- preflight ------------------------------------------------------------

    def _assert_staging_root_allowed(self) -> None:
        """Reject any staging directory the storage policy does not allowlist."""

        target = self.layout.staging_dir / CURRENT_BASENAME
        verdict = verify_path(target, self.policy)
        if not verdict.get("allowed"):
            raise RuntimeError(
                f"STAGING_ROOT_OUTSIDE_POLICY:{verdict.get('reason', 'UNKNOWN')}"
            )

    def _staging_bytes(self) -> int:
        if not self.layout.staging_dir.exists():
            return 0
        total = 0
        for path in self.layout.staging_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def preflight(self, *, staging_bytes_override: int | None = None) -> dict[str, Any]:
        """Report free space and quota status without creating anything."""

        self._assert_staging_root_allowed()
        try:
            probe = (
                self.layout.staging_dir
                if self.layout.staging_dir.exists()
                else Path(self.layout.staging_dir.anchor)
            )
            free_bytes = int(self._disk_usage(probe).free)
        except OSError:
            free_bytes = -1
        staging_bytes = (
            int(staging_bytes_override)
            if staging_bytes_override is not None
            else self._staging_bytes()
        )
        free_status = (
            "PASS" if free_bytes >= self.policy.minimum_free_gb * GIB else "BLOCK"
        )
        quota_status = (
            "PASS" if staging_bytes <= self.policy.maximum_staging_gb * GIB else "BLOCK"
        )
        status = "PASS" if free_status == "PASS" and quota_status == "PASS" else "BLOCK"
        return {
            "schema_version": CHECKPOINT_STAGING_SCHEMA_VERSION,
            "status": status,
            "free_space_status": free_status,
            "quota_status": quota_status,
            "free_bytes": free_bytes,
            "staging_bytes": staging_bytes,
            "minimum_free_gb": self.policy.minimum_free_gb,
            "maximum_staging_gb": self.policy.maximum_staging_gb,
            "checked_at_utc": _utc_now(),
        }

    def require_preflight(self) -> dict[str, Any]:
        """Fail closed. There is deliberately no silent fallback to H-per-step."""

        report = self.preflight()
        if report["status"] != "PASS":
            raise RuntimeError(
                "STAGING_PREFLIGHT_BLOCKED:"
                + json.dumps(
                    {
                        "free_space_status": report["free_space_status"],
                        "quota_status": report["quota_status"],
                    },
                    sort_keys=True,
                )
            )
        return report

    def require_projected_capacity(self, projected_payload_bytes: int) -> dict[str, Any]:
        """Check quota and the post-write free-space floor before serialization."""

        projected = int(projected_payload_bytes)
        if projected < 0:
            raise ValueError("PROJECTED_CHECKPOINT_SIZE_INVALID")
        self._assert_staging_root_allowed()
        probe = (
            self.layout.staging_dir
            if self.layout.staging_dir.exists()
            else Path(self.layout.staging_dir.anchor)
        )
        try:
            usage = self._disk_usage(probe)
        except OSError as exc:
            raise RuntimeError("STAGING_CAPACITY_NOT_MEASURABLE") from exc
        current_size = (
            self.layout.current_path.stat().st_size
            if self.layout.current_path.is_file()
            else 0
        )
        other_bytes = max(0, self._staging_bytes() - current_size)
        projected_total = other_bytes + current_size + projected
        if projected_total > self.policy.maximum_staging_gb * GIB:
            raise RuntimeError("STAGING_PROJECTED_QUOTA_EXCEEDED")
        if int(usage.free) - projected < self.policy.minimum_free_gb * GIB:
            raise RuntimeError("STAGING_PROJECTED_FREE_SPACE_BLOCKED")
        return {
            "status": "PASS",
            "projected_payload_bytes": projected,
            "projected_staging_bytes": projected_total,
            "projected_free_bytes": int(usage.free) - projected,
        }

    # -- full checkpoint ------------------------------------------------------

    def write_full_checkpoint(
        self,
        payload: Mapping[str, Any],
        *,
        optimizer_step: int,
        epoch: int,
        epoch_batch_cursor: int,
    ) -> dict[str, Any]:
        """Atomically rotate current -> previous and write a new current."""

        forbidden = sorted(_FORBIDDEN_PAYLOAD_KEYS.intersection(payload))
        if forbidden:
            raise RuntimeError(f"BEST_STATE_MUST_NOT_BE_STAGED:{','.join(forbidden)}")
        self._assert_staging_root_allowed()
        if self._last_payload_size_bytes is not None:
            self.require_projected_capacity(self._last_payload_size_bytes)
        self.layout.staging_dir.mkdir(parents=True, exist_ok=True)
        generation = self._generation + 1
        temp_path = self.layout.staging_dir / f"{CURRENT_BASENAME}.{generation}.tmp"
        enriched_payload = dict(payload)
        exact_binding = all(
            _is_sha256(enriched_payload.get(key))
            for key in (
                "run_contract_digest",
                "scientific_contract_digest",
                "runtime_resume_digest",
                "storage_policy_digest",
            )
        )
        if exact_binding:
            enriched_payload.update(
                {
                    "checkpoint_staging_schema_version": CHECKPOINT_STAGING_SCHEMA_VERSION,
                    "checkpoint_binding_mode": EXACT_BINDING_MODE,
                    "staging_run_id": self.layout.run_id,
                    "staging_backbone": self.layout.backbone,
                    "staging_fold": self.layout.fold,
                    "checkpoint_generation": generation,
                    "last_committed_optimizer_step": int(optimizer_step),
                    "optimizer_step_count": int(optimizer_step),
                    "epoch": int(epoch),
                    "epoch_batch_cursor": int(epoch_batch_cursor),
                }
            )
            enriched_payload["checkpoint_payload_identity_sha256"] = _canonical_sha256(
                _payload_identity_fields(enriched_payload)
            )
        try:
            self._save_fn(enriched_payload, temp_path)
            _fsync_path(temp_path)
            digest = _sha256_file(temp_path)
            payload_size = temp_path.stat().st_size
            # The first generation has no prior size estimate; validate its
            # actual size before rotating any committed payload.
            self.require_projected_capacity(payload_size)
            loaded = self._load_fn(temp_path)
            if not isinstance(loaded, Mapping):
                raise RuntimeError("STAGED_CHECKPOINT_SELF_VALIDATION_FAILED")
            if exact_binding and (
                int(loaded.get("checkpoint_generation", -1)) != generation
                or int(loaded.get("last_committed_optimizer_step", -1))
                != int(optimizer_step)
                or loaded.get("checkpoint_payload_identity_sha256")
                != enriched_payload["checkpoint_payload_identity_sha256"]
            ):
                raise RuntimeError("STAGED_CHECKPOINT_SELF_VALIDATION_FAILED")
            # Never rotate an unjournaled C payload over the last committed
            # previous generation.  ``os.replace(temp, current)`` safely
            # discards that uncommitted current after the replacement itself
            # has been fsynced and self-validated.
            if self.layout.current_path.exists() and (
                not exact_binding
                or self._payload_is_journal_committed(self.layout.current_path)
            ):
                _atomic_replace_with_retry(self.layout.current_path, self.layout.previous_path)
            _atomic_replace_with_retry(temp_path, self.layout.current_path)
            _fsync_path(self.layout.current_path)
        finally:
            # A partial temp must never become a resume source.
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        self._generation = generation
        self._last_payload_sha256 = digest
        self._last_payload_size_bytes = payload_size
        return {
            "generation": generation,
            "payload_sha256": digest,
            "payload_size_bytes": payload_size,
            "payload_identity_sha256": enriched_payload.get(
                "checkpoint_payload_identity_sha256"
            ),
            "binding_mode": EXACT_BINDING_MODE if exact_binding else "LEGACY_COMPAT",
            "optimizer_step": int(optimizer_step),
            "epoch": int(epoch),
            "epoch_batch_cursor": int(epoch_batch_cursor),
            "path": str(self.layout.current_path),
            "scientific_contract_digest": enriched_payload.get(
                "scientific_contract_digest"
            ),
            "runtime_resume_digest": enriched_payload.get("runtime_resume_digest"),
            "storage_policy_digest": enriched_payload.get("storage_policy_digest"),
        }

    def should_write_full(self, optimizer_step: int) -> bool:
        """True when this optimizer step lands on the full-checkpoint cadence."""

        return int(optimizer_step) % self.full_cadence == 0

    def record_step(
        self,
        *,
        payload_factory: Callable[[], Mapping[str, Any]],
        optimizer_step: int,
        epoch: int,
        epoch_batch_cursor: int,
        sampler_cursor: int,
        loss_sum: float,
        loss_count: int,
        rng_digest: str,
        force_full: bool = False,
    ) -> dict[str, Any]:
        """Journal every step; build and stage a full payload only on cadence.

        ``payload_factory`` is a callable rather than a value so that the
        expensive state copy is skipped entirely off cadence.
        """

        due = force_full or int(optimizer_step) % self.full_cadence == 0
        written: dict[str, Any] | None = None
        if due:
            written = self.write_full_checkpoint(
                payload_factory(),
                optimizer_step=optimizer_step,
                epoch=epoch,
                epoch_batch_cursor=epoch_batch_cursor,
            )
        record = build_journal_record(
            optimizer_step=optimizer_step,
            epoch=epoch,
            epoch_batch_cursor=epoch_batch_cursor,
            sampler_cursor=sampler_cursor,
            generation=self._generation,
            checkpoint_committed=bool(written),
            payload_sha256=written["payload_sha256"] if written else None,
            payload_size_bytes=written["payload_size_bytes"] if written else None,
            loss_sum=loss_sum,
            loss_count=loss_count,
            rng_digest=rng_digest,
            run_id=self.layout.run_id,
            backbone=self.layout.backbone,
            fold=self.layout.fold,
            scientific_contract_digest=(
                written.get("scientific_contract_digest") if written else None
            ),
            runtime_resume_digest=(
                written.get("runtime_resume_digest") if written else None
            ),
            storage_policy_digest=(
                written.get("storage_policy_digest") if written else None
            ),
            binding_mode=(written.get("binding_mode") if written else None),
        )
        append_journal_record(self.layout.journal_path, record)
        return {
            "checkpoint_committed": bool(written),
            "generation": self._generation,
            "journal_record": record,
        }

    def clean_stop(
        self,
        *,
        payload_factory: Callable[[], Mapping[str, Any]],
        optimizer_step: int,
        epoch: int,
        epoch_batch_cursor: int,
        sampler_cursor: int,
        loss_sum: float,
        loss_count: int,
        rng_digest: str,
        reason: str,
    ) -> dict[str, Any]:
        """Commit a full checkpoint, journal it, and mirror to H before exiting.

        A controlled stop must leave zero replay work: without this the staged
        state could sit up to one cadence behind, and Ctrl+C would silently cost
        the operator those steps on the next start.
        """

        staged = self.record_step(
            payload_factory=payload_factory,
            optimizer_step=optimizer_step,
            epoch=epoch,
            epoch_batch_cursor=epoch_batch_cursor,
            sampler_cursor=sampler_cursor,
            loss_sum=loss_sum,
            loss_count=loss_count,
            rng_digest=rng_digest,
            force_full=True,
        )
        mirror = self.mirror_to_h(reason="CLEAN_STOP")
        return {
            "reason": str(reason),
            "checkpoint_committed": bool(staged["checkpoint_committed"]),
            "generation": staged["generation"],
            "mirror": mirror,
            "exit_code": RESUMABLE_EXIT_CODE,
        }

    def commit_boundary(
        self,
        *,
        payload_factory: Callable[[], Mapping[str, Any]],
        optimizer_step: int,
        epoch: int,
        epoch_batch_cursor: int,
        sampler_cursor: int,
        loss_sum: float,
        loss_count: int,
        rng_digest: str,
        reason: str,
    ) -> dict[str, Any]:
        """Force a full, journaled generation and hash-bound H mirror.

        Boundary state includes scheduler/validation updates that occur after
        the last optimizer step, so merely mirroring the prior cadence payload
        is not sufficient evidence of an exact epoch or fold commit.
        """

        if str(reason) not in {"EPOCH_END", "FOLD_END", "TERMINAL"}:
            raise ValueError(f"INVALID_BOUNDARY_REASON:{reason}")
        staged = self.record_step(
            payload_factory=payload_factory,
            optimizer_step=optimizer_step,
            epoch=epoch,
            epoch_batch_cursor=epoch_batch_cursor,
            sampler_cursor=sampler_cursor,
            loss_sum=loss_sum,
            loss_count=loss_count,
            rng_digest=rng_digest,
            force_full=True,
        )
        mirror = self.mirror_to_h(reason=str(reason))
        return {
            "reason": str(reason),
            "checkpoint_committed": bool(staged["checkpoint_committed"]),
            "generation": staged["generation"],
            "journal_record": staged["journal_record"],
            "mirror": mirror,
        }

    def finalize_clean_stop(self, *, reason: str) -> dict[str, Any]:
        """Mirror the newest staged generation to H on a controlled stop.

        Used from the interrupt handler, where the live loop variables needed to
        rebuild a payload are out of scope. The journal already records every
        step, so this makes the newest durable payload authoritative and reports
        how many steps the next start must replay.
        """

        if not self.layout.current_path.is_file():
            # Nothing staged yet this fold: the pre-epoch snapshot on H is the
            # resume source and there is nothing to mirror.
            return {
                "reason": str(reason),
                "status": "NO_STAGED_CHECKPOINT",
                "exit_code": RESUMABLE_EXIT_CODE,
            }
        mirror = self.mirror_to_h(reason="CLEAN_STOP")
        tail = read_journal_tail(self.layout.journal_path) or {}
        committed_step = 0
        for record in reversed(_read_journal_records(self.layout.journal_path)):
            if record.get("checkpoint_committed"):
                committed_step = int(record.get("optimizer_step", 0))
                break
        return {
            "reason": str(reason),
            "status": "MIRRORED",
            "mirror": mirror,
            "last_journaled_step": int(tail.get("optimizer_step", 0)),
            "last_committed_step": committed_step,
            "replay_steps_on_resume": max(
                0, int(tail.get("optimizer_step", 0)) - committed_step
            ),
            "exit_code": RESUMABLE_EXIT_CODE,
        }

    def plan_recovery(self, *, max_replay_steps: int | None = None) -> RecoveryPlan:
        """Recovery plan for this fold, bounded by this stager's own cadence."""

        return plan_recovery(
            self.layout,
            load_fn=self._load_fn,
            max_replay_steps=(
                self.full_cadence - 1 if max_replay_steps is None else int(max_replay_steps)
            ),
            require_exact=True,
        )

    # -- mirror ---------------------------------------------------------------

    def mirror_to_h(
        self, *, reason: str, prune_previous: bool = False
    ) -> dict[str, Any]:
        """Copy the live generation to H, verify its hash, then optionally prune."""

        if str(reason) not in MIRROR_REASONS:
            raise ValueError(f"INVALID_MIRROR_REASON:{reason}")
        source = self.layout.current_path
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError("NO_CHECKPOINT_TO_MIRROR")
        source_digest = _sha256_file(source)
        source_size = source.stat().st_size
        if self._last_payload_sha256 and source_digest != self._last_payload_sha256:
            raise RuntimeError("MIRROR_SOURCE_HASH_MISMATCH")
        self.layout.mirror_dir.mkdir(parents=True, exist_ok=True)
        target = self.layout.mirror_dir / CURRENT_BASENAME
        temp_target = self.layout.mirror_dir / f"{CURRENT_BASENAME}.{self._generation}.tmp"
        try:
            self._copy_fn(source, temp_target)
            _fsync_path(temp_target)
            if _sha256_file(temp_target) != source_digest:
                raise RuntimeError("MIRROR_COMMIT_HASH_MISMATCH")
            if temp_target.stat().st_size != source_size:
                raise RuntimeError("MIRROR_COMMIT_SIZE_MISMATCH")
            _atomic_replace_with_retry(temp_target, target)
            _fsync_path(target)
        finally:
            if temp_target.exists():
                temp_target.unlink(missing_ok=True)
        pruned = False
        if prune_previous and self.layout.previous_path.exists():
            # Safe only now: the live generation is hash-verified on H.
            self.layout.previous_path.unlink(missing_ok=True)
            pruned = True
        exact_records = [
            record
            for record in _read_journal_records(
                self.layout.journal_path,
                require_chain=True,
            )
            if record.get("checkpoint_committed")
            and int(record.get("generation", -1)) == self._generation
            and record.get("payload_sha256") == source_digest
            and int(record.get("payload_size_bytes", -1)) == source_size
        ]
        if len(exact_records) > 1:
            raise RuntimeError("MIRROR_JOURNAL_BINDING_AMBIGUOUS")
        exact_record = exact_records[0] if exact_records else None
        mirror_state = {
            "schema_version": CHECKPOINT_STAGING_SCHEMA_VERSION,
            "artifact": "method_x_checkpoint_mirror_state",
            "binding_mode": (
                EXACT_BINDING_MODE if exact_record is not None else "LEGACY_COMPAT"
            ),
            "run_id": self.layout.run_id,
            "backbone": self.layout.backbone,
            "fold": self.layout.fold,
            "reason": str(reason),
            "generation": self._generation,
            "payload_sha256": source_digest,
            "payload_size_bytes": source_size,
            "optimizer_step": (
                int(exact_record["optimizer_step"]) if exact_record is not None else None
            ),
            "journal_record_sha256": (
                str(exact_record["record_sha256"]) if exact_record is not None else None
            ),
            "relative_path": str(target.relative_to(self.policy.authoritative_project_root))
            if target.is_relative_to(self.policy.authoritative_project_root)
            else str(target),
            "previous_pruned": pruned,
            "committed_at_utc": _utc_now(),
        }
        mirror_state["integrity_sha256"] = _canonical_sha256(mirror_state)
        _atomic_write_json_fsync(self.layout.mirror_state_path, mirror_state)
        return {
            "schema_version": CHECKPOINT_STAGING_SCHEMA_VERSION,
            "status": "COMMITTED",
            **mirror_state,
            "mirror_path": str(target),
            "mirror_state_path": str(self.layout.mirror_state_path),
        }

    # -- best state -----------------------------------------------------------

    def record_best_state(
        self, state: Mapping[str, Any], *, metric: float
    ) -> dict[str, Any]:
        """Write the best weights to H only when validation actually improves."""

        value = float(metric)
        if self._best_metric is not None and value <= self._best_metric:
            return {"written": False, "metric": value, "best_metric": self._best_metric}
        self.layout.best_state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.layout.best_state_path.with_suffix(
            self.layout.best_state_path.suffix + ".tmp"
        )
        try:
            self._save_fn(dict(state), temp_path)
            _fsync_path(temp_path)
            digest = _sha256_file(temp_path)
            _atomic_replace_with_retry(temp_path, self.layout.best_state_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        sidecar = {
            "schema_version": CHECKPOINT_STAGING_SCHEMA_VERSION,
            "artifact": "method_x_best_state_provenance",
            "run_id": self.layout.run_id,
            "backbone": self.layout.backbone,
            "fold": self.layout.fold,
            "sha256": digest,
            "byte_size": self.layout.best_state_path.stat().st_size,
            "metric": value,
            "written_at_utc": _utc_now(),
        }
        sidecar["integrity_sha256"] = _canonical_sha256(sidecar)
        _atomic_write_json_fsync(self.layout.best_state_sidecar_path, sidecar)
        self._best_metric = value
        return {
            "written": True,
            "metric": value,
            "sha256": digest,
            "path": str(self.layout.best_state_path),
            "sidecar_path": str(self.layout.best_state_sidecar_path),
        }

    def load_best_state(self) -> Mapping[str, Any] | None:
        """Load the hash-locked H-side best-state bundle for staged resume."""

        sidecar_path = self.layout.best_state_sidecar_path
        state_path = self.layout.best_state_path
        if not sidecar_path.is_file() and not state_path.is_file():
            return None
        if sidecar_path.is_symlink() or state_path.is_symlink():
            raise RuntimeError("BEST_STATE_REPARSE_REJECTED")
        if not sidecar_path.is_file() or not state_path.is_file():
            raise RuntimeError("BEST_STATE_ARTIFACT_INCOMPLETE")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("BEST_STATE_SIDECAR_INVALID") from exc
        if (
            not isinstance(sidecar, dict)
            or sidecar.get("integrity_sha256")
            != _canonical_sha256(sidecar, exclude=("integrity_sha256",))
            or sidecar.get("run_id") != self.layout.run_id
            or sidecar.get("backbone") != self.layout.backbone
            or int(sidecar.get("fold", -1)) != self.layout.fold
            or not _is_sha256(sidecar.get("sha256"))
            or _sha256_file(state_path) != sidecar.get("sha256")
            or state_path.stat().st_size != int(sidecar.get("byte_size", -1))
        ):
            raise RuntimeError("BEST_STATE_PROVENANCE_MISMATCH")
        payload = self._load_fn(state_path)
        if not isinstance(payload, Mapping):
            raise RuntimeError("BEST_STATE_PAYLOAD_INVALID")
        self._best_metric = float(sidecar["metric"])
        return payload


# --- Recovery -----------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryPlan:
    """What to load and how far to replay after an interruption.

    ``replay_steps`` is bounded by the cadence: the payload is a real optimizer
    state, so recovery re-runs only the steps journaled after it.
    """

    source: str
    payload_path: Path | None
    resume_from_step: int
    replay_steps: int
    target_step: int
    expected_rng_digest: str | None = None
    expected_loss_sum: float | None = None
    expected_loss_count: int | None = None
    generation: int | None = None
    payload_sha256: str | None = None
    payload_size_bytes: int | None = None
    journal_record_sha256: str | None = None

    def verify_replay(
        self, *, rng_digest: str, loss_sum: float, loss_count: int
    ) -> None:
        """Fail closed unless the replayed state matches the journal exactly."""

        if self.expected_rng_digest is not None and rng_digest != self.expected_rng_digest:
            raise RuntimeError("RNG_DIGEST_MISMATCH")
        if self.expected_loss_count is not None and int(loss_count) != int(
            self.expected_loss_count
        ):
            raise RuntimeError("LOSS_ACCUMULATOR_MISMATCH")
        if self.expected_loss_sum is not None and not _close(
            float(loss_sum), float(self.expected_loss_sum)
        ):
            raise RuntimeError("LOSS_ACCUMULATOR_MISMATCH")


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _readable_payload(path: Path, load_fn: Callable[[Path], Any]) -> bool:
    """A payload counts only if it is non-empty and actually deserializes."""

    if not path.is_file():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        load_fn(path)
    except Exception:
        return False
    return True


def _last_committed(records: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for record in reversed(records):
        if record.get("checkpoint_committed"):
            return record
    return None


def _read_journal_records(
    path: Path,
    *,
    require_chain: bool = False,
) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("JOURNAL_READ_FAILED") from exc
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty_indices[-1] if nonempty_indices else -1
    records: list[dict[str, Any]] = []
    previous_sha256 = _ZERO_SHA256
    previous_step = -1
    previous_generation = -1
    for index, line in enumerate(lines):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            record = json.loads(candidate)
        except json.JSONDecodeError as exc:
            if index == last_nonempty:
                break  # one torn trailing line from a kill mid-append
            raise RuntimeError("JOURNAL_CORRUPT_NONTERMINAL_LINE") from exc
        if not isinstance(record, dict) or "optimizer_step" not in record:
            raise RuntimeError("JOURNAL_RECORD_INVALID")
        has_chain = "record_sha256" in record or "previous_record_sha256" in record
        if require_chain and not has_chain:
            raise RuntimeError("JOURNAL_HASH_CHAIN_MISSING")
        if has_chain:
            if (
                record.get("previous_record_sha256") != previous_sha256
                or not _is_sha256(record.get("record_sha256"))
                or record.get("record_sha256")
                != _canonical_sha256(record, exclude=("record_sha256",))
            ):
                raise RuntimeError("JOURNAL_HASH_CHAIN_INVALID")
            previous_sha256 = str(record["record_sha256"])
        elif records and require_chain:
            raise RuntimeError("JOURNAL_HASH_CHAIN_MISSING")
        try:
            step = int(record["optimizer_step"])
            generation = int(record.get("generation", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("JOURNAL_CURSOR_INVALID") from exc
        if require_chain and (step < previous_step or generation < previous_generation):
            raise RuntimeError("JOURNAL_MONOTONICITY_INVALID")
        previous_step = step
        previous_generation = generation
        records.append(record)
    return records


def _payload_identity_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "staging_run_id",
            "staging_backbone",
            "staging_fold",
            "checkpoint_generation",
            "last_committed_optimizer_step",
            "epoch",
            "epoch_batch_cursor",
            "sampler_cursor",
            "run_contract_digest",
            "scientific_contract_digest",
            "runtime_resume_digest",
            "storage_policy_digest",
        )
    }


def _validate_exact_payload_candidate(
    path: Path,
    *,
    record: Mapping[str, Any],
    layout: StagingLayout,
    load_fn: Callable[[Path], Any],
) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        expected_size = int(record.get("payload_size_bytes", -1))
        if expected_size <= 0 or path.stat().st_size != expected_size:
            return False
        if _sha256_file(path) != record.get("payload_sha256"):
            return False
        payload = load_fn(path)
    except Exception:
        return False
    if not isinstance(payload, Mapping):
        return False
    required_digests = (
        "run_contract_digest",
        "scientific_contract_digest",
        "runtime_resume_digest",
        "storage_policy_digest",
    )
    if any(not _is_sha256(payload.get(key)) for key in required_digests):
        return False
    if (
        payload.get("checkpoint_binding_mode") != EXACT_BINDING_MODE
        or payload.get("staging_run_id") != layout.run_id
        or payload.get("staging_backbone") != layout.backbone
        or int(payload.get("staging_fold", -1)) != layout.fold
        or int(payload.get("checkpoint_generation", -1))
        != int(record.get("generation", -2))
        or int(payload.get("last_committed_optimizer_step", -1))
        != int(record.get("optimizer_step", -2))
        or record.get("run_id") != layout.run_id
        or record.get("backbone") != layout.backbone
        or int(record.get("fold", -1)) != layout.fold
    ):
        return False
    for key in required_digests[1:]:
        if record.get(key) != payload.get(key):
            return False
    return payload.get("checkpoint_payload_identity_sha256") == _canonical_sha256(
        _payload_identity_fields(payload)
    )


def _load_exact_mirror_binding(
    layout: StagingLayout,
    records_by_hash: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Path] | None:
    state_path = layout.mirror_state_path
    if state_path.is_symlink() or not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    recorded_integrity = state.get("integrity_sha256")
    if (
        state.get("binding_mode") != EXACT_BINDING_MODE
        or recorded_integrity != _canonical_sha256(state, exclude=("integrity_sha256",))
        or state.get("run_id") != layout.run_id
        or state.get("backbone") != layout.backbone
        or int(state.get("fold", -1)) != layout.fold
    ):
        return None
    record = records_by_hash.get(str(state.get("journal_record_sha256") or ""))
    if record is None:
        return None
    if (
        int(state.get("generation", -1)) != int(record.get("generation", -2))
        or int(state.get("optimizer_step", -1))
        != int(record.get("optimizer_step", -2))
        or state.get("payload_sha256") != record.get("payload_sha256")
        or int(state.get("payload_size_bytes", -1))
        != int(record.get("payload_size_bytes", -2))
    ):
        return None
    return record, layout.mirror_dir / CURRENT_BASENAME


def discard_incomplete_staging_temps(layout: StagingLayout) -> list[str]:
    """Unlink leftover atomic-write temps. They are never a resume source.

    A hard kill (Cursor stop / TerminateProcess) skips ``write_full_checkpoint``'s
    ``finally`` and leaves ``resume_current.pt.{generation}.tmp``. The next
    resume must not treat that file as current, and the next cadence write
    reuses the same temp name.
    """

    removed: list[str] = []
    staging_dir = layout.staging_dir
    if not staging_dir.is_dir():
        return removed
    for path in staging_dir.iterdir():
        if not path.is_file() or _STAGING_TEMP_NAME.fullmatch(path.name) is None:
            continue
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def _plan_exact_recovery(
    layout: StagingLayout,
    *,
    load_fn: Callable[[Path], Any],
    max_replay_steps: int,
) -> RecoveryPlan:
    discard_incomplete_staging_temps(layout)
    records = _read_journal_records(layout.journal_path, require_chain=True)
    if not records:
        return RecoveryPlan(
            source="COLD_START",
            payload_path=None,
            resume_from_step=0,
            replay_steps=0,
            target_step=0,
        )
    latest = records[-1]
    target_step = int(latest["optimizer_step"])
    committed = [
        record
        for record in records
        if record.get("checkpoint_committed")
        and record.get("binding_mode") == EXACT_BINDING_MODE
        and _is_sha256(record.get("record_sha256"))
        and _is_sha256(record.get("payload_sha256"))
        and isinstance(record.get("payload_size_bytes"), int)
    ]
    if not committed:
        raise RuntimeError("NO_VALID_RESUME_SOURCE:NO_EXACT_COMMITTED_CHECKPOINT")
    records_by_hash = {str(record["record_sha256"]): record for record in committed}
    candidates: list[tuple[int, int, str, Path, Mapping[str, Any]]] = []
    for priority, (source, path) in enumerate(
        (("C_CURRENT", layout.current_path), ("C_PREVIOUS", layout.previous_path))
    ):
        for record in reversed(committed):
            resume_step = int(record["optimizer_step"])
            replay_steps = target_step - resume_step
            if replay_steps < 0 or replay_steps > int(max_replay_steps):
                continue
            if _validate_exact_payload_candidate(
                path,
                record=record,
                layout=layout,
                load_fn=load_fn,
            ):
                candidates.append((resume_step, -priority, source, path, record))
                break
    mirror_binding = _load_exact_mirror_binding(layout, records_by_hash)
    if mirror_binding is not None:
        record, path = mirror_binding
        resume_step = int(record["optimizer_step"])
        replay_steps = target_step - resume_step
        if 0 <= replay_steps <= int(max_replay_steps) and _validate_exact_payload_candidate(
            path,
            record=record,
            layout=layout,
            load_fn=load_fn,
        ):
            candidates.append((resume_step, -1, "H_MIRROR", path, record))
    if not candidates:
        raise RuntimeError("NO_VALID_EXACT_RESUME_SOURCE_FOR_LATEST_COMMIT")
    resume_step, _priority, source, path, record = max(candidates, key=lambda row: (row[0], row[1]))
    replay_steps = target_step - resume_step
    return RecoveryPlan(
        source=source,
        payload_path=path,
        resume_from_step=resume_step,
        replay_steps=replay_steps,
        target_step=target_step,
        expected_rng_digest=str(latest.get("rng_digest") or "") or None,
        expected_loss_sum=float(latest.get("loss_sum", 0.0)),
        expected_loss_count=int(latest.get("loss_count", 0)),
        generation=int(record["generation"]),
        payload_sha256=str(record["payload_sha256"]),
        payload_size_bytes=int(record["payload_size_bytes"]),
        journal_record_sha256=str(record["record_sha256"]),
    )


def plan_recovery(
    layout: StagingLayout,
    *,
    load_fn: Callable[[Path], Any],
    max_replay_steps: int = MAX_DETERMINISTIC_REPLAY_STEPS,
    require_exact: bool = False,
) -> RecoveryPlan:
    """Choose a resume source and compute the deterministic replay window.

    Preference order is newest-verifiable-first: the live C generation, then the
    previous C generation, then the H mirror. A finalized fold short-circuits
    entirely so completed work is never restarted.
    """

    if require_exact:
        return _plan_exact_recovery(
            layout,
            load_fn=load_fn,
            max_replay_steps=max_replay_steps,
        )
    records = _read_journal_records(layout.journal_path, require_chain=False)
    if not records:
        return RecoveryPlan(
            source="COLD_START",
            payload_path=None,
            resume_from_step=0,
            replay_steps=0,
            target_step=0,
        )
    latest = records[-1]
    if latest.get("fold_finalized"):
        return RecoveryPlan(
            source="FOLD_FINALIZED",
            payload_path=None,
            resume_from_step=int(latest["optimizer_step"]),
            replay_steps=0,
            target_step=int(latest["optimizer_step"]),
        )
    committed = _last_committed(records)
    if committed is None:
        raise RuntimeError("NO_VALID_RESUME_SOURCE:NO_COMMITTED_CHECKPOINT")

    target_step = int(latest["optimizer_step"])
    # The corruption signal is a journal that kept advancing while full
    # checkpoints stopped landing. Measure that against the newest commit, once:
    # falling back to an older generation legitimately replays more than one
    # cadence, so a per-candidate bound would reject a healthy fallback.
    uncommitted_tail = target_step - int(committed["optimizer_step"])
    if uncommitted_tail > int(max_replay_steps):
        raise RuntimeError(
            f"REPLAY_BOUND_EXCEEDED:{uncommitted_tail}>{int(max_replay_steps)}"
        )
    candidates: list[tuple[str, Path, int]] = [
        ("C_CURRENT", layout.current_path, int(committed["optimizer_step"]))
    ]
    previous_commits = [
        record
        for record in records
        if record.get("checkpoint_committed")
        and int(record["optimizer_step"]) < int(committed["optimizer_step"])
    ]
    if previous_commits:
        candidates.append(
            ("C_PREVIOUS", layout.previous_path, int(previous_commits[-1]["optimizer_step"]))
        )
    candidates.append(
        ("H_MIRROR", layout.mirror_dir / CURRENT_BASENAME, int(committed["optimizer_step"]))
    )

    for source, path, resume_step in candidates:
        if not _readable_payload(path, load_fn):
            continue
        replay = target_step - resume_step
        return RecoveryPlan(
            source=source,
            payload_path=path,
            resume_from_step=resume_step,
            replay_steps=replay,
            target_step=target_step,
            expected_rng_digest=str(latest.get("rng_digest") or "") or None,
            expected_loss_sum=float(latest.get("loss_sum", 0.0)),
            expected_loss_count=int(latest.get("loss_count", 0)),
        )
    raise RuntimeError("NO_VALID_RESUME_SOURCE:ALL_CANDIDATES_UNREADABLE")


# --- Trainer-facing helpers ---------------------------------------------------


def add_staging_arguments(parser: Any) -> Any:
    """Register the Task 4 runtime flags on an argparse parser.

    Defaults are inert: an empty staging root keeps the existing behaviour, so
    the change is opt-in per invocation rather than a silent switch.
    """

    parser.add_argument(
        "--checkpoint_staging_root",
        default="",
        help=(
            "Allowlisted NVMe staging root for active resume payloads. Empty keeps "
            "the legacy per-step write to the authoritative volume."
        ),
    )
    parser.add_argument(
        "--checkpoint_full_cadence",
        type=int,
        default=32,
        help="Optimizer steps between full staged checkpoints. Every step is journaled regardless.",
    )
    parser.add_argument(
        "--checkpoint_h_mirror",
        choices=sorted(_ALLOWED_H_MIRROR_POLICIES),
        default="epoch",
        help="When a staged payload is mirrored to the authoritative volume.",
    )
    parser.add_argument(
        "--resume_replay_policy",
        choices=sorted(_ALLOWED_REPLAY_POLICIES),
        default="deterministic",
        help="Recovery policy for uncommitted steps. Only deterministic replay is permitted.",
    )
    return parser


def resolve_mirror_policy(policy: str) -> dict[str, bool]:
    """Translate the mirror flag into per-boundary decisions.

    Fold end always mirrors: a fold boundary is a scientific commit point, so its
    payload must exist on the authoritative volume no matter how the flag is set.
    """

    value = str(policy)
    if value not in _ALLOWED_H_MIRROR_POLICIES:
        raise ValueError(f"INVALID_H_MIRROR_POLICY:{policy}")
    return {"epoch": value in {"epoch", "epoch_and_fold"}, "fold": True}


def build_stager_from_args(
    args: Any,
    *,
    run_id: str,
    backbone: str,
    fold: int,
    h_output_dir: Path,
    policy_path: Path | None = None,
) -> CheckpointStager | None:
    """Construct a stager, or return None when staging is not requested."""

    from src.training.storage_locality import load_storage_policy

    staging_root = str(getattr(args, "checkpoint_staging_root", "") or "").strip()
    if not staging_root:
        return None
    cadence = int(getattr(args, "checkpoint_full_cadence", 32))
    if cadence < 1:
        raise ValueError(f"INVALID_CHECKPOINT_FULL_CADENCE:{cadence}")
    replay = str(getattr(args, "resume_replay_policy", "deterministic"))
    if replay not in _ALLOWED_REPLAY_POLICIES:
        raise ValueError(f"INVALID_RESUME_REPLAY_POLICY:{replay}")
    resolve_mirror_policy(str(getattr(args, "checkpoint_h_mirror", "epoch")))
    resolved_policy_path = Path(
        policy_path
        or getattr(args, "storage_policy", None)
        or DEFAULT_STORAGE_POLICY_PATH
    )
    policy = load_storage_policy(resolved_policy_path)
    h_dir = Path(h_output_dir)
    resolved_run_id = resolve_staging_run_id(
        live_run_id=run_id,
        journal_path=fold_journal_path(h_dir, fold),
    )
    layout = StagingLayout.build(
        policy=policy,
        run_id=resolved_run_id,
        backbone=backbone,
        fold=fold,
        h_output_dir=h_dir,
        staging_root_override=Path(staging_root),
    )
    stager = CheckpointStager(policy=policy, layout=layout, full_cadence=cadence)
    # Fail fast at construction rather than at the first cadence hit mid-epoch.
    stager._assert_staging_root_allowed()
    return stager
