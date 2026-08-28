"""Small real-process worker for BrainIAC transaction kill/resume tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research import brainiac_preprocessing_v2 as v2
from src.research.brainiac_preprocessing_v2_schema import (
    build_aggregate,
    canonical_json,
    sha256_file,
)
from scripts.research.prepare_brainiac_raw_t1_v2 import (
    _StopFlag,
    _install_signal_handlers,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


BINDINGS = {
    "policy_sha256": _sha("policy"),
    "config_sha256": _sha("config"),
    "pipeline_contract_sha256": _sha("pipeline"),
    "dependency_closure_sha256": _sha("closure"),
    "membership_sha256": _sha("membership"),
    "cache_sha256": _sha("cache"),
    "catalog_sha256": _sha("catalog"),
    "runtime_lock_sha256": _sha("runtime"),
    "vendor_tree_sha256": _sha("vendor"),
    "template_lock_sha256": _sha("template-lock"),
    "official_transform_source_sha256": _sha("transform-source"),
    "qc_threshold_manifest_sha256": _sha("threshold"),
}


def _pipeline(marker: Path | None, sleep_after_marker: bool):
    def run(item, scratch):
        sample = str(item["sample_hash"])
        output = scratch / "output.nii.gz"
        transform = scratch / "transform.tfm"
        output.write_bytes(("fixture-output:" + sample).encode("ascii"))
        transform.write_bytes(("fixture-transform:" + sample).encode("ascii"))
        if marker is not None and not marker.exists():
            marker.write_text(sample, encoding="utf-8")
            if sleep_after_marker:
                # Long enough for the signal thread/parent kill to act, but
                # bounded so a failed handler cannot stall the suite for a minute.
                time.sleep(2)
        identity_affine = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        identity_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        return {
            "output_path": output,
            "transform_path": transform,
            "source_sha256": _sha("source:" + sample),
            "source_size": 100,
            "affine_before": identity_affine,
            "affine_after": identity_affine,
            "spacing_before": [1.0, 1.0, 1.0],
            "spacing_after": [1.0, 1.0, 1.0],
            "origin_before": [0.0, 0.0, 0.0],
            "origin_after": [0.0, 0.0, 0.0],
            "direction_before": identity_direction,
            "direction_after": identity_direction,
            "com_coordinate_frame": "TEMPLATE_PHYSICAL_MM_RELATIVE",
            "rigid_determinant": 1.0,
            "orthogonality_error": 0.0,
            "rotation_deg": 0.0,
            "translation_norm_mm": 0.0,
            "foreground_fraction": 0.2,
            "finite": True,
            "normalized_nonzero_mean": 0.0,
            "normalized_nonzero_std": 1.0,
            "center_of_mass_before": [1.0, 0.0, 0.0],
            "center_of_mass_after": [0.5, 0.0, 0.0],
            "registration_similarity_before": -1.0,
            "registration_similarity_after": -1.1,
        }

    return run


def _start_signal_trigger(marker: Path | None, signal_name: str | None) -> None:
    """Raise a catchable process signal after a sample enters its transaction."""

    if marker is None or not signal_name:
        return
    signum = getattr(signal, signal_name, None)
    if signum is None:
        raise RuntimeError(f"UNSUPPORTED_FIXTURE_SIGNAL:{signal_name}")

    def trigger() -> None:
        deadline = time.monotonic() + 30.0
        while not marker.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        if marker.is_file():
            signal.raise_signal(signum)

    threading.Thread(target=trigger, name="brainiac-signal-fixture", daemon=True).start()


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    marker = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    sleep_after_marker = os.environ.get("BRAINIAC_FIXTURE_SLEEP_AFTER_MARKER") == "1"
    signal_name = os.environ.get("BRAINIAC_FIXTURE_SIGNAL") or None
    parallel = os.environ.get("BRAINIAC_FIXTURE_PARALLEL") == "1"
    # Writer discovery is tested independently; this fixture isolates recovery.
    v2.assert_no_cnn_writer = lambda: None
    stop = _StopFlag()
    _install_signal_handlers(stop)
    _start_signal_trigger(marker, signal_name)
    items = [
        {"sample_hash": f"s_{index:016x}", "t1_file_hash": f"t1-{index}"}
        for index in range(4)
    ]
    run_batch = v2.run_orchestrated_batch if parallel else v2.run_batch
    kwargs = {
        "pipeline_factory": lambda: _pipeline(marker, sleep_after_marker),
        "bindings": BINDINGS,
        "stop_requested": lambda: stop.requested,
    }
    if parallel:
        kwargs.update({"cpu_workers": 2, "gpu_workers": 1})
    report = run_batch(items, root, **kwargs)
    if report["status"] == "PAUSED_RESUMABLE":
        return 75
    if report["status"] != "COMPLETED":
        return 3
    receipts = [
        v2.load_committed_receipt(
            item["sample_hash"], root, expected_bindings=BINDINGS
        )
        for item in items
    ]
    aggregate = build_aggregate(
        receipts=receipts,
        expected_sample_hashes=[item["sample_hash"] for item in items],
        artifact_root=root,
    )
    (root / "aggregate.json").write_bytes(canonical_json(aggregate))
    # Rehash once more so the child exits only after durable, readable evidence.
    if len(sha256_file(root / "aggregate.json")) != 64:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
