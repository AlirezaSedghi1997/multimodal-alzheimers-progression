"""Label-blind rigid-optimizer diagnostic for BrainIAC preprocessing.

The tool persists no raw path, label, PET, tabular value, or subject identifier.
It exists to make a QC-triggered optimizer repair reproducible and hash-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import brainiac_preprocessing_v2 as v2
import scripts.research.prepare_brainiac_raw_t1_v2 as prep


DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "method_x_registry"
    / "brainiac_r1"
    / "brainiac_rigid_optimizer_diagnostic.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _method_base(fixed: Any, moving: Any, initial: Any) -> Any:
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetNumberOfWorkUnits(1)
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    return method


def _evaluate(fixed: Any, moving: Any, transform: Any) -> dict[str, Any]:
    canonical = prep._canonicalize_rigid_transform(transform, sitk)
    registered = sitk.Resample(
        moving, fixed, canonical, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    template_com = prep._foreground_com_physical(fixed)
    relative_com = prep._relative_com(
        prep._foreground_com_physical(registered), template_com
    )
    return {
        "registration_similarity": prep._metric_value(fixed, registered, sitk),
        "com_norm_mm": float(np.linalg.norm(np.asarray(relative_com))),
        "parameters": [float(value) for value in canonical.GetParameters()],
        "transform_qc": prep._transform_qc(canonical, sitk, fixed),
    }


def _run_variant(
    fixed: Any,
    moving: Any,
    initial: Any,
    configure: Callable[[Any], None],
) -> dict[str, Any]:
    method = _method_base(fixed, moving, initial)
    configure(method)
    method.SetOptimizerScalesFromPhysicalShift()
    started = time.monotonic()
    transform = method.Execute(fixed, moving)
    result = _evaluate(fixed, moving, transform)
    result.update(
        {
            "elapsed_seconds": time.monotonic() - started,
            "optimizer_metric": float(method.GetMetricValue()),
            "stop_condition": method.GetOptimizerStopConditionDescription(),
        }
    )
    return result


def run(sample_hash: str, output: Path, *, profile: str = "comprehensive") -> dict[str, Any]:
    if not sample_hash.startswith("s_") or len(sample_hash) != 18:
        raise RuntimeError("BRAINIAC_RIGID_DIAGNOSTIC_SAMPLE_HASH_INVALID")
    membership = v2.load_locked_membership()
    item = next(
        (row for row in membership if row["sample_hash"] == sample_hash), None
    )
    if item is None:
        raise RuntimeError("BRAINIAC_RIGID_DIAGNOSTIC_SAMPLE_NOT_IN_MEMBERSHIP")
    source_path = v2.resolve_sources([item], index=v2.build_raw_index())[sample_hash]
    source = sitk.ReadImage(str(source_path), sitk.sitkFloat32)
    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
    n4_filter.SetNumberOfThreads(8)
    n4_filter.SetNumberOfWorkUnits(8)
    n4_started = time.monotonic()
    moving = n4_filter.Execute(source)
    n4_seconds = time.monotonic() - n4_started
    fixed = prep._load_fixed_template()
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    baseline = _evaluate(fixed, moving, initial)

    variants: dict[str, dict[str, Any]] = {}
    if profile == "comprehensive":
        for learning_rate in (0.02, 0.05, 0.1):
            variants[f"gradient_descent_lr_{learning_rate}"] = _run_variant(
                fixed,
                moving,
                initial,
                lambda method, lr=learning_rate: method.SetOptimizerAsGradientDescent(
                    learningRate=lr,
                    numberOfIterations=100,
                    convergenceMinimumValue=1e-6,
                    convergenceWindowSize=10,
                ),
            )
        variants["regular_step_lr_0.25"] = _run_variant(
            fixed,
            moving,
            initial,
            lambda method: method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=0.25,
                minStep=1e-4,
                numberOfIterations=150,
                relaxationFactor=0.5,
                gradientMagnitudeTolerance=1e-8,
            ),
        )
        variants["gradient_descent_line_search_lr_0.5"] = _run_variant(
            fixed,
            moving,
            initial,
            lambda method: method.SetOptimizerAsGradientDescentLineSearch(
                learningRate=0.5,
                numberOfIterations=60,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10,
                lineSearchLowerLimit=0.0,
                lineSearchUpperLimit=2.0,
                lineSearchEpsilon=0.01,
                lineSearchMaximumIterations=10,
            ),
        )
    elif profile == "fast":
        for learning_rate in (0.001, 0.01, 0.1):
            variants[f"gradient_descent_never_lr_{learning_rate}"] = _run_variant(
                fixed,
                moving,
                initial,
                lambda method, lr=learning_rate: method.SetOptimizerAsGradientDescent(
                    learningRate=lr,
                    numberOfIterations=50,
                    convergenceMinimumValue=1e-6,
                    convergenceWindowSize=10,
                    estimateLearningRate=sitk.ImageRegistrationMethod.Never,
                ),
            )
        for iterations in (20, 40):
            variants[f"regular_step_lr_0.25_iter_{iterations}"] = _run_variant(
                fixed,
                moving,
                initial,
                lambda method, n=iterations: method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=0.25,
                    minStep=1e-4,
                    numberOfIterations=n,
                    relaxationFactor=0.5,
                    gradientMagnitudeTolerance=1e-8,
                ),
            )
        for iterations in (10, 20):
            variants[f"line_search_lr_0.5_iter_{iterations}"] = _run_variant(
                fixed,
                moving,
                initial,
                lambda method, n=iterations: method.SetOptimizerAsGradientDescentLineSearch(
                    learningRate=0.5,
                    numberOfIterations=n,
                    convergenceMinimumValue=1e-6,
                    convergenceWindowSize=10,
                    lineSearchLowerLimit=0.0,
                    lineSearchUpperLimit=2.0,
                    lineSearchEpsilon=0.02,
                    lineSearchMaximumIterations=5,
                ),
            )
    elif profile == "repeat_line_search":
        for repeat in range(3):
            variants[f"line_search_lr_0.5_iter_10_repeat_{repeat + 1}"] = _run_variant(
                fixed,
                moving,
                initial,
                lambda method: method.SetOptimizerAsGradientDescentLineSearch(
                    learningRate=0.5,
                    numberOfIterations=10,
                    convergenceMinimumValue=1e-6,
                    convergenceWindowSize=10,
                    lineSearchLowerLimit=0.0,
                    lineSearchUpperLimit=2.0,
                    lineSearchEpsilon=0.02,
                    lineSearchMaximumIterations=5,
                ),
            )
    else:
        raise RuntimeError("BRAINIAC_RIGID_DIAGNOSTIC_PROFILE_INVALID")
    for result in variants.values():
        result["both_com_and_similarity_worsened"] = bool(
            result["com_norm_mm"] > baseline["com_norm_mm"]
            and result["registration_similarity"]
            > baseline["registration_similarity"]
        )
        result["finite"] = bool(
            math.isfinite(result["com_norm_mm"])
            and math.isfinite(result["registration_similarity"])
        )

    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "BRAINIAC_RIGID_OPTIMIZER_LABEL_BLIND_DIAGNOSTIC",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "sample_hash": sample_hash,
        "selection_role": "QC_FAILURE_REPAIR_LABEL_BLIND",
        "diagnostic_profile": profile,
        "n4_seconds": n4_seconds,
        "baseline_centered_initial": baseline,
        "variants": variants,
        "bindings": {
            "policy_sha256": v2.policy_sha256(),
            "membership_sha256": prep.canonical_sha(membership),
            "diagnostic_source_sha256": _sha256(Path(__file__)),
        },
        "labels_read": False,
        "pet_read": False,
        "raw_path_persisted": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    body["integrity_sha256"] = prep.canonical_sha(body)
    prep._atomic_write_json(output, body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-hash", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--profile",
        choices=("comprehensive", "fast", "repeat_line_search"),
        default="comprehensive",
    )
    args = parser.parse_args()
    try:
        result = run(args.sample_hash, args.output, profile=args.profile)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": type(exc).__name__}))
        return 3
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact": str(args.output),
                "integrity_sha256": result["integrity_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
