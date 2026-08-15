"""Build Task-4 Method Y pretraining verification without starting training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.method_y_revival.atomic_io import atomic_write_bytes, atomic_write_json
from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.endpoint_builder import (
    build_amyloid_endpoint_rows,
    build_paired_at_endpoint_rows,
    compatibility_endpoint_config,
    load_first_amyloid_scans,
    sha256_file,
    treatment_active_value,
)
from src.method_y_revival.pretraining_contracts import (
    PretrainingContractError,
    assert_public_contract_privacy,
    build_private_split_artifacts,
    commit_pretraining_state,
    external_intake_contracts,
    predictor_temporality_contract,
    scanner_contract_from_aggregate,
    validate_locked_endpoint,
    verify_pacc_secondary_endpoint,
)
from src.method_y_revival.process_guard import WriterLock
from src.method_y_revival.review_receipt import TASK4_IMPLEMENTATION_RELATIVE_PATHS


def _render_markdown(
    endpoint: dict,
    temporality: dict,
    split: dict,
    scanner: dict,
    external: dict,
) -> str:
    lines = [
        "# Method Y v2 pretraining contracts: Task 4",
        "",
        "No training, HPO, outer evaluation, image analysis, or external outcome access was run.",
        "",
        "## Endpoint verification",
        "",
    ]
    for branch in ("amyloid", "tau", "paired_at"):
        item = endpoint["primary_endpoints"][branch]
        lines.append(
            f"- `{branch}`: `{item['status']}`; {item['row_count']} people, "
            f"{item['event_count']} events; sustained future-only CDR endpoint."
        )
    pacc = endpoint["secondary_longitudinal_pacc"]
    lines.extend(
        [
            f"- `longitudinal_pacc`: `{pacc['status']}`; {pacc['subject_count']} people, "
            f"{pacc['row_count']} post-PET measurements.",
            "",
            "## Predictor temporality",
            "",
            f"- status: `{temporality['status']}`",
            "- `baseline_pacc_change`, CDR/CDRSB/FAQ/diagnosis/outcome/follow-up fields are forbidden.",
            "- Baseline cognition with unknown timing remains blocked; unknown timing is never imputed.",
            "",
            "## Split roles",
            "",
        ]
    )
    for branch in (
        "amyloid_source_regional",
        "tau_source_regional",
        "paired_source_regional",
    ):
        item = split["branches"][branch]
        lines.append(
            f"- `{branch}`: `{item['status']}`; role `{item['analysis_role']}`; "
            f"{item['subject_count']} people, {item['event_count']} events; 5 outer / 3 inner."
        )
    transport = split["branches"]["amyloid_learn_transport"]
    lines.extend(
        [
            f"- `amyloid_learn_transport`: `{transport['analysis_role']}`; "
            f"{transport['subject_count']} people, {transport['event_count']} events; no HPO.",
            "",
            "## Scanner and external gates",
            "",
            f"- Amyloid metadata feasibility: `{scanner['branches']['amyloid']['scanner_feasibility_status']}`.",
            f"- Tau metadata feasibility: `{scanner['branches']['tau']['scanner_feasibility_status']}`.",
            "- Post-model scanner sensitivity is not run until locked OOF predictions exist.",
            f"- External transport: `{external['status']}`; outcomes remain embargoed.",
            "- Internal source-regional development is not blocked by absent external access.",
            "",
            "## Claim boundary",
            "",
            "- Positive, image, scanner-generalization and external claims remain closed.",
            "- Layered agent multi-pass verification is the completion mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_reconstructed_endpoints(config: dict):
    """Independently reconstruct logical rows without rewriting old artifacts."""

    import pandas as pd

    from scripts.build_method_y_tau_progression import (
        build_label_rows,
        load_first_tau_scans,
        read_csv_rows,
    )

    raw_root = Path(config["paths"]["raw_root"])
    source = {
        name: raw_root / config["clinical_sources"][name]
        for name in ("amyloid_suvr", "tau_suvr", "cdr", "pacc", "subject_info")
    }
    source_rows = {
        "amyloid": read_csv_rows(source["amyloid_suvr"]),
        "cdr": read_csv_rows(source["cdr"]),
        "pacc": read_csv_rows(source["pacc"]),
        "subject": read_csv_rows(source["subject_info"]),
    }
    first_amyloid, _ = load_first_amyloid_scans(source_rows["amyloid"])
    amyloid_rows, _ = build_amyloid_endpoint_rows(
        first_amyloid,
        source_rows["subject"],
        source_rows["cdr"],
        source_rows["pacc"],
        config,
    )
    compatibility = compatibility_endpoint_config(config)
    first_tau, _ = load_first_tau_scans(source["tau_suvr"], compatibility)
    tau_rows, _, _ = build_label_rows(
        first_tau,
        source_rows["subject"],
        source_rows["cdr"],
        source_rows["pacc"],
        compatibility,
    )
    for row in tau_rows:
        row["treatment_active"] = treatment_active_value(row.get("treatment_arm"))
    paired_rows, _, _ = build_paired_at_endpoint_rows(
        first_amyloid,
        first_tau,
        source_rows["subject"],
        source_rows["cdr"],
        source_rows["pacc"],
        config,
    )
    amyloid_frame = pd.DataFrame(amyloid_rows)
    return {
        "amyloid": amyloid_frame,
        "tau": pd.DataFrame(tau_rows),
        "paired_at": pd.DataFrame(paired_rows),
        "longitudinal_pacc": _reconstruct_longitudinal_pacc(
            config=config,
            first_amyloid=first_amyloid,
            locked_amyloid_frame=amyloid_frame,
        ),
    }


def _reconstruct_longitudinal_pacc(
    *,
    config: dict,
    first_amyloid: Mapping[tuple[str, str], Mapping[str, object]],
    locked_amyloid_frame,
):
    """Rebuild every secondary PACC row from immutable raw sources in memory."""

    import numpy as np
    import pandas as pd

    from scripts.build_method_y_tau_progression import patient_hash_for_key

    rules = config.get("pretraining_contracts_v2", {})
    if not (
        rules.get("pacc_exact_date_rule")
        == "MEDIAN_EXACT_QSDTC_DAYS_T0_ACROSS_PACC_COMPONENTS"
        and sorted(str(value).upper() for value in rules.get("pacc_component_codes", []))
        == ["DIGIT", "FCSRT96", "LMIIA", "MMSE"]
        and float(rules.get("pacc_post_pet_baseline_window_days", -1)) == 183.0
        and float(rules.get("pacc_followup_window_days", -1)) == 1640.0
        and int(rules.get("pacc_minimum_longitudinal_measurements", -1)) == 2
    ):
        raise PretrainingContractError("PACC_V2_RECONSTRUCTION_RULES_INVALID")
    raw_root = Path(config["paths"]["raw_root"])
    adqs_path = raw_root / "Clinical" / "Derived Data" / "ADQS.csv"
    pacc_path = raw_root / config["clinical_sources"]["pacc"]
    adqs = pd.read_csv(
        adqs_path,
        usecols=[
            "SUBSTUDY",
            "BID",
            "VISITCD",
            "QSTESTCD",
            "QSDTC_DAYS_T0",
        ],
    )
    component_codes = {"DIGIT", "FCSRT96", "LMIIA", "MMSE"}
    adqs = adqs.loc[
        adqs["QSTESTCD"].astype(str).str.upper().isin(component_codes)
    ].copy()
    adqs["exact_date_days_t0"] = pd.to_numeric(
        adqs["QSDTC_DAYS_T0"], errors="coerce"
    )
    exact_dates = (
        adqs.groupby(["SUBSTUDY", "BID", "VISITCD"], as_index=False)[
            "exact_date_days_t0"
        ]
        .median()
    )
    pacc = pd.read_csv(
        pacc_path,
        usecols=["SUBSTUDY", "BID", "VISCODE", "PACC.raw"],
    )
    pacc = pacc.merge(
        exact_dates,
        left_on=["SUBSTUDY", "BID", "VISCODE"],
        right_on=["SUBSTUDY", "BID", "VISITCD"],
        how="left",
        validate="one_to_one",
    )
    pacc["patient_hash"] = [
        patient_hash_for_key((str(substudy).strip().upper(), str(bid).strip()))
        for substudy, bid in zip(pacc["SUBSTUDY"], pacc["BID"], strict=True)
    ]
    scan_dates = {
        patient_hash_for_key(key): float(scan["scan_date"])
        for key, scan in first_amyloid.items()
    }
    pacc["scan_date_days_t0"] = pacc["patient_hash"].map(scan_dates)
    pacc["time_days_from_pet"] = (
        pacc["exact_date_days_t0"] - pacc["scan_date_days_t0"]
    )
    pacc["pacc_raw"] = pd.to_numeric(pacc["PACC.raw"], errors="coerce")
    covariates = [
        "patient_hash",
        "cohort",
        "age",
        "sex_binary",
        "education_years",
        "treatment_active",
        "baseline_amyloid_centiloid",
    ]
    missing_covariates = set(covariates).difference(locked_amyloid_frame.columns)
    if missing_covariates:
        raise PretrainingContractError(
            f"PACC_AMYLOID_COVARIATES_MISSING:{','.join(sorted(missing_covariates))}"
        )
    amyloid = locked_amyloid_frame.loc[:, covariates].copy()
    if amyloid["patient_hash"].duplicated().any():
        raise PretrainingContractError("PACC_AMYLOID_MEMBERSHIP_DUPLICATED")
    eligible_members = set(amyloid["patient_hash"].astype(str))
    finite_measurement = pd.to_numeric(pacc["pacc_raw"], errors="coerce")
    finite_date = pd.to_numeric(pacc["exact_date_days_t0"], errors="coerce")
    finite_scan = pd.to_numeric(pacc["scan_date_days_t0"], errors="coerce")
    pacc = pacc.loc[
        pacc["patient_hash"].isin(eligible_members)
        & finite_measurement.notna()
        & np.isfinite(finite_measurement)
        & finite_date.notna()
        & np.isfinite(finite_date)
        & finite_scan.notna()
        & np.isfinite(finite_scan)
    ].copy()
    baseline_members = set(
        pacc.loc[
            (pacc["time_days_from_pet"] >= 0.0)
            & (pacc["time_days_from_pet"] <= 183.0),
            "patient_hash",
        ].astype(str)
    )
    pacc = pacc.loc[
        pacc["patient_hash"].isin(baseline_members)
        & (pacc["time_days_from_pet"] > 0.0)
        & (pacc["time_days_from_pet"] <= 1640.0)
    ].copy()
    measurement_counts = pacc.groupby("patient_hash").size()
    retained = set(measurement_counts[measurement_counts >= 2].index.astype(str))
    pacc = pacc.loc[pacc["patient_hash"].isin(retained)].copy()
    pacc = pacc.loc[
        :, ["patient_hash", "time_days_from_pet", "pacc_raw"]
    ].merge(amyloid, on="patient_hash", how="left", validate="many_to_one")
    pacc["time_years"] = pacc["time_days_from_pet"] / 365.25
    columns = [
        "patient_hash",
        "cohort",
        "time_days_from_pet",
        "time_years",
        "pacc_raw",
        "age",
        "sex_binary",
        "education_years",
        "treatment_active",
        "baseline_amyloid_centiloid",
    ]
    return pacc.loc[:, columns].sort_values(
        ["patient_hash", "time_days_from_pet"], kind="mergesort"
    ).reset_index(drop=True)


def _clinical_source_hashes(config: dict) -> dict[str, str]:
    raw_root = Path(config["paths"]["raw_root"])
    aliases = {
        "amyloid_suvr_csv": "amyloid_suvr",
        "tau_suvr_csv": "tau_suvr",
        "cdr_csv": "cdr",
        "pacc_csv": "pacc",
        "subjinfo_csv": "subject_info",
    }
    return {
        contract_name: sha256_file(
            raw_root / config["clinical_sources"][config_name]
        )
        for contract_name, config_name in aliases.items()
    }


def _pacc_source_hashes(config: dict) -> dict[str, str]:
    raw_root = Path(config["paths"]["raw_root"])
    sources = {
        "adqs": raw_root / "Clinical" / "Derived Data" / "ADQS.csv",
        "amyloid_suvr": raw_root
        / config["clinical_sources"]["amyloid_suvr"],
        "pacc": raw_root / config["clinical_sources"]["pacc"],
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise PretrainingContractError(
            f"PACC_SOURCE_FILES_MISSING:{','.join(sorted(missing))}"
        )
    return {name: sha256_file(path) for name, path in sorted(sources.items())}


def _implementation_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PretrainingContractError("IMPLEMENTATION_COMMIT_UNAVAILABLE") from exc
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise PretrainingContractError("IMPLEMENTATION_COMMIT_INVALID")
    return commit


def _implementation_file_hashes(config: Mapping[str, object]) -> dict[str, str]:
    del config  # The exact locked YAML bytes are included below.
    return {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in TASK4_IMPLEMENTATION_RELATIVE_PATHS
    }


def validate_cli_roots(
    *,
    project_root: Path,
    source_root: Path,
    output_root: Path,
    private_output_root: Path,
) -> None:
    roots = {
        "project": project_root.resolve(),
        "source": source_root.resolve(),
        "output": output_root.resolve(),
        "private": private_output_root.resolve(),
    }
    if any(path.drive.upper() != "H:" for path in roots.values()):
        raise PretrainingContractError("TASK4_ROOTS_MUST_BE_ON_H_DRIVE")
    if not roots["source"].is_dir():
        raise PretrainingContractError("LOCKED_METHOD_Y_SOURCE_ROOT_MISSING")
    try:
        roots["output"].relative_to(roots["project"])
    except ValueError as exc:
        raise PretrainingContractError("OUTPUT_ROOT_OUTSIDE_PROJECT") from exc
    private_boundary = roots["output"] / "_private"
    try:
        roots["private"].relative_to(private_boundary)
    except ValueError as exc:
        raise PretrainingContractError(
            "PRIVATE_OUTPUT_OUTSIDE_OUTPUT_PRIVATE_BOUNDARY"
        ) from exc
    if roots["source"] == roots["output"] or roots["source"].is_relative_to(
        roots["output"]
    ):
        raise PretrainingContractError("SOURCE_ROOT_MUST_HAVE_READ_ONLY_INPUT_ROLE")


def build_all(
    *,
    source_root: Path,
    output_root: Path,
    private_output_root: Path,
    config: dict,
    seed: int,
    expected_pacc_source_hashes: Mapping[str, str],
    expected_pacc_input_contract_sha256: str,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    private_output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_root / "pretraining_validation_state.json",
        {
            "schema_version": "2.0",
            "status": "RUNNING",
            "stage": "VERIFYING_ENDPOINT_SPLIT_SCANNER_EXTERNAL",
            "training_started": False,
            "outer_evaluation_started": False,
            "external_outcomes_accessed": False,
        },
    )
    try:
        source_hashes_before = _clinical_source_hashes(config)
        pacc_source_hashes_before = _pacc_source_hashes(config)
        if dict(
            sorted(pacc_source_hashes_before.items())
        ) != dict(sorted(expected_pacc_source_hashes.items())):
            raise PretrainingContractError("PACC_SOURCE_SNAPSHOT_MISMATCH")
        rebuilt_frames = _build_reconstructed_endpoints(config)
        missing_rebuilds = {
            "amyloid", "tau", "paired_at", "longitudinal_pacc"
        }.difference(rebuilt_frames)
        if missing_rebuilds:
            raise PretrainingContractError(
                "INDEPENDENT_ENDPOINT_REBUILDS_MISSING:"
                + ",".join(sorted(missing_rebuilds))
            )

        endpoint_reports = {}
        endpoint_frames = {}
        for branch in ("amyloid", "tau", "paired_at"):
            report, frame = validate_locked_endpoint(
                source_root,
                branch,
                rebuilt_frame=rebuilt_frames[branch] if rebuilt_frames is not None else None,
            )
            endpoint_reports[branch] = report
            endpoint_frames[branch] = frame
            declared = report["source_hashes"]
            if any(
                source_hashes_before.get(name) != digest for name, digest in declared.items()
            ):
                raise ValueError(f"LOCKED_SOURCE_SNAPSHOT_MISMATCH:{branch}")
        if _clinical_source_hashes(config) != source_hashes_before:
            raise ValueError("CLINICAL_SOURCE_CHANGED_DURING_VERIFICATION")
        if _pacc_source_hashes(config) != pacc_source_hashes_before:
            raise PretrainingContractError("PACC_SOURCE_CHANGED_DURING_VERIFICATION")
        endpoint_payload = {
            "schema_version": "2.0",
            "status": "PASS",
            "primary_endpoints": endpoint_reports,
            "secondary_longitudinal_pacc": verify_pacc_secondary_endpoint(
                source_root,
                expected_source_hashes=expected_pacc_source_hashes,
                expected_input_contract_sha256=expected_pacc_input_contract_sha256,
                rebuilt_frame=rebuilt_frames["longitudinal_pacc"],
            ),
            "training_started": False,
            "private_rows_published": False,
        }

        # PET predictors are governed by Task-3 feature contracts.  This
        # contract is deliberately limited to clinical/cognitive candidates
        # and endpoint fields that must be rejected.
        clinical_candidate_columns = [
            "age",
            "sex_binary",
            "education_years",
            "apoe4_carrier",
            "treatment_active",
            "baseline_pacc_raw",
            "baseline_pacc_change",
            "baseline_fctotal96",
            "baseline_ldeltotal",
            "baseline_digittotal",
            "baseline_mmscore",
            "baseline_cdglobal",
            "baseline_cdrsb",
            "baseline_faq",
            "diagnosis",
            "followup_diagnosis",
            "event",
            "time_days",
            "censoring_rule",
            "followup_cdr_visits",
            "single_visit_cdglobal_ge_0_5",
        ]
        temporality = predictor_temporality_contract(
            clinical_candidate_columns,
            timing_evidence={
                "sex_binary": "TIME_INVARIANT",
                "education_years": "TIME_INVARIANT",
                "apoe4_carrier": "TIME_INVARIANT",
                "treatment_active": "EXACT_NONFUTURE",
            },
        )
        temporality["source_endpoint_contract_sha256"] = {
            branch: report["contract_sha256"]
            for branch, report in endpoint_reports.items()
        }
        temporality.pop("contract_sha256", None)
        temporality["contract_sha256"] = sha256_mapping(temporality)

        split = build_private_split_artifacts(
            endpoint_frames,
            endpoint_reports,
            private_output_root=private_output_root,
            seed=seed,
        )
        scanner = scanner_contract_from_aggregate(
            source_root / "qc" / "scanner_metadata_audit.json"
        )
        external = external_intake_contracts()

        for payload in (endpoint_payload, temporality, split, scanner, external):
            assert_public_contract_privacy(payload)

        artifacts = {
            "endpoint_verification_v2.json": output_root
            / "endpoint_verification_v2.json",
            "predictor_temporality_contract_v2.json": output_root
            / "predictor_temporality_contract_v2.json",
            "split_verification_v2.json": output_root / "split_verification_v2.json",
            "scanner_feasibility_and_sensitivity_contract_v2.json": output_root
            / "scanner_feasibility_and_sensitivity_contract_v2.json",
            "external_intake_contract_v2.json": output_root
            / "external_intake_contract_v2.json",
        }
        for path, payload in (
            (artifacts["endpoint_verification_v2.json"], endpoint_payload),
            (artifacts["predictor_temporality_contract_v2.json"], temporality),
            (artifacts["split_verification_v2.json"], split),
            (artifacts["scanner_feasibility_and_sensitivity_contract_v2.json"], scanner),
            (artifacts["external_intake_contract_v2.json"], external),
        ):
            atomic_write_json(path, payload)

        markdown_path = output_root / "PRETRAINING_CONTRACTS_TASK4.md"
        atomic_write_bytes(
            markdown_path,
            _render_markdown(endpoint_payload, temporality, split, scanner, external).encode(
                "utf-8"
            ),
        )
        artifacts["PRETRAINING_CONTRACTS_TASK4.md"] = markdown_path
        state = commit_pretraining_state(
            output_root,
            artifacts,
            implementation_commit=_implementation_commit(),
            run_contract_context={
                "seed": int(seed),
                "clinical_source_snapshot_sha256": sha256_mapping(
                    dict(sorted(source_hashes_before.items()))
                ),
                "pacc_source_snapshot_sha256": sha256_mapping(
                    dict(sorted(pacc_source_hashes_before.items()))
                ),
                "pacc_legacy_input_contract_sha256": (
                    expected_pacc_input_contract_sha256
                ),
                "implementation_file_sha256": _implementation_file_hashes(config),
                "endpoint_contract_sha256": {
                    branch: report["contract_sha256"]
                    for branch, report in sorted(endpoint_reports.items())
                },
                "split_seed": int(seed),
            },
        )
        return {
            "status": state["status"],
            "artifact_count": len(artifacts),
            "training_started": False,
            "external_outcomes_accessed": False,
        }
    except Exception as exc:
        atomic_write_json(
            output_root / "pretraining_validation_state.json",
            {
                "schema_version": "2.0",
                "status": "BLOCKED",
                "terminal_reason": f"{type(exc).__name__}:{exc}",
                "training_started": False,
                "outer_evaluation_started": False,
                "external_outcomes_accessed": False,
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--source-root",
        default="H:/New project/results/method_y_revival_v2",
        help="Read-only root containing locked endpoint and scanner artifacts.",
    )
    parser.add_argument(
        "--output-root",
        default="results/method_y_revival_v2/pretraining_contracts",
    )
    parser.add_argument(
        "--private-output-root",
        default="results/method_y_revival_v2/pretraining_contracts/_private",
    )
    parser.add_argument(
        "--config", default="configs/method_y_revival_v1.yaml"
    )
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()

    def resolve_project(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    source_root = resolve_project(args.source_root)
    output_root = resolve_project(args.output_root)
    private_root = resolve_project(args.private_output_root)
    config_path = resolve_project(args.config)
    validate_cli_roots(
        project_root=project_root,
        source_root=source_root,
        output_root=output_root,
        private_output_root=private_root,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pacc_source_hashes = _pacc_source_hashes(config)
    pacc_rules = config.get("pretraining_contracts_v2", {})
    expected_pacc_input_contract_sha256 = str(
        pacc_rules.get("pacc_legacy_input_contract_sha256", "")
    ).strip().lower()
    if len(expected_pacc_input_contract_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_pacc_input_contract_sha256
    ):
        raise PretrainingContractError(
            "PACC_LEGACY_INPUT_CONTRACT_SHA256_MISSING_OR_INVALID"
        )
    with WriterLock(output_root / ".writer.lock", output_root=output_root):
        result = build_all(
            source_root=source_root,
            output_root=output_root,
            private_output_root=private_root,
            config=config,
            seed=args.seed,
            expected_pacc_source_hashes=pacc_source_hashes,
            expected_pacc_input_contract_sha256=(
                expected_pacc_input_contract_sha256
            ),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
