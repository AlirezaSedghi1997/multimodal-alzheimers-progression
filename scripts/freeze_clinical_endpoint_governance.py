from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.state_manager import StateManager

FROZEN = {
    "schema_version": 1,
    "status": "FROZEN_BEFORE_MODELING",
    "branch": "clinical_endpoint",
    "phase": 3,
    "claim_boundary": (
        "Exploratory clinical/cognitive progression proxy only. "
        "Do not call Alzheimer diagnosis or conversion-to-AD without independent validated label."
    ),
    "forbidden_claims": [
        "Alzheimer disease diagnosis",
        "conversion-to-AD",
        "clinical AD classification",
    ],
    "endpoints": {
        "cdr_worsening": {
            "status": "FROZEN",
            "definition": (
                "Binary progression endpoint: baseline global CDR equals 0.0 and any later visit "
                "global CDR is greater than or equal to 0.5 within the locked enrolled A4/LEARN follow-up window."
            ),
            "baseline_visit_rule": "baseline_only_predictors",
            "minimum_followup_visits": 2,
            "worsening_threshold": "global_cdr >= 0.5 after baseline global_cdr == 0.0",
            "enrolled_only": True,
            "sf_longitudinal_limitation": "SF rows excluded from primary endpoint modeling; report separately if used.",
        },
        "pacc_decline": {
            "status": "FROZEN",
            "definition": (
                "Binary decline endpoint: PACC composite decline of at least 1.0 point from participant baseline "
                "PACC on any later locked visit within enrolled A4/LEARN follow-up."
            ),
            "baseline_visit_rule": "baseline_only_predictors",
            "minimum_followup_visits": 2,
            "decline_threshold": "delta_pacc <= -1.0 from baseline",
            "enrolled_only": True,
            "sf_longitudinal_limitation": "SF rows excluded from primary endpoint modeling; report separately if used.",
        },
    },
    "baseline_predictor_allowlist": [
        "age",
        "sex",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
        "baseline_cognitive_scores_non_future",
    ],
    "non_future_visit_discipline": True,
    "sf_reporting_policy": "SF may be used only for internal decoupling/stress tests; never external validation.",
    "date_frozen": str(date.today()),
    "approved_by": "PROJECT_OWNER_PHASE3_GOVERNANCE",
    "change_policy": "No endpoint definition change after modeling starts or performance inspection.",
    "modeling_started_before_freeze": False,
    "supports_method_x_strict_claim": False,
    "pet_as_input": False,
    "raw_identifiers_written": False,
    "raw_paths_written": False,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze clinical endpoint progression governance (CPU-only).")
    ap.add_argument("--out_json", default="results/governance/clinical_endpoint_freeze.json")
    ap.add_argument("--out_md", default="governance/clinical_endpoint_progression_policy.md")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    atomic_write_json(out_json, FROZEN)

    md = [
        "# Clinical Endpoint Progression Policy\n\n",
        f"- status: **`{FROZEN['status']}`**\n",
        f"- date frozen: `{FROZEN['date_frozen']}`\n",
        f"- branch: `{FROZEN['branch']}`\n\n",
        "## Claim boundary\n\n",
        f"{FROZEN['claim_boundary']}\n\n",
        "## CDR worsening (frozen)\n\n",
        f"{FROZEN['endpoints']['cdr_worsening']['definition']}\n\n",
        f"- threshold: `{FROZEN['endpoints']['cdr_worsening']['worsening_threshold']}`\n\n",
        "## PACC decline (frozen)\n\n",
        f"{FROZEN['endpoints']['pacc_decline']['definition']}\n\n",
        f"- threshold: `{FROZEN['endpoints']['pacc_decline']['decline_threshold']}`\n\n",
        "## SF policy\n\n",
        f"{FROZEN['sf_reporting_policy']}\n\n",
        "## Forbidden claims\n\n",
    ]
    for claim in FROZEN["forbidden_claims"]:
        md.append(f"- {claim}\n")
    atomic_write_text(out_md, "".join(md))

    StateManager().append_event(
        "clinical_endpoint_governance",
        FROZEN["status"],
        "Clinical endpoint progression definitions frozen.",
    )
    print(json.dumps({"clinical_endpoint_governance": FROZEN["status"], "path": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
