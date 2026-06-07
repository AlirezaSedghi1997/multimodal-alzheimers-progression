from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.state_manager import StateManager

TEMPLATE = {
    "schema_version": 1,
    "status": "TEMPLATE_NOT_FROZEN",
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
            "status": "PLACEHOLDER",
            "definition": "[INSERT PRESPECIFIED CDR WORSENING DEFINITION BEFORE MODELING]",
            "baseline_visit_rule": "baseline_only_predictors",
            "minimum_followup_visits": 2,
            "worsening_threshold": "[INSERT THRESHOLD]",
            "sf_longitudinal_limitation": "SF rows lack enrolled-style longitudinal coverage; report separately.",
        },
        "pacc_decline": {
            "status": "PLACEHOLDER",
            "definition": "[INSERT PRESPECIFIED PACC DECLINE DEFINITION BEFORE MODELING]",
            "baseline_visit_rule": "baseline_only_predictors",
            "minimum_followup_visits": 2,
            "decline_threshold": "[INSERT THRESHOLD]",
            "sf_longitudinal_limitation": "SF rows lack enrolled-style longitudinal coverage; report separately.",
        },
    },
    "required_fields_before_freeze": [
        "endpoint_definition",
        "worsening_or_decline_threshold",
        "baseline_predictor_allowlist",
        "non_future_visit_discipline",
        "sf_reporting_policy",
        "date_frozen",
        "approved_by",
        "change_policy",
    ],
    "change_policy": "No endpoint definition change after modeling starts or performance inspection.",
    "modeling_started_before_freeze": False,
    "supports_method_x_strict_claim": False,
    "pet_as_input": False,
    "raw_identifiers_written": False,
    "raw_paths_written": False,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Write clinical endpoint governance template (CPU-only).")
    ap.add_argument("--out_json", default="results/governance/clinical_endpoint_freeze_template.json")
    ap.add_argument("--out_md", default="governance/clinical_endpoint_progression_policy.md")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    atomic_write_json(out_json, TEMPLATE)

    md = [
        "# Clinical Endpoint Progression Policy (Template)\n\n",
        f"- status: **`{TEMPLATE['status']}`**\n",
        f"- branch: `{TEMPLATE['branch']}`\n\n",
        "## Claim boundary\n\n",
        f"{TEMPLATE['claim_boundary']}\n\n",
        "## Endpoints (not frozen)\n\n",
        "### CDR worsening\n\n",
        f"- definition: {TEMPLATE['endpoints']['cdr_worsening']['definition']}\n",
        f"- worsening threshold: {TEMPLATE['endpoints']['cdr_worsening']['worsening_threshold']}\n",
        f"- SF limitation: {TEMPLATE['endpoints']['cdr_worsening']['sf_longitudinal_limitation']}\n\n",
        "### PACC decline\n\n",
        f"- definition: {TEMPLATE['endpoints']['pacc_decline']['definition']}\n",
        f"- decline threshold: {TEMPLATE['endpoints']['pacc_decline']['decline_threshold']}\n",
        f"- SF limitation: {TEMPLATE['endpoints']['pacc_decline']['sf_longitudinal_limitation']}\n\n",
        "## Required before freeze\n\n",
    ]
    for item in TEMPLATE["required_fields_before_freeze"]:
        md.append(f"- {item}\n")
    md.extend(["\n", "## Forbidden claims\n\n"])
    for claim in TEMPLATE["forbidden_claims"]:
        md.append(f"- {claim}\n")
    atomic_write_text(out_md, "".join(md))

    StateManager().append_event(
        "clinical_endpoint_governance",
        TEMPLATE["status"],
        "Clinical endpoint freeze template written.",
    )
    print(json.dumps({"clinical_endpoint_governance": TEMPLATE["status"], "path": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
