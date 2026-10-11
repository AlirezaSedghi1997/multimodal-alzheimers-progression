"""Explicit input roots and an output boundary for restricted-data calculations."""
from pathlib import Path
import argparse, hashlib, json, sys

REPO=Path(__file__).resolve().parents[2]
_ARGS=None

def check_output_path(output,project_root,repo=REPO):
    output=Path(output).resolve(); project_root=Path(project_root).resolve(); repo=Path(repo).resolve()
    if output==project_root or project_root.is_relative_to(output):
        raise ValueError('OUTPUT_CANNOT_CONTAIN_SOURCE_PROJECT')
    if output.is_relative_to(project_root/'results'):
        raise ValueError('OUTPUT_CANNOT_MODIFY_LOCKED_RESULTS')
    if output.is_relative_to(repo) or repo.is_relative_to(output):
        raise ValueError('PRIVATE_OUTPUT_MUST_BE_OUTSIDE_GIT_CHECKOUT')
    return output

def configure():
    global _ARGS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True,help='Authorized original research tree')
    parser.add_argument('--clinical-root',type=Path,required=True,help='Authorized A4/LEARN Clinical directory')
    parser.add_argument('--output',type=Path,required=True,help='Private directory outside this checkout and locked results')
    parser.add_argument('--allow-frozen-inference',action='store_true',help='Allow 45 frozen CNN inference/rehead components; no CNN training')
    _ARGS=parser.parse_args()
    root=_ARGS.project_root.resolve(); clinical=_ARGS.clinical_root.resolve()
    output=check_output_path(_ARGS.output,root)
    inputs={
        'fusion_contract':root/'results/xyz_fusion_v2/private/contract.json',
        'clinical_subjects':clinical/'Derived Data/SUBJINFO.csv',
    }
    if not all(p.is_file() for p in inputs.values()):
        raise FileNotFoundError('RESTRICTED_PARENT_INPUTS_REQUIRED; see docs/STATISTICAL_SUPPLEMENT.md')
    state={'version':'THESIS_STATISTICS_V1_20261011',
           'input_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in inputs.items()},
           'source_code_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    output.mkdir(parents=True,exist_ok=True)
    marker=output/'.analysis_context.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf8'))!=state:
            raise ValueError('RESUME_INPUT_OR_CODE_MISMATCH; use a new private output directory')
    elif any(output.iterdir()):
        raise ValueError('UNVERIFIED_EXISTING_OUTPUT; use an empty private output directory')
    else:marker.write_text(json.dumps(state,indent=2),encoding='utf8')
    # The retained parent modules expose one historical clinical-root constant.
    # Set it before importing modules that bind the constant by value.
    from scripts.xyz_fusion_v1 import audit
    audit.RAW=clinical
    for name in ('scripts.xyz_fusion_v2.data','scripts.xyz_postprimary_v1.feasibility'):
        if name in sys.modules:sys.modules[name].RAW=clinical
    return root,output

def require_frozen_inference():
    if _ARGS is None or not _ARGS.allow_frozen_inference:
        raise RuntimeError('FROZEN_INFERENCE_REQUIRES_EXPLICIT_FLAG; --allow-frozen-inference')
