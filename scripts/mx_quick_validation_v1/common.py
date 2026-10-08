"""Private, atomic, content-bound receipts and strict parent verification."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2_1.supervise_v21 import immutable_json
from scripts.mx_science_extension_v1.audit_and_lock import OUTPUT as EXT, PARENT

CONFIG = Path('configs/experiments/method_x_quick_validation_v1.json')
CACHE = Path('data/cached/t1_volume_cache_128.h5')
INDEX = Path('docs/fasttrack_private/cache_full_index.csv')
REGISTRY = Path('results/spatial_attribution_v1/provenance/atlas_registry.json')
PANEL = EXT / '_private_resume/outer_v1/patient_panel.v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def progress(stage, **values):
    print(json.dumps({'quick_step': stage, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


class Store:
    def __init__(self, root, config, protocol):
        self.root, self.config, self.protocol = Path(root), config, protocol
        self.out = self.root / config['output']
        require(self.out.resolve().is_relative_to((self.root / 'results/final_results').resolve()), 'OUTPUT_SCOPE')
        self.private = self.out / '_private_resume'
        self.reports = self.out / 'reports'
        self.binding = digest(protocol)

    def receipt(self, name, compute=None, *, public=False):
        path = (self.reports if public else self.private) / (name + '.json')
        if path.exists():
            row = read(path)
            require(row.get('unit') == name and row.get('protocol_sha256') == self.binding and
                    row.get('payload_sha256') == digest(row.get('payload')), 'RECEIPT_DRIFT:' + name)
            return row['payload']
        require(compute is not None, 'RECEIPT_MISSING:' + name)
        payload = compute()
        immutable_json(path, {'unit': name, 'protocol_sha256': self.binding, 'payload_sha256': digest(payload), 'payload': payload})
        return payload


def source_hashes(root):
    # Include the actual imported scientific implementation, not just the CLI.
    paths = {CONFIG, Path('docs/plans/method_x_quick_validation_v1_implementation_FA.md')}
    for directory in ('scripts/mx_quick_validation_v1', 'scripts/mx_science_extension_v1', 'src'):
        paths.update(p.relative_to(root) for p in (root / directory).rglob('*.py'))
    paths.update(p.relative_to(root) for d in (root / 'scripts').glob('mx_fasttrack*') for p in d.rglob('*.py'))
    paths.add(Path('scripts/train_3d_mri.py'))
    return {p.as_posix(): sha(root / p) for p in sorted(paths)}


def verify_parents(root):
    from scripts.mx_fasttrack_v2_4_1.compatible_lock import check_lock_compatible
    from scripts.mx_fasttrack_v2.freeze_report import evaluated
    from scripts.mx_science_extension_v1.execution import ensure_execution_lock
    from scripts.mx_science_extension_v1.outer import ensure_outer_lock
    from scripts.mx_science_extension_v1.report import xai_audit_payload
    from scripts.mx_science_extension_v1.analysis import validate_panel
    from scripts.mx_science_extension_v1.terminal import require_parent_mx7_terminal

    progress('parent_verification', status='HASHING_CACHE_AND_CHECKPOINTS')
    parent = root / PARENT
    lock = check_lock_compatible(root, parent, hash_cache=True)
    terminal = require_parent_mx7_terminal(root)
    frozen = read(parent / 'reports/fasttrack_thesis_traceability.json')
    require(frozen['status']['MX07_fasttrack'] == 'READY_SCOPED_FASTTRACK', 'PARENT_NOT_FROZEN')
    for row in frozen['asset_inventory'].values():
        require(sha(root / row['relative_path']) == row['sha256'], 'FROZEN_ASSET_DRIFT')
    thesis = read(parent / 'reports/fasttrack_v22_thesis_manifest.json')
    for name, expected in {
        'fasttrack_v22_protocol.json': thesis['protocol_sha256'],
        'xai_v22_summary.json': thesis['xai_v22_summary_sha256'],
        'xai_v22_context.json': thesis['section_sources']['methods']['v22_xai_context_sha256'],
    }.items():
        require((parent/'reports'/name).is_file() and sha(parent/'reports'/name)==expected,
                'THESIS_XAI_PROVENANCE_DRIFT')
    for row in thesis['documents'].values():
        require(sha(root / row['relative_path']) == row['sha256'], 'THESIS_DRIFT')
    lock_sha = sha(parent / 'reports/fasttrack_inner_lock.json')
    # The original evaluator verifier checks sample counts, states and paired results.
    pair = evaluated(parent, 'evaluation_outer_hybrid_pair', lock_sha, None, paired=True)
    scratch = evaluated(parent, 'evaluation_outer_mx04_scratch', lock_sha, 'mx04_scratch')
    head = evaluated(parent, 'evaluation_outer_mx04_head_last', lock_sha, 'mx04_head_last')
    require(all((pair, scratch, head)), 'MANDATORY_OUTER_MISSING')
    ensure_execution_lock(root, create=False)
    ensure_outer_lock(root, create=False)
    er = root / EXT / 'reports'
    manifest = read(er / 'extension_mx7_manifest.v1.json')
    require(manifest['status'] == 'COMPLETED_POST_HOC_INTERNAL_EXPLORATORY', 'EXTENSION_NOT_COMPLETE')
    links = {
        'execution_preregistration_sha256': er / 'execution_preregistration.v1.json',
        'fold_local_selection_sha256': er / 'fold_local_hybrid_selection.v1.json',
        'outer_lock_sha256': er / 'extension_outer_lock.v1.json',
        'private_panel_sha256': root / PANEL,
        'paired_analysis_sha256': er / 'extension_paired_analysis.v1.json',
        'xai_receipt_audit_sha256': er / 'extension_xai_receipt_audit.v1.json',
        'persian_results_limitations_sha256': er / 'extension_mx7_results_limitations_FA.v1.md',
    }
    require(all(sha(p) == manifest[k] for k, p in links.items()), 'EXTENSION_MANIFEST_DRIFT')
    require(manifest['parent_freeze_sha256'] == terminal['freeze_sha256'] and
            manifest['parent_thesis_manifest_sha256'] == terminal['thesis_manifest_sha256'], 'EXTENSION_PARENT_DRIFT')
    panel = read(root / PANEL)
    validate_panel(panel['rows'])
    require(panel['outer_lock_sha256'] == manifest['outer_lock_sha256'], 'PANEL_BINDING')
    # Bind every panel patient to the exact split and saved fold-arm probabilities.
    selection = read(er / 'fold_local_hybrid_selection.v1.json')
    choices = {(c['fold'], c['seed']): c['selected_variant'] for c in selection['choices']}
    for fold in range(5):
        folder = root / EXT / '_private_resume/outer_v1' / f'fold_{fold}'
        context = read(folder / 'context.json')['rows']
        rows = [r for r in panel['rows'] if r['fold'] == fold]
        require([(r['patient_hash'], r['label']) for r in rows] ==
                [(r['patient_hash'], r['label']) for r in context], 'PANEL_MEMBERSHIP_DRIFT')
        for seed in (13, 29, 47):
            v = choices[fold, seed]
            probs = read(folder / f'hybrid_{v}_seed{seed}.json')['probabilities']
            require([r['predictions']['hybrid_selected'][str(seed)] for r in rows] == probs, 'PANEL_SELECTION_DRIFT')
    progress('parent_verification', status='VERIFYING_XAI_SPM_RECEIPTS')
    xai = xai_audit_payload(root)
    require(xai == read(er / 'extension_xai_receipt_audit.v1.json'), 'XAI_RECEIPT_DRIFT')
    return {'fasttrack': frozen['status'], 'mx5_outer_n': pair['n'],
            'extension_outer_n': len(panel['rows']), 'extension_six_arms': 'COMPLETED_VERIFIED',
            'xai_computations': 'COMPLETED_FIVE_FOLDS_40_IMAGES', 'xai_scientific_gate': xai['status'],
            'temporal_direction_appendix': 'PRESENT', 'external_validation': False,
            'original_full_program_completed': False,
            'unresolved': ['COHORT_LABEL_CONFOUNDING', 'NO_EXTERNAL_DATA', 'PSEUDO_HISTORY_UNKNOWN',
                           'SITE_SCANNER_UNKNOWN', 'PET_FILE_PROVENANCE_UNVERIFIED'],
            'new_work': ['IMAGE_CONTENT_AUDIT', 'NONLINEAR_TABULAR', 'DEMOGRAPHIC_FOLD_SENSITIVITY', 'FIXED_CHECKPOINT_STRESS']}


def protocol_payload(root, config, audit, mask_meta):
    inputs = {INDEX, REGISTRY, PANEL}
    inputs.update(p.relative_to(root) for p in (root / 'splits').glob('*.csv'))
    for base in (PARENT, EXT):
        for section, pattern in (('reports', '*.json'), ('reports', '*.md'), ('inner_tasks', 'method_x_3d_fold_[0-4].pt'),
                                 ('inner_tasks', '*state.json'), ('inner_tasks', 'run_contract.json'),
                                 ('inner_tasks', 'three_d_metrics.json'), ('_private_resume/outer_v1', '*.json')):
            inputs.update(p.relative_to(root) for p in (root / base / section).rglob(pattern))
    # Hash cache once in verify_parents on every invocation; avoid a second 21 GB pass.
    cache_sha = read(root / PARENT / 'reports/fasttrack_preregistration.json')['cache_sha256']
    return {'schema_version': 1, 'config': config, 'source_sha256': source_hashes(root),
            'input_sha256': {p.as_posix(): sha(root / p) for p in sorted(inputs)},
            'cache_sha256': cache_sha, 'cache_bytes': (root / CACHE).stat().st_size,
            'cache_mtime_ns': (root / CACHE).stat().st_mtime_ns,
            'parent_audit': audit, 'mask': mask_meta,
            'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'scikit-learn', 'torch', 'h5py', 'nibabel')},
            'registration': 'BEFORE_NEW_ANALYSES_BUT_AFTER_HISTORICAL_OUTER', 'external_validation': False}
