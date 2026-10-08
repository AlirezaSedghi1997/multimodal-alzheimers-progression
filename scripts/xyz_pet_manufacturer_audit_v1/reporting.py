"""Aggregate manufacturer transport estimates; no automatic robustness claim."""
from scripts.xyz_fusion_postall_v2.core import atomic_json, atomic_text, file_sha
from scripts.xyz_fusion_postall_v2.release import assert_public
from scripts.xyz_fusion_v2.reporting import paired_ci
from .protocol import ARMS

PUBLIC_FILES={'PET_MANUFACTURER_AUDIT.json','PET_MANUFACTURER_AUDIT_FA.md'}


def check_public(root,complete=True):
    present={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if not present<=PUBLIC_FILES or complete and present!=PUBLIC_FILES:
        raise ValueError('LMO_PUBLIC_FILE_WHITELIST')
    from scripts.xyz_fusion_v2.privacy import scan_public_bundle
    if scan_public_bundle(root):raise ValueError('LMO_PUBLIC_PRIVACY_FAILED')


def build_report(ctx,plan,cells,options):
    if {c['fold'] for c in cells}!={r['group'] for r in plan['supported']}:
        raise ValueError('LMO_MISSING_SUPPORTED_MANUFACTURER')
    rows=[]
    for group,cell in zip(plan['supported'],cells):
        if cell['fold']!=group['group']:raise ValueError('LMO_GROUP_ORDER')
        def comparison():
            r=paired_ci([cell],*ARMS,ctx.config['fusion']['ibs_grid_days'],ctx.config['seed'],
                        options['bootstrap_resamples'],len(plan['supported']))
            return dict(r,family_size=len(plan['supported']),resampling='PAIRED_SUBJECT_WITHIN_HELD_MANUFACTURER',
                        full_training_uncertainty_included=False)
        contrast=ctx.units.job('contrast_'+group['group'].lower(),comparison)
        rows.append({'manufacturer':group['group'],'n':group['n'],'events':group['events'],
                     'arms':{a:r['metrics'] for a,r in cell['arms'].items()},'matched_contrast':contrast})
    result={'status':'PET_MANUFACTURER_AUDIT_COMPLETED' if cells else 'PET_MANUFACTURER_AUDIT_NOT_ESTIMABLE',
        'claim_status':'POST_HOC_INTERNAL_EXPLORATORY','groups':rows,'excluded':plan['excluded'],
        'scope':options['scope'],'training_seed':options['training_seed'],'components':options['components'],
        'new_expert_fits':plan['expected_expert_fits'],'site_lmo_status':plan['site_status'],
        'both_mri_and_pet_training_selection_exclude_held_manufacturer':True,
        'parent_training_or_results_changed':False,'external_validation':False,
        'scanner_invariance_proven':False,'clinical_claim_allowed':False,
        'robustness_generalization_to_unobserved_devices_not_established':True,
        'parent_oof_performance_not_directly_compared_to_different_lmo_training_sizes':True}
    assert_public(result)
    public=ctx.out/'public';public.mkdir(parents=True,exist_ok=True);check_public(public,complete=False)
    atomic_json(public/'PET_MANUFACTURER_AUDIT.json',result)
    lines=['# حساسیت انتقال میان سازندگان PET','',
        'تحلیل داخلی و پس‌نگر؛ یک seed ثابت و سه جزء، نه ensemble اصلیِ نه‌جزئی.',
        'هر دو expert MRI و PET، انتخاب LR و epoch، و headها بدون افراد سازندهٔ کنارگذاشته‌شده ساخته شده‌اند.',
        'دو بازو در هر سازنده افراد، پیامد، censoring و fold یکسان دارند. دادهٔ خارجی در دسترس نیست.',
        '', '| سازنده | N / رویداد | اختلاف IBS با افزودن PET | CI 95% مشروط | CI تعدیل خانواده |',
        '|---|---|---|---|---|']
    for row in rows:
        c=row['matched_contrast'];lines.append(f"| {row['manufacturer']} | {row['n']} / {row['events']} | {c['delta_ibs_mean']} | {c['delta_ibs_ci95']} | {c['delta_ibs_bonferroni_ci']} |")
    lines+=['','IBS پایین‌تر مطلوب است؛ Uno بالاتر مطلوب است. تمام calibrationها، horizon support و CIها در JSON آمده‌اند.',
        'CIها مشروط به مدل‌های برازش‌شده‌اند؛ bootstrap بازآموزی شبکه‌ها را تکرار نمی‌کند.',
        'کمبود support در گروه یا site حذف پنهانی نمی‌شود. UNKNOWN حدس زده نمی‌شود و GE/Philips ادغام نمی‌شوند.',
        'این حساسیت نه اثبات استقلال scanner، نه ارزیابی خارجی، و نه جایگزین نتیجهٔ ensemble اصلی است.']
    atomic_text(public/'PET_MANUFACTURER_AUDIT_FA.md','\n'.join(lines)+'\n');check_public(public)
    return {'status':result['status'],'manufacturers_evaluated':len(cells),'privacy':'PASS',
            'outputs':{p.relative_to(ctx.out).as_posix():file_sha(p) for p in public.iterdir() if p.is_file()}}
