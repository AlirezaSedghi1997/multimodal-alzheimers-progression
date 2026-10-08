"""Cross-root hashes, exact stage coverage and aggregate-only dossier."""
import json
import shutil
from pathlib import Path
from scripts.xyz_fusion_postall_v2.core import Units,ReadOnlyParent,read_json,digest,file_sha,atomic_json,atomic_text
from scripts.xyz_fusion_postall_v2.context import Context
from scripts.xyz_fusion_postall_v2.release import verify_final,assert_public,PUBLIC_FILES

ROOTS={'postall':'results/xyz_fusion_postall_v2','manufacturer':'results/xyz_pet_manufacturer_audit_v1_ram',
       'regional':'results/xyz_regional_evidence_v1'}


def checked(root,relative):
    path=root/relative;contract=read_json(path/'private/contract.json')
    for name,sha in contract['source_sha256'].items():
        if file_sha(root/name)!=sha:raise ValueError('FINAL_SOURCE_DRIFT:'+name)
    units=Units(path,contract)
    for receipt in (path/'private/units').glob('*.json'):units.read(receipt.stem)
    return units,contract


def evidence(root):
    parent=ReadOnlyParent(root/'results/xyz_fusion_v2')
    post,pc=checked(root,ROOTS['postall'])
    ctx=Context(root,parent,post,pc['config']);final=verify_final(ctx)
    if not final.get('verifier','').startswith('PASS'):raise ValueError('FINAL_POSTALL_INCOMPLETE')
    lmo,lc=checked(root,ROOTS['manufacturer']);regional,rc=checked(root,ROOTS['regional'])
    if lc['postall']!=post.binding or rc['postall_contract']!=post.binding or lc['parent']!=parent.binding:
        raise ValueError('FINAL_PARENT_OR_COMPANION_BINDING_DRIFT')
    from scripts.xyz_pet_manufacturer_audit_v1.pipeline import validate_completed
    from scripts.xyz_pet_manufacturer_audit_v1.reporting import check_public
    import pandas as pd
    # validate_completed needs verified memberships only, not image loading.
    lctx=Context(root,parent,lmo,pc['config'])
    plan=lmo.job('plan')
    # Full private membership is bound at LMO input reconstruction; the frame
    # contains group labels recovered from its mutually exclusive partitions.
    members={}
    for group in plan['supported']:
        for key in group['test']:
            if key in members:raise ValueError('LMO_OVERLAPPING_HELD_GROUPS')
            members[key]=group['group']
    all_ids=set(k for g in plan['supported'] for k in g['train']+g['test'])
    lctx.pet=pd.DataFrame({'patient_hash':sorted(all_ids),'pet_manufacturer':[members.get(k,'EXCLUDED_GROUP') for k in sorted(all_ids)]})
    cells=validate_completed(lctx,plan,lc['options'])
    report=lmo.read('report',{'plan':digest(plan),'cells':digest(cells)})
    if report is None:raise ValueError('FINAL_LMO_REPORT_MISSING')
    check_public(lmo.root/'public')
    region_report=regional.read('report',{'f08':digest(post.read('f08')),'f04':digest(post.read('f04'))})
    if region_report is None:raise ValueError('FINAL_REGION_REPORT_MISSING')
    association=regional.read('associations')
    if association is None:raise ValueError('FINAL_REGION_ASSOCIATION_MISSING')
    from scripts.xyz_regional_evidence_v1.reporting import require_complete
    for endpoint in ('cdr','pacc'):require_complete(association[endpoint],rc['options']['regions'])
    public_region=read_json(regional.root/'public/REGIONAL_EVIDENCE.json')
    if public_region['associations']!=association:raise ValueError('FINAL_REGION_PUBLIC_MISMATCH')
    assert_public(public_region)
    from scripts.xyz_fusion_v2.privacy import scan_public_bundle
    if scan_public_bundle(regional.root/'public'):raise ValueError('FINAL_REGIONAL_PRIVACY')
    matrix={k.upper():post.read(k)['status'] for k in ('f00','f01','f02','f03','f04','f05','f06','f07','f08')}
    matrix['F05']=report['status']+'; PROBES_FROM_CORRECTED_POSTALL'
    matrix['F08']='COMBINED_DOSSIER_COMPUTED_PENDING_ACTUAL_POST_RUN_REVIEW'
    dependencies={'postall_contract':post.binding,'f08':digest(post.read('f08')),
        'manufacturer_contract':lmo.binding,'manufacturer_report':digest(report),
        'regional_contract':regional.binding,'regional_report':digest(region_report)}
    result={'status':'ALL_POSTFUSION_COMPUTATIONS_COMPLETE_PENDING_SUPERVISORY_REVIEW',
        'claim_status':'POST_HOC_INTERNAL_EXPLORATORY','stage_matrix':matrix,
        'regional_analysis_status':region_report['status'],'clinical_or_external_claim_allowed':False,
        'alzheimer_lesion_localization_allowed':False,'all_scientific_gates_passed':False,
        'parent_fusion_retrained':False,'supersession_promoted':False,
        'not_estimable_or_negative_results_are_valid_reported_outcomes_not_successful_gates':True,
        'sources':ROOTS}
    return result,dependencies


def build(root,units):
    result,dependencies=evidence(root)
    def render():
        public=units.root/'public';public.mkdir(parents=True,exist_ok=True)
        from scripts.xyz_pet_manufacturer_audit_v1.reporting import PUBLIC_FILES as LMO_FILES
        expected={*(f'postall/{x}' for x in PUBLIC_FILES),*(f'manufacturer/{x}' for x in LMO_FILES),
            'regional/REGIONAL_EVIDENCE.json','regional/REGIONAL_EVIDENCE_FA.md','regional/regional_hazard_ratios.svg',
            'FINAL_VALIDATION_MATRIX.json','START_HERE_FA.md'}
        present={p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file()}
        if not present<=expected:raise ValueError('COMBINED_PUBLIC_UNAPPROVED_FILE')
        for key,relative in ROOTS.items():
            for source in (root/relative/'public').rglob('*'):
                if source.is_file():
                    target=public/key/source.relative_to(root/relative/'public')
                    if target.relative_to(public).as_posix() not in expected:raise ValueError('COMPANION_PUBLIC_UNAPPROVED_FILE')
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        atomic_json(public/'FINAL_VALIDATION_MATRIX.json',result)
        lines=['# پروندهٔ یکپارچهٔ محاسبات اصلاح‌شده پس از فیوژن','',
            'این پوشه نتیجهٔ اجرای محاسبات است؛ پیش از استفادهٔ نهایی در پایان‌نامه، بازبینی واقعی اعداد و تصاویر توسط Codex باقی می‌ماند.',
            'تأیید انسانی مستقل شرط نیست. هیچ فایل Muse پاک نشده و مرجع رسمی هنوز خودکار جایگزین نشده است.',
            '', '| گام | وضعیت ثبت‌شده |','|---|---|']
        lines += [f'| {k} | {v} |' for k,v in result['stage_matrix'].items()]
        lines += ['','## محل نتایج','',
            '- postall: مقایسه‌های matched، calibration، CI، Tau، PACC، probe و XAI.',
            '- manufacturer: حساسیت محدود واقعی PET-manufacturer با MRI/PET expertهای تازه و بدون نشت سازندهٔ کنارگذاشته‌شده.',
            '- regional: ارتباط حجم نواحی پایه با CDR آینده و PACC، تعدیل مخدوش‌گرها و اصلاح چندگانگی.',
            '', '## مواردی که اجرای کامل آن‌ها را ثابت نمی‌کند','',
            '- داده خارجی نداریم؛ تمام تحلیل‌ها داخلی و post-hoc هستند.',
            '- چهاروجهی در support ناکافی fit نمی‌شود؛ NOT_ESTIMABLE نتیجهٔ امکان‌سنجی است.',
            '- XAI نامعتبر، صفر، ناپایدار یا دارای خطای عددی با عنوان تشخیص ضایعه منتشر نمی‌شود.',
            '- آزمون permutation برچسب کل ensemble اجرا نشده؛ random-weight کنترل مستقل دیگری است، نه معادل آن.',
            '- در F05 یک seed و سه جزء استفاده شده است؛ نتیجهٔ حساسیت با ensemble اصلیِ نه‌جزئی یکسان نیست.',
            '- native-space lesion localization و تشخیص اختصاصی آلزایمر اثبات نشده‌اند.',
            '- داده‌های فردی، checkpoint و تصاویر خصوصی در این bundle کپی نشده‌اند.']
        atomic_text(public/'START_HERE_FA.md','\n'.join(lines)+'\n')
        actual={p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file()}
        if actual!=expected:raise ValueError('COMBINED_PUBLIC_INCOMPLETE')
        from scripts.xyz_fusion_v2.privacy import scan_public_bundle
        if scan_public_bundle(public):raise ValueError('COMBINED_PUBLIC_PRIVACY')
        return dict(result,privacy='PASS',outputs={p.relative_to(units.root).as_posix():file_sha(p) for p in public.rglob('*') if p.is_file()})
    return units.job('combined_release',render,dependencies)


def verify(root,units):
    result,deps=evidence(root)
    proof=units.read('combined_release',deps)
    if proof is None:raise ValueError('COMBINED_RELEASE_MISSING')
    if any(proof.get(k)!=v for k,v in result.items()):raise ValueError('COMBINED_REPORT_MISMATCH')
    actual={p.relative_to(units.root).as_posix() for p in (units.root/'public').rglob('*') if p.is_file()}
    if actual!=set(proof['outputs']):raise ValueError('COMBINED_PUBLIC_WHITELIST_DRIFT')
    from scripts.xyz_fusion_v2.privacy import scan_public_bundle
    if scan_public_bundle(units.root/'public'):raise ValueError('COMBINED_PUBLIC_PRIVACY')
    return {'status':'PASS_ALL_POSTFUSION_EXECUTION_AND_CONSISTENCY_NOT_POSITIVE_SCIENCE',
            'actual_post_run_supervisory_review_still_required':True,'claim_status':result['claim_status']}
