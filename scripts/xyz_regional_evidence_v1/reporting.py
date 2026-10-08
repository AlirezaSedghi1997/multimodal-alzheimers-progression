"""Separate observed regional associations from illustrative model maps."""
from scripts.xyz_fusion_postall_v2.core import atomic_json, atomic_text, file_sha
from scripts.xyz_fusion_postall_v2.release import assert_public


def require_complete(values, regions):
    if set(values) != set(regions):
        raise ValueError('REGIONAL_COVERAGE_INCOMPLETE')


def anatomical_mapping(labels):
    expected={'hippocampus':['Left Hippocampus','Right Hippocampus'],
              'entorhinal':['Left Entorhinal Cortex','Right Entorhinal Cortex'],
              'lateral_ventricle':['Left Lateral Ventricle','Right Lateral Ventricle']}
    result={}
    for roi, names in expected.items():
        matched=[]
        for name in names:
            ids=[int(k) for k,v in labels.items() if v==name]
            if len(ids)!=1:break
            matched+=ids
        result[roi]={'status':'ANATOMICAL_CORRESPONDENCE_NOT_QUANTITATIVE_MASK_EQUIVALENCE' if len(matched)==2 else
                     'NO_EXACT_LABEL_NO_IMAGE_MAPPING','labels':matched if len(matched)==2 else [],
                     'fuzzy_mapping_used':False}
    return result


def write_report(out, associations, spatial, options):
    require_complete(associations['cdr'],options['regions'])
    require_complete(associations['pacc'],options['regions'])
    value={'status':'REGIONAL_POSTHOC_DOSSIER_COMPLETED','claim_status':'POST_HOC_INTERNAL_EXPLORATORY',
        'question':'Baseline regional volume associated with future confirmed CDR progression, not Alzheimer disease causation',
        'analysis_timing':'ADDED_AFTER_ORIGINAL_STUDY_RESULTS_BEFORE_THIS_EXTENSION_FITS',
        'region_selection':'LOCKED_SOURCE_VOLUMETRY_AND_LITERATURE_NOT_HEATMAP_OR_LOWEST_P_VALUE',
        'associations':associations,'spatial_evidence':spatial,
        'disease_region_discovery_claim_allowed':False,'external_validation':False,
        'causal_localization':False,'all_results_retained':True}
    assert_public(value)
    public=out/'public';public.mkdir(parents=True,exist_ok=True)
    allowed={'REGIONAL_EVIDENCE.json','REGIONAL_EVIDENCE_FA.md','regional_hazard_ratios.svg'}
    if any(p.relative_to(public).as_posix() not in allowed for p in public.rglob('*') if p.is_file()):
        raise ValueError('UNAPPROVED_REGIONAL_PUBLIC_FILE')
    atomic_json(public/'REGIONAL_EVIDENCE.json',value)
    rows=['# شواهد ناحیه‌ای اکتشافی؛ نه شناسایی ضایعهٔ آلزایمر','',
        'تحلیل پس از مشاهدهٔ نتایج اصلی پژوهش اضافه شده است و confirmatory یا preregistered نیست.',
        'پرسش اصلی رابطهٔ حجم ناحیه‌ای پایه با زمان تا پیشرفت تأییدشدهٔ آینده CDR است؛ شدت heatmap پیامد این رگرسیون نیست.',
        '', '| ناحیه | N / رویداد | HR به‌ازای یک SD لگاریتم حجم | CI 95% | p اصلاح‌شده Holm | وضعیت |',
        '|---|---|---|---|---|---|']
    for roi, r in associations['cdr'].items():
        rows.append(f"| {roi} | {r.get('n')} / {r.get('events')} | {r.get('hazard_ratio')} | {r.get('ci95')} | {r.get('p_holm')} | {r.get('interpretation')} |")
    rows+=['','## حدود تفسیر','',
        '- تعدیل: سن، جنس، تحصیلات، APOE، شناخت پایه، treatment، بار amyloid و حجم داخل‌جمجمه؛ baseline hazard بر اساس manufacturer MRI طبقه‌بندی شده است.',
        '- تحلیل complete-case محدود به همان جمعیت قابل‌اندازه‌گیری است؛ حذف مخدوش‌سازی یا سازوکار MCAR/MAR اثبات نشده است.',
        '- CIهای ۹۵٪ اسمی‌اند؛ تصمیم خانوادهٔ سه‌ناحیه‌ای با Holm گرفته شده، نه صرف CI منفرد.',
        '- آزمون PH و شکل رابطه محدودند؛ عدم معناداری آن‌ها اثبات صحت همهٔ فرض‌ها نیست. شکاف یا بی‌ثباتی باید کنار HR گزارش شود.',
        '- بررسی leave-one-fold-out تحلیل پایداری روی زیرمجموعه‌های همپوشان است، نه پنج replication مستقل.',
        '- PACC نتیجهٔ ثانویه در همان cohort است و replication مستقل نیست.',
        '- بطن جانبی فضای مایع و شاخص مورفولوژیک غیراختصاصی است؛ ناحیهٔ اختصاصی آسیب آلزایمر معرفی نشود.',
        '- همخوانی با ادبیات، نوآوریِ کشف ناحیهٔ جدید یا اثبات سازوکار علّی ایجاد نمی‌کند.',
        '- XAI سهم در تابع پیش‌بینی مدل را توصیف می‌کند. نقشهٔ زیبا، association ناحیه‌ای و تشخیص بافت بیمار سه مفهوم جدا هستند.',
        '- انتورینال بدون label دقیق اطلس، از Parahippocampal Gyrus یا fuzzy mapping روی تصویر ساخته نمی‌شود.',
        '', '## الگوی نگارش نتیجه','',
        '«در تحلیل داخلی و اکتشافی این cohort، اندازه‌گیری پایهٔ [نام ناحیه] با [پیامد واقعی] ارتباط [مشاهده شد/به‌طور کافی پشتیبانی نشد]؛ HR، CI و p اصلاح‌شده در جدول گزارش شده‌اند. این ارتباط نشان‌دهندهٔ نقش علّی یا شناسایی اختصاصی آسیب آلزایمر نیست.»',
        '', 'نتایج و فرض‌های کامل CDR، PACC، حساسیت‌ها و وضعیت مستقل نقشه در فایل JSON تجمیعی همراه ثبت شده‌اند.']
    atomic_text(public/'REGIONAL_EVIDENCE_FA.md','\n'.join(rows)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for i,(roi,r) in enumerate(associations['cdr'].items()):
        if r.get('status')=='ESTIMATED':
            hr=r['hazard_ratio'];lo,hi=r['ci95']
            ax.errorbar(hr,i,xerr=[[hr-lo],[hi-hr]],fmt='o',color='navy',capsize=3)
    ax.set_yticks(range(len(options['regions'])),list(options['regions']))
    ax.axvline(1,color='gray',linestyle='--');ax.set_xscale('log')
    ax.set_xlabel('Hazard ratio per SD of log regional volume; nominal 95% CI')
    ax.set_title('Adjusted exploratory association; not Alzheimer lesion localization')
    fig.tight_layout();fig.savefig(public/'regional_hazard_ratios.svg');plt.close(fig)
    from scripts.xyz_fusion_v2.privacy import scan_public_bundle
    if scan_public_bundle(public):raise ValueError('REGIONAL_PUBLIC_PRIVACY_FAILED')
    return {'status':value['status'],'outputs':{p.relative_to(out).as_posix():file_sha(p) for p in public.iterdir()},
            'privacy':'PASS','disease_region_discovery_claim_allowed':False}
