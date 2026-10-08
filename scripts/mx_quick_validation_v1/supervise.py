"""One command; restart the same command to reuse each verified committed unit."""
from __future__ import annotations
import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.mx_quick_validation_v1.common import (
    CONFIG, PANEL, CACHE, PARENT, EXT, Store, read, sha, digest, require, progress, source_hashes,
    protocol_payload, verify_parents, immutable_json,
)


def freeze_text(path, text):
    if path.exists():
        require(path.read_text(encoding='utf-8')==text,'FINAL_TEXT_DRIFT')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.new')
    try:
        with temporary.open('x',encoding='utf-8',newline='') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def report(store):
    audit=store.receipt('data_audit',public=True)
    statistics=store.receipt('sensitivity_summary',public=True)
    stress=store.receipt('stress_summary',public=True)
    parent=store.protocol['parent_audit']
    def effect(item,metric='brier'):
        r=item.get('effects',{}).get(metric)
        return f"{r['delta']:.5f} [{', '.join(f'{v:.5f}' for v in r['ci95'])}]" if r and r['ci95'] else 'غیرقابل‌برآورد'
    methods='\n'.join([
        '# روش افزونهٔ سریع Method X', '',
        'این تحلیل از پیش برای خروجی‌های تازه مشخص شده، اما داده‌های outer تاریخی قبلاً دیده شده‌اند؛ اعتبارسنجی مستقل نیست.',
        'واحد تحلیل بیمار است. پنج فولد و سه بذر ۱۳/۲۹/۴۷؛ دادهٔ گمشده فقط با میانهٔ fit پر می‌شود.',
        'کنترل غیرخطی HistGradientBoosting فقط پنج ویژگی مجاز دارد؛ عمق ۲ یا ۳ تنها با BCE در inner انتخاب می‌شود؛ outer برای انتخاب استفاده نمی‌شود.',
        'پاسخ اصلی مقایسه، Brier مدل Hybrid منتخب منهای کنترل غیرخطی است. AUROC، AUPRC و کالیبراسیون ثانویه‌اند.',
        'Bootstrap جفتی ۲۰۰۰ بار، با حفظ تمام بذرهای هر بیمار انجام می‌شود. CI به مدل‌های آموزش‌دیده مشروط است و تغییرپذیری بازآموزی را پوشش نمی‌دهد.',
        'زیرگروه با کمتر از ۲۰ بیمار در هر کلاس برآورد نمی‌شود. این حد، تضمین توان آماری نیست. حذف یک فولد، تحلیل حساسیت است نه CV جدید.',
        'تطبیق بدون جایگزینی با جنس یکسان، فاصلهٔ سن حداکثر سه سال، در همان فولد و بدون استفاده از پیش‌بینی انجام می‌شود؛ bootstrap روی جفت‌های تطبیق است.',
        'پایداری: ۴۰ بیمار تصادفی طبقه‌بندی‌شده در هر فولد؛ دو مسیر checkpoint ثابت؛ هفت وضعیت. وزن N/n طبقه و bootstrap داخل طبقهٔ فولد/کلاس استفاده می‌شود.',
        'ماسک ثابت از مجموع سه بافت TPM نرم‌افزار SPM و آستانهٔ ۰٫۵ روی شبکهٔ مدل ساخته می‌شود؛ ماسک اختصاصی هر فرد نیست.',
        'نویز/میدان شدت/blur داخل ماسک؛ زمینهٔ حذف‌شده صفر است. تغییرات مصنوعی جانشین بازآزمایی MRI واقعی نیستند.',
        'رسید هر تصویر ممیزی‌شده، هر فولد/بذر و هر بیمار/شرط اتمی و متصل به هش پروتکل است. پس از قطع فقط واحد ثبت‌نشده تکرار می‌شود.',
        f"هش پروتکل: `{store.binding}`",''])
    lines=['# نتایج افزونهٔ سریع Method X','','## تطبیق برنامه‌های قبلی','',
           f"MX4 دو بازوی محدود و MX5 دو Hybrid با outer جفتی {parent['mx5_outer_n']} بیمار تأیید شدند؛ افزونهٔ قبلی نیز {parent['extension_outer_n']} بیمار دارد.",
           f"محاسبات XAI: {parent['xai_computations']}؛ گیت علمی: `{parent['xai_scientific_gate']}`. شکست گیت با موفقیت اجرای محاسبات عوض نشده است.",
           'MX7 fast-track و پیوست افزونه موجودند؛ برنامهٔ کامل اولیهٔ MX4–MX7 با این دامنهٔ محدود معادل نیست.',
           '', '## ممیزی تصاویر', '',
           f"حجم‌های واقعی: {audit['real_volumes']}؛ گروه‌های یکسان: {audit['exact_duplicate_groups']}؛ نامزدهای نیازمند بررسی انسانی: {audit['near_duplicate_candidates']}.",
           '', '## مقایسهٔ Hybrid منتخب با کنترل غیرخطی', '',
           '| دامنه | n | Δ Brier [CI95] | Δ AUROC [CI95] | Δ AUPRC [CI95] |','|---|---:|---|---|---|']
    for name,row in statistics['subsets'].items():
        lines.append(f"| {name} | {row['n']} | {effect(row)} | {effect(row,'auroc')} | {effect(row,'auprc')} |")
    lines += ['', '### کالیبراسیون کل نمونه (برآورد نقطه‌ای بدون CI تازه)', '',
              '| بذر | Hybrid intercept / slope | کنترل غیرخطی intercept / slope |', '|---|---|---|']
    calibration=statistics['subsets']['all'].get('calibration_by_seed',{})
    def show_calibration(value):
        return ' / '.join(f'{v:.5f}' for v in value) if value else 'غیرقابل‌برآورد'
    if calibration:
        for j,seed in enumerate(store.config['seeds']):
            lines.append(f"| {seed} | {show_calibration(calibration['first'][j])} | {show_calibration(calibration['second'][j])} |")
    lines+=['',f"جفت‌های تطبیق: {statistics['matched_pairs']}؛ افراد خارج از نمونهٔ تطبیق: {statistics['unmatched_patients']}.",
            'Δ منفی Brier بهتر است؛ Δ مثبت AUROC/AUPRC بهتر است. همهٔ جهت‌های منفی و CIهای شامل صفر حفظ شده‌اند.',
            '', '## پایداری با وزن ثابت', '', '| مدل | شرط نسبت به clean | Δ Brier [CI95] | Δ AUROC [CI95] | تغییر احتمال | نرخ تغییر تصمیم |', '|---|---|---|---|---:|---:|']
    for arm,conditions in stress['comparisons'].items():
        for condition,row in conditions.items():
            lines.append(f"| {arm} | {condition} | {effect(row)} | {effect(row,'auroc')} | {row['mean_probability_change']:.5f} | {row['fixed_threshold_05_flip_rate']:.5f} |")
    lines+=['',f"کسر متوسط زمینهٔ غیرصفر: {stress['mean_nonzero_background_fraction']:.6f}. زمینه ممکن است از پیش در پیش‌پردازش حذف شده باشد؛ در این صورت آزمون زمینه شاهد مستقل تازه‌ای نیست.",'']
    discussion='\n'.join(['# بحث، نتایج منفی و خطر سوگیری','','| حوزه | وضعیت و اثر بر ادعا |','|---|---|',
        '| cohort و برچسب | هم‌بستگی کامل؛ اثر بیماری از cohort قابل جداسازی نیست |',
        '| دادهٔ بیرونی | موجود نیست؛ هیچ ادعای تعمیم مستقل یا کاربرد بالینی مجاز نیست |',
        '| outer تاریخی | قبلاً دیده شده؛ تحلیل پسینی و مقایسه‌های متعدد اکتشافی‌اند |',
        '| pseudo-holdout | سابقهٔ دسترسی UNKNOWN؛ پیش‌بینی تازه انجام نشد |',
        '| site/scanner و فایل PET | منشأ کامل تأیید نشده؛ UNKNOWN حفظ شده است |',
        '| XAI | گیت علمی رد شده؛ تأیید ناحیهٔ بیماری یا علیت به دست نیامده |',
        '| نشت تصویری | نامزدهای مشابه خودکار بیمار تکراری محسوب نمی‌شوند؛ موارد حل‌نشده بررسی انسانی لازم دارند |',
        '| پایداری | تغییر مصنوعی و ماسک جمعیتی؛ تأیید پایداری بین اسکنرها نیست |',
        '| تطبیق جمعیتی | فقط زیرجمعیت دارای هم‌پوشانی؛ هم‌خطی کامل cohort را رفع نمی‌کند |',
        '| عدم‌قطعیت | CI مشروط به مدل ثابت است؛ بذر و شرط تکراری بیمار مستقل نیستند |','',
        'این پیوست برای گزارش شفاف هم‌راستا با TRIPOD+AI و توجه به حوزه‌های خطر سوگیری PROBAST+AI است؛ ادعای انطباق کامل یا داوری مستقل ندارد.',
        'هیچ آزمونی برای مثبت‌شدن XAI یا بهبود AUROC تنظیم نشده است. نتایج نامطلوب به اندازهٔ نتایج مطلوب باید در پایان‌نامه گزارش شوند.',''])
    for section,text in (('methods',methods),('results','\n'.join(lines)),('discussion',discussion)):
        freeze_text(store.reports/f'mx7_quick_{section}_FA.md',text)
    inventory={p.relative_to(store.out).as_posix():sha(p) for p in sorted(store.out.rglob('*'))
               if p.is_file() and p.suffix in ('.json','.md') and p.name!='manifest.json'}
    status='COMPLETED_WITH_REVIEW_REQUIRED' if audit['near_duplicate_candidates'] else 'ALL_COMPLETED_POST_HOC_INTERNAL_EXPLORATORY'
    manifest={'status':status,'protocol_sha256':store.binding,'inventory':inventory,
              'external_validation':False,'xai_scientific_gate':parent['xai_scientific_gate'],
              'clinical_or_disease_specific_claim_allowed':False,'original_full_program_completed':False}
    immutable_json(store.reports/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('all','preflight','status'))
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--stop-after',choices=('audit','nonlinear','statistics','stress'))
    args=parser.parse_args(); root=args.project_root.resolve()
    require(root==ROOT.resolve(),'PROJECT_ROOT_MUST_MATCH_SCRIPT')
    config=read(root/CONFIG)
    out=root/config['output']
    if args.mode=='status':
        path=out/'reports/manifest.json'
        progress('status',status=read(path)['status'] if path.exists() else 'NOT_TERMINAL',
                 committed_receipts=sum(1 for _ in (out/'_private_resume').rglob('*.json')),
                 log=str(out/'reports/progress.log'))
        return 0
    from scripts.mx_fasttrack_v2_1.supervise_v21 import coordinator_lock, foreign_method_x_writers
    from scripts.mx_science_extension_v1.process_log import console_log
    from scripts.mx_quick_validation_v1.stress import load_mask,select_sample,benchmark,run_stress
    from scripts.mx_quick_validation_v1.audit import run_audit
    from scripts.mx_quick_validation_v1.tabular import run_tabular
    from scripts.mx_quick_validation_v1.paired_analysis import run_statistics
    require(not foreign_method_x_writers(root),'OTHER_METHOD_X_WRITER_ACTIVE')
    # Protect the completed source trees too: their original coordinators use
    # these same OS locks. All locks release automatically on interruption.
    with coordinator_lock(out/'reports'), coordinator_lock(root/EXT/'reports'), \
         coordinator_lock(root/PARENT/'reports'), console_log(out/'reports/progress.log'):
        cache_before=(root/CACHE).stat()
        parent=verify_parents(root)
        cache_after=(root/CACHE).stat()
        require((cache_before.st_size,cache_before.st_mtime_ns)==
                (cache_after.st_size,cache_after.st_mtime_ns),'CACHE_CHANGED_DURING_VERIFICATION')
        mask,mask_meta=load_mask(root,config)
        # Select sample from IDs/fold/labels only, before reading new predictions.
        rows=[{k:r[k] for k in ('patient_hash','fold','label')} for r in read(root/PANEL)['rows']]
        protocol=protocol_payload(root,config,parent,mask_meta)
        protocol['stress_sample']=select_sample(rows,config['stress_patients_per_fold'],config['bootstrap_seed'])
        lock=out/'_private_resume/protocol.json'
        if lock.exists():
            require(read(lock)==protocol,'PROTOCOL_INPUT_MASK_CHECKPOINT_OR_SOURCE_DRIFT_NEW_VERSION_REQUIRED')
        if args.mode=='preflight':
            progress('synthetic_cuda_benchmark',**benchmark(root))
            progress('preflight',status='PASS',scientific_new_outcomes_computed=False,
                     prior_mx5_outer_n=parent['mx5_outer_n'],xai_scientific_gate=parent['xai_scientific_gate'])
            return 0
        if not lock.exists():
            require(not any((out/'_private_resume').rglob('*.json')),'PROTOCOL_TOO_LATE')
        immutable_json(lock,protocol)
        store=Store(root,config,protocol)
        store.receipt('plan_reconciliation',lambda:parent,public=True)
        for name,run in [('audit',lambda:run_audit(store)),('nonlinear',lambda:run_tabular(store)),
                         ('statistics',lambda:run_statistics(store)),('stress',lambda:run_stress(store,mask))]:
            progress(name,status='RUNNING_OR_RESUMING')
            run()
            progress(name,status='COMPLETED')
            if args.stop_after==name:
                progress('supervisor',status='STOPPED_AT_REQUESTED_BOUNDARY',resume_command='same all command without --stop-after')
                return 0
        require(source_hashes(root)==protocol['source_sha256'],'SOURCE_CHANGED_DURING_RUN')
        cache_end=(root/CACHE).stat()
        require(cache_end.st_size==protocol['cache_bytes'] and cache_end.st_mtime_ns==protocol['cache_mtime_ns'],
                'CACHE_CHANGED_DURING_RUN')
        require(load_mask(root,config)[1]==protocol['mask'],'MASK_CHANGED_DURING_RUN')
        manifest=report(store)
        progress('supervisor',status=manifest['status'],manifest_sha256=sha(store.reports/'manifest.json'))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        progress('supervisor',status='INTERRUPTED_RESUMABLE',resume='Run exactly the same all command')
        raise SystemExit(130)
    except Exception as exc:
        progress('supervisor',status='BLOCKED',reason=f'{type(exc).__name__}:{exc}')
        raise SystemExit(3)
