# نمایه‌ی حافظه‌ی پروژه

## زنده‌ترین — ۲۰۲۶-۰۹-۰۴ عصر (تست resume FAIL-closed؛ master `be75` در برابر `d7e0`؛ بدون corruption)

0. **نتیجه تست resumability (به درخواست اپراتور):** اجرای آزمایشی بلافاصله `BLOCKED:RUN_CONTRACT_MISMATCH:MASTER` شد؛ هیچ epoch شروع نشد، GPU خالی، lock نیست، چک‌پوینت‌ها دست‌نخورده. این fail-closed درست است — retry بیهوده نکنید.
1. **ریشه:** master ذخیره‌شده `be75...` با ورودی‌های قفل فعلی بازتولید نمی‌شود (`d7e0...`؛ با باندل قدیمی هم `7cb9...`). migration `654f` آن را frozen-chain حفظ کرده بود؛ launcher آن را strict چک می‌کند. Preflight `READY` فقط global است.
2. **قدم بعدی:** migration حسابرسی‌شده master (`be75->d7e0`) با re-verify و provenance، سپس تست مجدد resume. تا آن زمان فرمان تولیدی را اجرا نکنید.
3. **وضعیت:** commit `c9945a2` (C2/C3 + docs) پوش نشده؛ آموزش متوقف؛ verifier 0/50؛ claim سقف `POST_HOC_INTERNAL_EXPLORATORY`.

## زنده‌ترین — ۲۰۲۶-۰۹-۰۴ (C1/C2/C3 رفع شد؛ preflight READY 654f؛ winner به master relabel؛ verifier 0/50)

0. **سه مانع resume رفع شد (بدون تغییر علمی):** C2 wrapper `_winner_target_digest` (تست 12/12 سبز)؛ C3 relabel `winner_A0_outer_0` از global `654f...` به master `be75...` با re-verify inner-IBS (`lr1e4_wd1e3`) و `read_verified` PASS؛ C1 wiring `binding_core` حفظ شد. Preflight `READY` با digest `654f8ce59934...` حتی با فایل‌های dirty. Verifier نهایی FAIL 0/50 (درست).
1. **وضعیت اجرا:** بدون writer/GPU فعال؛ `orchestration_state` stale `TRAINING_RUNNING`؛ نقطه resume = آخرین epoch اتمیک (A1 outer_0). تله‌متری: ~9.4s/epoch (A1) و ~13-14s (A0) → کل ≈۱۸-۳۰h. فرمان ادامه همان `run_method_y_image_factorial_v4.ps1` (تک‌نویسنده).
2. **ناهماهنگی سندی:** handover/memory قبلی digest `ec968b...` را زنده می‌دانست؛ زنده `654f...` است (migration `ec968->654f`). Provenance relabel در `governance/method_y_v4_winner_global_to_master_relabel_654f8ce5.json`.
3. **باقی‌مانده:** ۴۹ winner، ۵۰ outer exactly-once، گزارش‌های paired، privacy/claim gates، verifier 50/50. N واقعی amyloid 560/110 و tau 306/75؛ augmentation سوژه نمی‌سازد؛ claim سقف `POST_HOC_INTERNAL_EXPLORATORY`.

## زنده‌ترین — ۲۰۲۶-۰۹-۰۳ (قرارداد v4 کاملاً مستقل از HEAD/dirty؛ آموزش متوقف)

0. **دومین عود `RUN_CONTRACT_MISMATCH` ریشه‌یابی و رفع نهایی شد (`7628305`):**
   ریشهٔ کامل = `git_commit`/`git_provenance_status`/`scoped_dirty_digest` هنوز
   داخل `software_runtime_contract` بودند → digest با هر commit و حتی متن
   dirty-status زنده جابه‌جا می‌شد. اکنون digest فقط به bytes منابع bound
   (`bundle_sha256`) + نسخه‌های toolchain + CUDA + سیاست‌های قفل‌شده بسته است؛
   متادیتای git زیر یک بلاک `git_provenance` (INFORMATIONAL_ONLY) خارج از
   digest و خارج از برابری resume.
1. **ابزار migration با frozen-chain کامل شد:** digestهای قدیمی هر سلول از خود
   فایل بازمی‌گردند (بازمحاسبه با فرمول امروز ممکن نیست چون نسل‌های قبلی متن
   status زنده را embed کرده بودند)؛ winner_*.json فقط وقتی migrate می‌شود که
   `_reverify_winner_selection` انتخاب را از inner-IBS (one-SE+rank) بازتولیدپذیر
   اثبات کند؛ self-hash برنده + مسیر idempotent `REPAIR_ONLY_CURRENT`.
2. **Migration اعمال‌شده:** `a2013821→ec968b9b` (۱۸ checkpoint/۵۲ فایل/۱
   master/winner؛ وزن‌ها دست‌نخورده؛ بدون outer exposure). برندهٔ amyloid/A0/outer_0
   (`lr1e4_wd1e3`) بازتولیدپذیری‌اش اثبات شد.
3. **تأیید:** binding ۶/۶ + v4 suites ۴۸/۴۸ سبز؛ preflight READY حتی با فایل‌های
   dirty؛ digest فعال `ec968b9b1897...`.
4. **وضعیت اجرا:** آموزش تولیدی v4 به درخواست اپراتور **متوقف** است؛ نقطهٔ
   resume = آخرین epoch_checkpoint اتمیک (A1 در حال آموزش بود). فرمان ادامه همان
   `run_method_y_image_factorial_v4.ps1`.

## زنده‌ترین — ۲۰۲۶-۰۹-۰۲ (v3.2 VERIFY PASS + v4 factorial در حال اجرا)

0. **v3.2 (root v2) تکمیل و VERIFY شد:** `verify --level final` = **PASS**
   (۲۰/۲۰ سلول، exactly-once=۱، privacy PASS، inner = IPCW-IBS). نتایج: amyloid
   base 0.676C/0.097IBS → aug 0.695C/0.096IBS (dIBS CI شامل صفر → عدم‌برتری
   ثبت شد)؛ tau base 0.616C/0.149IBS → aug 0.706C/0.125IBS (بهبود داخلی؛ claim
   بسته). `positive/external=false`, POST_HOC/EXPLORATORY حفظ.
1. **رفع نقص provenance در verifier (`211dd49`):** readiness/final قبلاً
   software contract را با HEAD زنده مقایسه می‌کردند → هر commit بعدی نتیجهٔ
   finalize شده را RETRO-FAIL می‌کرد. اکنون readiness به قرارداد frozen داخل
   `run_contract.json` می‌بندد؛ final digest per-fold را از state.json
   self-hashed بازیابی می‌کند (frozen-chain). `audit_post_execution_drift`
   فقط اطلاع‌رسانی (INFORMATIONAL_ONLY). +۴ تست adversarial (۱۹/۱۹ سبز).
2. **v4 factorial (کد Codex، بازبینی این جلسه):** ۵ arm matched
   A0/A1(+aug)/A2(ResNet10-GN)/A3(+aug+R10)/A4(R18-GN@0.25)؛ aug ۱°/۰٫۵voxel
   فقط train (QC-locked، envelope ۲° رد، scale/PSF/noise بدون شاهد غیرفعال)؛
   انتخاب فقط inner IPCW-IBS. اجرا در amyloid/A0/outer_0/inner_1/cand۳/epoch13
   متوقف شد (توقف اپراتور) — resumable. تله‌متری ~۱۲-۱۵s/epoch → کل ≈۱۸-۲۸h.
   تعمیر device-mismatch + migration قراردادی (وزن‌ها ثابت) بازبینی شد.

> **SUPERSESSION 2026-08-26 — ممیزی ناظر نهایی Image v3.2.**

## Live 2026-08-26 - P0-P3 repair gate 10/12

Gate 12-sample equivalence running: 7/12 sequential receipts (PID 13276, wall ~5h, CPU ~5400s)
P0 freeze f72bc9f4 + superseded moves after DEAD proof 19216/1360
P1 dependency closure + limited commit
P2 morphology 3->7 fixed with scipy.ndimage, 14 repair tests green
P3 run_full reordered preflight-first, auto-flip removed
Digest 9b0f90a5... and battery 40a4a009... preserved

> گزارش `c42ef08/483aa81` بخش عمده‌ی تعمیرات را درست ثبت کرده بود، اما برای
> شروع امن آموزش کافی نبود. ممیزی نهایی این شکاف‌ها را نیز بست: closure کامل
> source/runtime contract، self-hash صحیح master state پس از resume، winner
> self-hashed و بازتولیدپذیر از inner-only evidence، بازیابی mirror پس از outer
> commit بدون دسترسی دوباره به test، تطبیق hash رکورد خصوصی/چک‌پوینت در verifier،
> و جلوگیری از جلو رفتن scheduler در AMP بدون optimizer-step. AMP تولیدی روی
> `bfloat16` با likelihood محاسبه‌شده در float32 قفل شده است. باتری زنده:
> `475 passed`، بدون failure و warning. digest قدیمی `7fa02033...` منسوخ است؛
> digest معتبر را همیشه از
> `results/method_y_revival_v3_image_v2/run_contract.json` بخوانید. آموزش تولیدی
> هنوز آغاز نشده و `final` تا تکمیل ۲۰/۲۰ سلول باید FAIL بماند. گزارش:
> `advisor-plans/agent-reports/method-y-image-v3-final-supervisor-audit-20260826.md`.

> **SUPERSESSION 2026-08-26 — v1 تصویری Method Y باطل شد.**
> همهٔ ادعاهای قبلیِ «true IBS»، «privacy PASS» و «DONE» برای
> `results/method_y_revival_v3_image_v1` **نادرست/ابطال‌شده** است. سند رسمی:
> `results/method_y_revival_v3_image_v1/provenance/INVALIDATION_20260826.json`
> (mirror: `governance/method_y_revival_v3_image_v1_invalidation_20260826.json`)
> با دلایل INNER_SELECTION_METRIC_FELL_BACK_TO_NLL / SOFTWARE_CONTRACT_NOT_BOUND /
> OUTER_EXACTLY_ONCE_WRAPPING_VIOLATION / PUBLIC_ROW_LEVEL_OUTCOME_LEAK.
> ریشهٔ فعال اکنون `results/method_y_revival_v3_image_v2` با
> `configs/method_y_image_training_v3_2.yaml` (deterministic CUDA) است؛
> run_contract جدید `7fa02033...`. fold_0 amyloid/scalar_tabular یک exposure
> تاریخی ثبت شده و از هر aggregation آینده خارج است. نتایج همچنان exploratory.

برای ادامه‌ی کار، **ابتدا دفتر ثبت نشست زنده** را بخوانید، نه snapshot تیرماه synthesis.

## زنده‌ترین — ۲۰۲۶-۰۸-۲۵ شب ۲ (رفع crash در resume اجرای تولیدی)

0. خطا کاربر: `STAGE_ORDER:cannot_move_backwards:GPU_SMOKE_PASSED->PREFLIGHT_DONE` در اجرای دوم launcher — گیت‌های preflight هر بار re-verify می‌شوند ولی ماشین مونوتیک رد می‌کرد.
1. رفع (`7a50e84`): `_REVERIFIABLE` در `image_launcher_state_v3.py` — این گیت‌ها در resume فقط facts را refresh می‌کنند و pointer عقب نمی‌رود؛ مراحل آموزشی/گزارشی همچنان backwards=fail-closed. پرش مجدد به stageهای branch تکمیل‌شده هم guard شد.
2. تأیید: +۲ تست RED-first؛ ۷/۷ state machine؛ اجرای زندهٔ محدود از وسط HPO amyloid رزومه شد، برندهٔ `lr3e4_wd1e4` (mean_ibs 0.689) انتخاب و fit نهایی outer_0 آغاز شد.

## زنده‌ترین — ۲۰۲۶-۰۸-۲۵ شب (حسابرسی دور ۳ پلن‌های 009-012 + شتاب 6.1x)

0. **F-R3-A:** امتیازدهی IBS درونی هنگام خطای load چک‌پوینت، بی‌صدا وزن‌های آخرین epoch را می‌سنجید — اکنون `load_locked_state_for_eval` در `src/method_y_revival/image_training_orchestrator_v3.py` برای missing/corrupt/contract-mismatch با `INNER_IBS_CHECKPOINT_LOAD_FAILED` fail-closed می‌شود.
1. **F-R3-B:** تحلیل‌های پلن 012 اجراکنندهٔ واقعی نداشتند و launcher فقط با وجود فایل OOF مرحله را COMPLETE می‌کرد — جدید `build_real_oof_reports()` در `image_reporting_v3.py` (calibration/subgroups/paired-bootstrap قفل روی 2000/incremental gate/IMAGE_CLAIM_POLICY + privacy scan) و launcher فقط بر اساس status اجراکننده مرحله می‌دهد، وگرنه `PENDING_REAL_OOF`.
2. **شتاب 6.1x بدون تغییر علمی:** cache رمِ lazy حجم‌ها در `H5PatternSurvivalDataset` (float16 دست‌نخورده، سقف 6GiB، fallback به دیسک)؛ حذف worker-spawn زیر IMAGE_V3_PERF؛ بنچمارک یکسان ۱۲۰ سوژه×۲epoch: **45.80s → 7.52s**، epoch پایا 26.6→1.9s (~14x)، GPU peak همان 265MB.
3. **تأیید:** تست‌های جدید RED 5/5→GREEN 5/5 (`test_audit_round3_v3.py`)؛ کل suite **450 passed**؛ smoke PASSED cuda 18MiB؛ preflight READY؛ گیت‌ها true/false/false.
4. گزارش: `advisor-plans/agent-reports/method-y-image-v3-hard-audit-round3-plans009-012-perf-20260825T160000Z.md`. دستور تولیدی همان PS1 است.

## زنده‌ترین — ۲۰۲۶-۰۸-۲۵ عصر (حسابرسی دور ۲: دو رفع بحرانی + state machine پلن 013)

0. **دو نقص بحرانی قبل از کامیت شدن نتیجهٔ جعلی رفع شد:** (۱) evaluator های outer در launcher مقدار هاردکد برمی‌گرداندند — جایگزین با ماژول واقعی `src/method_y_revival/image_outer_eval_v3.py` (`evaluate_locked_outer`: load سخت‌گیرانهٔ checkpoint با تطبیق digest، censoring فقط از outer-train، `integrated_brier_score` + `concordance_index_ipcw` از وزن‌های مدل؛ خروجی ثابت غیرممکن). (۲) lr/wd برندهٔ HPO به fit نهایی منتقل نمی‌شد — اکنون در candidate_records هست و `run_final_training` بدون آن‌ها `WINNER_HYPERPARAMS_MISSING` می‌دهد.
1. **state machine پلن 013 ساخته شد:** `src/method_y_revival/image_launcher_state_v3.py` — ۱۶ مرحله self-hashed و contract-bound؛ launcher مراحل PREFLIGHT..GPU_SMOKE_PASSED را با facts جلو می‌برد، per-branch stageها را روی تکمیل کامیت می‌کند و ROBUSTNESS/REPORTS فقط با OOF واقعی PASS می‌شوند وگرنه `PENDING_REAL_OOF`.
2. **نتایج دورهٔ کد معیوب حذف شد (به دست کاربر):** `orchestration/{amyloid,tau}/**` و `.smoke_test/**` زیر root اصلی؛ lock یتیم بعد از تأیید مرگ pid حذف شد. هیچ outer_result جعلی قبلاً کامیت نشده بود.
3. **تأیید:** تست‌های جدید RED(۱۰ fail)→GREEN(۱۰ pass)؛ کل `tests/method_y_revival` **445 passed**؛ smoke PASSED cuda 26MiB؛ preflight READY؛ اجرای زندهٔ محدود خطوط STAGE_COMMITTED را چاپ کرد و وارد HPO amyloid شد.
4. گزارش کامل: `advisor-plans/agent-reports/method-y-image-v3-hard-audit-round2-20260825T144330Z.md`. دستور تولیدی همان است: `powershell -NoProfile -ExecutionPolicy Bypass -File 'H:\New project\scripts\run_method_y_image_training_v3.ps1'`

## زنده‌ترین — ۲۰۲۶-۰۸-۲۵ بعدازظهر (image v3: فعال‌سازی آموزش کامل + بهینه‌سازی GPU + تحقیق 10×)

0. **آموزش کامل فعال شد** (commit `cd6a65f`): `scripts/run_method_y_image_training_v3.py` حالت `run` اکنون حلقه واقعی nested-CV را اجرا می‌کند — amyloid سپس tau × دو variant قفل‌شده × outer_0..4؛ `src/method_y_revival/image_training_orchestrator_v3.py` با پارامتر جدید `h5_cache_path` از `H5PatternSurvivalDataset` روی cache واقعی می‌خواند و امتیاز HPO را با IBS واقعی (`survival_at_times` + `integrated_brier_score`) می‌سنجد نه NLL. رزوم سه‌لایه: winner JSON → `locked_checkpoint.pt` → state کنترلر outer (`OUTER_EVAL_COMMITTED/FINALIZED`).
1. رفع خطای resume دود (`c514df4`): `_restore_rng` در `src/method_y_revival/image_training_runtime_v3.py` اکنون tensorهای rng را قبل از `set_rng_state` به CPU برمی‌گرداند — خطای «RNG state must be a torch.ByteTensor» هنگام resume با `map_location=cuda` برطرف؛ smoke دوم پاس (`peak 18 MiB` در `results/method_y_revival_v3_image_v1/.smoke_test/`).
2. رفع `RUN_CONTRACT_MISMATCH` شروع آموزش واقعی (`97dec14`): launcher فقط برای همین خطا، state ناقص placeholder-era را حذف و با digestهای واقعی بازسازی می‌کند؛ mismatch واقعی config/cache/split همچنان fail-closed.
3. بهینه‌سازی بدون تغییر علمی (`eaf6d57` + `41d1c7f`): چاپ ترمینال فقط هر ۴۰ ثانیه (+اولین/آخرین epoch)؛ TF32/high همیشه، `cudnn.benchmark` فقط با `IMAGE_V3_PERF=1` در `scripts/run_method_y_image_training_v3.ps1`؛ `batch 2→4 / grad_accum 4→2` (effective batch همان ۸) در `configs/method_y_image_training_v3_1.yaml`؛ `num_workers=2` خودکار برای دیتاست HDF5؛ `channels_last_3d` برای مدل و ورودی روی cuda. مدل/loss/split/HPO/exactly-once دست نخورده. `435 passed`.
4. تحقیق 10×: run کامل hyperresearch در `research/runs/gpu-3d-cnn-10x-9a1e13/` (search-plan با ۳۸ کوئری) و گزارش `research/notes/final_report_gpu-3d-cnn-10x-9a1e13.md`. نتیجه: 10× تک-GPU بدون شکستن منطق علمی ممکن نیست؛ سرعت فنی ترکیبی ≈3.8× و نسبت به سناریوی بدبینانه ۹۰h ≈9× مؤثر (early stopping میانگین ~۱۲ epoch از ۴۰). ردشدگان به‌دلیل تغییر علمی: DDP تک-GPU، patch 64³، pruning/compression، gradient checkpointing، torch.compile روی Windows (triton ناموجود — تأیید آزمایشی).
5. اندازه‌گیری واقعی: ۵۶۰ سوژه ۱۲۸³ ≈۴۰ ثانیه/epoch (هر دو حالت batch2/ga4 و batch4/ga2) → برآورد کل برنامه ≈۲۰ ساعت wall-time (amyloid ≈۱۲h، tau ≈۸h)؛ هر لحظه قابل قطع و رزومه با همان دستور.
6. جزئیات کامل و آدرس‌دهی commitها (`cd6a65f`, `c514df4`, `97dec14`, `eaf6d57`, `41d1c7f`): `docs/plans/agent_session_ledger_20260815.md` بخش Append 2026-08-25T16:00Z.

## زنده‌ترین — ۲۰۲۶-۰۸-۲۵ صبح (Method Y image v3 — Plans 009-013 + hard audit)

0. شاخه تصویری Method Y v3 اکنون **READY_FOR_FINAL_INTERNAL_IMAGE_TRAINING** (`results/method_y_revival_v3_image_v1/readiness/FINAL_IMAGE_TRAINING_READINESS.json:6` `19baa31d0acfbb15e8bcc11828bae8bee2b9e0ed3b1beea06f7a1119a74979c5`)؛ `training_execution_allowed true`، `positive_claim_allowed false`، `external_claim_allowed false`.
1. Cacheهای pattern v3 کامل و provenance-bound روی H: `caches/method_y/revival_v3_image/amyloid_pattern_128_v1.h5` 895 گروه (31981B33...), `tau_pattern_128_v1.h5` 306 گروه (4095FCBC...), هر گروه `COMPLETE` + `finite_fraction 1.0` + `float16` + `source_sha256` hard-checked (`src/method_y_revival/image_cache_v3.py:76` strict).
2. Trainer تو در تو: `src/method_y_revival/image_modeling_v3.py:24` `survival_at_times` و `src/method_y_revival/image_training_orchestrator_v3.py:12` با انتخاب winner فقط `mean_inner_IBS` + 1-SE + `complexity_rank`، outer exactly-once via `OuterFoldController:48`.
3. Robustness/reporting: `src/method_y_revival/image_robustness_v3.py:12` و `image_reporting_v3.py:12` — scanner support ≠ robustness proof، balanced weights از train-only، leave-manufacturer-out بدون pooling، privacy scan PASS.
4. Launcher واحد: `scripts/run_method_y_image_training_v3.ps1:1` و `scripts/run_method_y_image_training_v3.py:118` با preflight/smoke/run، `WriterLock`, H-only temp, GPU smoke PASSED (cuda 26 MiB), `435 passed` در `tests/method_y_revival`.
5. Hard audit 2026-08-25T13:30Z: `advisor-plans/agent-reports/method-y-image-v3-hard-audit-20260825T133000Z.md` — سه رفع (strict HDF5 provenance, source_artifact binding, true IBS, smoke)؛ همه تست‌ها سبز.

## زنده — ۲۰۲۶-۰۸-۲۵ (BrainIAC-CAFT governance drift fix و باتری ترمینال)

0. باتری پنج‌مدلی اکنون **TERMINAL** (`40a4a009b3a2453fb953e2b2e9b9ec789bdf896741dc2278a898a3b38fdb971d`، 5/5 `COMPLETED`، `.battery.lock` ABSENT) — `src/research/brainiac_battery_terminal.py: verify_battery_terminal` `terminal:true`.
1. `method_x_training_source_contract()` جدید `9b0f90a5a14f8764397c6a15ff8016462f0872e2f827b2ec47fe54ab773871b3` (پس از سازگار‌سازی `src/evaluation/calibration.py:49`, `cohort_probe.py:128`, `backbone_factory.py:6` و `scripts/train_3d_mri.py:57`).
2. Governance fix 4 مرحله‌ای 2026-08-25 (predecessor `e3dd7aaf`/`6b313303`, registry `1c45b3f4`, vendor CRLF→LF `c26b42ba`) — `scripts/research/prepare_brainiac_raw_t1_v2.py --mode preflight` اکنون `PASS` و `PREPARE_RAW_T1 allowed:true`.
3. دستور واحد P003 (resumable, Brainiac venv): `& 'H:\New project\.venvs\brainiac\Scripts\python.exe' 'H:\New project\scripts\research\prepare_brainiac_raw_t1_v2.py' --mode full --device cuda` — اجرای مجدد همان دستور از last commit ادامه می‌دهد (exit 75). **2026-08-25 شب:** full اکنون SELF-GATE است؛ اگر sequential_only=true و artifact PASS نباشد ابتدا gate هم‌ارزی ۱۲نمونه‌ای قطعی را اجرا می‌کند، FAIL=>exit 3، PASS=>فلپ خودکار policy + آرشیو pre-pin threshold/calibration و سپس اجرای موازی ≥2x (commits 239319e, 884197a؛ tests 11/11).
4. مستندات به‌روز: `advisor-plans/README.md` و `advisor-plans/000-20260822-work-log-and-live-state.md` Addendum 3، `docs/plans/method_x_project_roadmap.md` و `results/method_x_registry/project_roadmap.json` به TERMINAL.

## زنده‌ترین — ۲۰۲۶-۰۸-۲۴ (Method Y تکمیل داخلی با ادعاهای محدود)

0. رسید نهایی ماشینی: `results/method_y_revival_v2/reports/POSTTRAINING_READINESS_V2.json` — وضعیت `LOCAL_INTERNAL_PIPELINE_COMPLETED_CLAIMS_RESTRICTED`، بازبینی مستقل انسانی لازم نیست.
1. گزارش نهایی قابل‌خواندن: `docs/METHOD_Y_V2_FINAL_INTERNAL_STATUS_20260824.md`.
2. قرارداد کامل اجرا و resume: `docs/METHOD_Y_V2_POSTTRAINING_EXECUTION_CONTRACT.md`.
3. رجیستری artifact: `results/method_y_revival_v2/artifact_registry_v2.json`.
4. بسته عمومی aggregate: `results/method_y_revival_v2/public_bundle/`؛ privacy status باید `PASS` بماند.
5. تفسیر: source-regional کامل است؛ image branches اختیاری و مسدود؛ external primary اختیاری و در داده‌های کاملاً باز یافت نشد؛ DLBS فقط replication ثانویه اکتشافی است.

## زنده‌ترین — ۲۰۲۶-۰۸-۲۲ (نشست نظارت ZCode)

0. **حافظه اجرای پس از آموزش (معتبرترین صف POST):** `results/method_x_registry/post_training_execution_memory_20260822.json` (schema v3؛ کارت‌های D02/MX-03/R1a/R1b/R1c/MX-04/MX-05/MX-06/T08/MX-07 + قرارداد watchdog؛ پایش خودکار ۲ساعته فعال)
1. وضعیت BrainIAC: `results/method_x_registry/BRAINIAC_P08_STATUS_20260822.json` (R0 = COMPLETE با ۲۸/۲۹ مرور فول‌تکست؛ اجرای R1 همچنان HARD_DISABLED با ۶ blocker) و `research/brainiac_caft_20260808/evidence_registry_v3.json`
2. چک‌لیست جاری: `tasks/method_x_completion_20260821_todo.md` (P08b تکمیل شد؛ D01 زنده؛ D02 در انتظار ترمینال MedNeXt)
3. دفتر کار Method X (append-only): `docs/plans/method_x_work_log.md` — ورودی کامل ۲۰۲۶-۰۸-۲۲ بعدازظهر در انتها

## زنده — ۲۰۲۶-۰۸-۱۵

1. دفتر ثبت نشست (Method Y repair + Method X untouched): `docs/plans/agent_session_ledger_20260815.md`
2. همزاد ماشینی Method Y: `results/method_y_revival_v2/journals/session_ledger_20260815.json`
3. یادداشت انزوای Method X: `results/method_x_registry/SESSION_NOTE_METHOD_Y_DEFECT_REPAIR_20260815.json`
4. آمادگی زنده Method Y: `results/method_y_revival_v2/readiness/PRETRAINING_READINESS_V2.json` (`training_allowed=false`؛ Task 4 الان `COMMITTED` است ولی این فایل رسید اجرای A4 نیست؛ دود مصنوعی `SYNTHETIC_SMOKE_ONLY` در ۲۰۲۶-۰۸-۱۶)
5. کار باقی‌مانده Method Y: `docs/plans/method_y_remaining_work_20260815.md`
6. نقشه زنده Method X: `docs/plans/method_x_project_roadmap.md` و `results/method_x_registry/project_roadmap.json`
7. کار باقی‌مانده Method X: `docs/plans/method_x_remaining_work_20260815.md`
8. دفتر کار Method Y (append-only): `docs/plans/method_y_work_log.md`
9. دفتر کار Method X (append-only): `docs/plans/method_x_work_log.md`

## برنامه و قفل‌های علمی (تاریخچه، نه وضعیت GPU)

10. منبع کامل برنامه: `docs/PROGRAM_DEFAULT_SOURCE_OF_TRUTH_2026-07-19.md`
11. ADR تصمیم پیش‌فرض: `docs/decisions/ADR-2026-07-19-default-method-xy-program.md`
12. تصمیم اجرایی synthesis: `supervisor_synthesis_5agents/08_final_outputs/00_FINAL_EXECUTIVE_DECISION.md`
13. مشخصات کامل ۴۱‌بندی: `supervisor_synthesis_5agents/08_final_outputs/07_SELECTED_PROJECT_SPECIFICATION.md`
14. قرارداد transform: `results/method_y_revival_v2/transforms/eligibility_lock.json` و `results/method_y_revival_v2/transforms/qc_audit.json`

## کهنه — اولویت ندهید

- `supervisor_synthesis_5agents/08_final_outputs/01_CURRENT_PROJECT_STATE.md` — snapshot ۲۰۲۶-۰۷-۱۸؛ HEAD آن `480752d` است.
- `docs/method_x_live_progress_report.md` — هنوز مدل‌ها را `NOT_STARTED` می‌نویسد؛ با باتری زنده در تضاد است.

این نمایه جایگزین فایل‌های کامل نیست. اگر بین این نمایه، گزارش کهنه، و artifact دارای hash/contract اختلاف بود، artifact زنده و دفتر ثبت نشست ۲۰۲۶-۰۸-۱۵ بر snapshot تیرماه مقدم است.

<!-- BRAINIAC_PREPROCESSING_PLAN018_LIVE_START -->
## BrainIAC preprocessing — generated live status (Plan 018)

- R0: `R0_COMPLETE`; citation: `CITATION_CORROBORATION_PARTIAL`; novelty: `NOVELTY_CANDIDATE`.
- Five-CNN battery: `VERIFIED_TERMINAL`.
- Plan 002: `ACCEPTED_READY_FOR_PRIMARY`; Plan 003: `NOT_STARTED_PRIMARY_0_OF_1669`.
- Plans 014–017: `DONE/DONE/DONE/PASS`; Plan 019 may start only when final readiness validates live.
- Plans 005–007: `NOT_EXECUTED / CODE_ONLY_UNACCEPTED / NOT_EXECUTED` and remain hard-disabled until `1669/1669`.
- Tests: `596 passed`, `0 failed`, `0 errors` (live manifest-bound run).
- External data: `OPTIONAL_UNAVAILABLE_NONBLOCKING`; internal substitute is `ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION`.
- Claim ceiling: `POST_HOC_INTERNAL_TECHNICAL_EVIDENCE`.
- Runtime counts must be read from `prepare_brainiac_raw_t1_v2.py --mode status`; no PID/count in this section is treated as timeless.
<!-- BRAINIAC_PREPROCESSING_PLAN018_LIVE_END -->
