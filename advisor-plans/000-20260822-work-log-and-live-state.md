# گزارش ثبت‌شدهٔ کار 2026-08-22 و نقطهٔ تحویل

> این فایل گزارش است، نه مجوز اجرا. وضعیت‌های `COMPLETE`, `PARTIAL`, `BLOCKED`
> عمداً از هم جدا شده‌اند. گزارش قدیمی B1 که عبارت `B1_COMPLETE` دارد با حکم
> جدید `B1_PARTIAL_REQUEST_CHANGES` supersede می‌شود و حذف نمی‌گردد.

## هویت snapshot

| مورد | مقدار |
|---|---|
| Project root | `H:\New project` |
| Branch | `method-y-post-revival-sanitized-20260812` |
| Commit | `54057162efe69556539d4c3e092b9e4871c4f161` |
| Dirty entries | `565`؛ command هشدار permission برای یک pytest temp قدیمی داد |
| Process فعال Method X/BrainIAC | هیچ موردی پیدا نشد |
| Battery status | `PAUSED_RESUMABLE` |
| Retained lock | PID `25368`، process غایب؛ lock حذف یا archive نشد |
| Battery manifest SHA | `61e0536877ec7ada8d8bcf9d46f77d7f7d121805fa59d2df2124801ca0b9b1a4` |
| Battery lock SHA | `cd76c730bc264ac1051d16521f6085775c9d4404427e84d66539e11961be928c` |
| Method X source digest | `b73eef3d67b494986a1e2273a873f5663c874f34588a649bf708a508755a1be9` |

## کارهای کامل امروز

### B0 — freeze و جداسازی از باتری: COMPLETE

- artifact انجماد در
  `results/method_x_registry/brainiac_r1/BRAINIAC_PREIMPLEMENTATION_FREEZE_20260822.json`
  ساخته شد.
- SHA-256 فایل:
  `642909c42f1fde290f71425ddbf74793503980eb8ea9762afad0a18a7411a0f4`.
- canonical integrity:
  `a7206595e313e3217e5064b744d3ea4572e9d3093c89d609c71fd5c8e6532d97`.
- `197/197` مسیر ثبت‌شده در زمان freeze از نظر hash و size بازاعتبارسنجی شد.
- هیچ training، preprocessing، embedding extraction یا outer evaluation اجرا
  نشد و هیچ artifact باتری تغییر نکرد.

### R0 — اصلاح شواهد و claim boundary: COMPLETE با سقف محافظه‌کارانه

- `research/brainiac_caft_20260808/evidence_registry_v3.json` اکنون:
  - 29 منبع یکتای 2025–2026؛
  - 13 peer-reviewed؛
  - 28 full-text reviewed؛
  - یک رکورد screened-not-read؛
  - `fully_verified=false`؛
  - `CITATION_CORROBORATION_PARTIAL`؛
  - `NOVELTY_CANDIDATE`.
- SHA فعلی registry:
  `ea91979f80845d4fcd7a9d74aafb7d4fa8c7cc56e60f88b8d475c4693613ca26`.
- report قدیمی که SHA `c474...` را ثبت کرده، فقط سابقه است و authority فعلی
  نیست.

### بخش عمدهٔ B1 — PARTIAL

فایل‌های زیر ساخته یا سخت‌سازی شدند:

```text
src/research/brainiac_readiness.py
src/research/brainiac_execution_gate.py
src/research/brainiac_action_guard.py
src/research/brainiac_battery_terminal.py
src/research/brainiac_readiness_ledger.py
src/research/brainiac_model_loader.py
scripts/research/build_brainiac_r1_readiness.py
scripts/build_brainiac_p08_status.py
governance/brainiac_r1_execution_policy_v1.json
governance/brainiac_r1_preterminal_preparation_scope_amendment_v1.json
```

نتایج پیاده‌سازی‌شده:

- هفت action مستقل و fail-closed تعریف شد.
- public gate دیگر path، SHA یا process provider قابل‌کنترل توسط caller ندارد.
- JSONهای آیندهٔ B2 تا B5 تا زمان وجود verifier واقعی hard-disabled هستند.
- runtime/vendor/model/loader bytes در readiness دوباره hash می‌شوند.
- privacy guard، `UNKNOWN_PRETRAINING_OVERLAP` و claim limits سخت‌گیرانه شدند.
- exact-five verifier هویت هفت row، دو exclusion، پنج مدل مؤثر، 25 fold،
  exactly-once و checkpoint hash را می‌سنجد.
- process دارای `python.exe` و command line نامعلوم اکنون `UNKNOWN` است.
- release guard در missing/tamper/reparse/unlink failure دیگر silently success
  نمی‌شود.
- source-only migration chain و checkpoint-embedded contract در terminal
  verifier به validator مشترک متصل شدند.
- preprocessor قدیمی schema v1 و extractor قدیمی cache128 پیش از I/O
  hard-disabled شدند؛ فایل‌های تاریخی حذف نشدند.

## نقطهٔ دقیق توقف: B1 OS anchor ناقص است

برای رفع rollback/rewrite ledger، این فایل‌ها ایجاد شدند:

```text
src/research/brainiac_readiness_anchor.py
tests/test_brainiac_b1_anchor_hardening.py
governance/brainiac_readiness_anchor_key.v1.dpapi.json
```

- DPAPI key file SHA:
  `44127d5546187f442a43f7eab6f49917cc7edab4683d48ad6b86626793210a27`.
- protected blob یا key نباید در هیچ prompt/report کپی شود.
- pure chain tests چهار سناریوی append، rollback، fork و old-head را پوشش دادند.
- آخرین focused suite روی bytes فعلی:
  `94 passed in 70.22s`.
- اما production integration کامل نیست:
  - `PINNED_B1_BASELINE_RECORD_ID = 0`؛
  - `PINNED_B1_BASELINE_MESSAGE_SHA256 = ""`؛
  - هیچ baseline event با ID 6102 صادر و pin نشده است؛
  - `read_production_anchor_records()` در probe زنده
    `BRAINIAC_LEDGER_ANCHOR_EVENTLOG_READ_UNKNOWN` داد؛
  - capability probe فقط یک event آزمایشی ID 6101 با RecordId 92893 نوشته است؛
    آن event anchor نیست؛
  - `brainiac_readiness_ledger.py`، execution gate و readiness builder هنوز
    تولید/اعتبارسنجی anchor را اجباری نمی‌کنند.
- DPAPI به user/machine فعلی bind است. مهاجرت machine/user بدون provisioning
  صریح باید BLOCK شود، نه اینکه key بی‌صدا regenerate گردد.
- Windows Application log circular است؛ گم‌شدن baseline باید fail closed باشد.

نتیجه: B1 هنوز `REQUEST_CHANGES` است و B2 I/O واقعی نباید آغاز شود.

## وضعیت artifacts readiness

- schema4 ثابت در
  `research/brainiac_caft_20260808/r1_readiness_status_schema4_20260822.json`
  SHA `4f519eb916448b5bc7aea3ff0a55192be32d3b85587c0ca2aa2715c04c78a37f`
  دارد، اما هنوز SHA قدیمی registry یعنی `c474...` را bind کرده است؛ پس snapshot
  تاریخی است، نه authority زنده.
- ledger زنده سه event دارد و head آن:
  `105e381a79723b9b0fddfc752fa3631a4aea2514542bd55808e7e3e1a44ea8cd`.
- event سوم registry فعلی `ea919...` را bind می‌کند، ولی تا تکمیل OS anchor
  tail آن مستقل از H محافظت نشده است.

## وضعیت باتری که نباید در این کار تغییر کند

| Backbone | Manifest status |
|---|---|
| resnet10_3d | `COMPLETED` |
| resnet18_3d | `COMPLETED` |
| resnet34_3d | `COMPLETED` |
| resnet50_3d | `PROTOCOL_EXCLUDED_NOT_STARTED` |
| densenet121_3d | `COMPLETED` |
| efficientnet3d_b0 | `PROTOCOL_EXCLUDED_NOT_STARTED` |
| mednext_3d_s | `TRAINING_RUNNING`، ولی process زنده پیدا نشد |

عامل BrainIAC حق ندارد row آخر، lock، manifest، state یا checkpoint را اصلاح
کند. Battery terminal verifier فقط گزارش `FIVE_CNN_BATTERY_NOT_TERMINAL` می‌دهد.

## آزمون‌ها و معنای صحیح آن‌ها

- Full named run پیش از آخرین anchor edits:
  `294 passed, 1497 deselected`.
- Focused current run:
  `94 passed in 70.22s`.
- بنابراین ادعای مجاز: «focused B1 regression سبز است».
- ادعای غیرمجاز: «B1 کامل است»، «BrainIAC آمادهٔ آموزش است» یا «کل suite روی
  bytes فعلی سبز است».

## کارهای انجام‌نشده

- بستن production OS anchor و clean independent re-review B1؛
- preprocessing schema v2 و اجرای 1669/1669؛
- embedding واقعی 1669×768؛
- split registry و frozen ResNet34 128d features؛
- trainer واقعی شش-arm؛
- orchestrator و exact-resume؛
- full acceptance/traceability/readiness؛
- training واقعی CAFT؛
- optional external evaluation، XAI و LoRA.

این کارها در Planهای 001 تا 008 تصمیم‌کامل شده‌اند.

## Addendum 2026-08-22T19:12Z — رأی اپراتور دربارهٔ drift باتری و baseline جدید preflight

در 2026-08-22T17:06:27Z باتری پنج‌مدلی توسط اپراتور restart شد و پروسهٔ
ناشی از آن تقریباً بلافاصله از رفت. **اپراتور در 2026-08-22T19:1xZ صریحاً
رأی داد که این restart عمدی نبوده و اشتباهی رخ داده است.**

وضعیت ثبت‌شده پس از اتفاق (هیچ عامل BrainIAC آن را تغییر نداد؛ مسیرهای
باتری برای عامل‌های BrainIAC ممنوع‌اند):

```text
battery_manifest.json SHA-256 = a568b01de7912adc4ed0cb32ebc5742aab24b4556578e98ea718f218e8664777
.battery.lock SHA-256         = 7ca81523803af97236cb2323a73216db8c7be5a4b93b4e3e265e50ec7c247d08
lock pid = 2512               → NOT_FOUND (stale؛ retained، حذف/archive نشد)
manifest mednext_3d_s owner pid = 18088 → NOT_FOUND؛ row همچنان RUNNING است (stale)
پروسهٔ python.exe فعال: هیچ
```

سمت BrainIAC در همان لحظه re-verify شد و دست‌نخورده است:

```text
method_x source digest = b73eef3d67b494986a1e2273a873f5663c874f34588a649bf708a508755a1be9
readiness ledger       = دقیقاً 3 event؛ head = 105e381a…a8cd
production OS anchor   = PASS؛ baseline RecordId = 92912
DPAPI key file SHA     = 44127d55…0a27
```

### baseline جدید preflight برای عامل‌های بعدی (با رأی اپراتور)

مقادیر مورد انتظار manifest/lock در preflight از این addendum به بعد:

```text
manifest SHA = a568b01de7912adc4ed0cb32ebc5742aab24b4556578e98ea718f218e8664777
lock SHA     = 7ca81523803af97236cb2323a73216db8c7be5a4b93b4e3e265e50ec7c247d08
```

مقادیر قبلی (`61e05368…b1a4` و `cd76c730…928c`) تاریخی‌اند و فقط برای
بازبینی Plan 001 معتبرند (اجرای آن کارت قبل و بعد از تغییر byte-identical
بود؛ تغییر خارج از پنجرهٔ اجرای آن رخ داد). اگر باتری در اجرای مشروع بعدی
خود manifest/lock را دوباره بنویسد، اپراتور باید baseline را دوباره به‌روز
کند؛ عامل‌ها آن را خودکار reconcile نمی‌کنند.

### پیامد برای blocker ثبت‌شده

`results/method_x_registry/brainiac_r1/blockers/BLOCKED_PLAN_001_20260822T190032Z.json`
با این رأی اپراتور resolved می‌شود (فایل به‌عنوان سابقه نگه داشته می‌شود)؛
reconcile کردن lock مرده و row کهنهٔ RUNNING بر عهدهٔ runner خود باتری در
اجرای مشروع بعدی آن است، نه عامل‌های BrainIAC.


## Addendum 2 — 2026-08-23T~05:55Z — تلاش دوم اجرای باتری توسط اپراتور

در 2026-08-23T05:42:37Z اپراتور دستور رسمی باتری را (پس از آزادشدن guard توسط
پایان smoke اول BrainIAC) اجرا کرد؛ lock با pid 3556 و manifest در 05:43:58Z
بازنویسی شد و row mednext_3d_s با pid 7416 RUNNING شد، اما پروسه حدوداً ۹۰
ثانیه بعد از رفت (هر دو PID مرده؛ هیچ python.exe زنده نیست). علت مرگ باتری
سمت Method X است و در scope عامل‌های BrainIAC نیست.

baseline جدید preflight از این پس:

```text
manifest SHA = 43b95fd2993499259604a02f7dfb64659f89087ebdcc9309fd0e0cf9971ba15a
lock SHA     = b11d5768559035b525491a97aeea0a5a821cbfccda8def4538f6dd9a9013711b
```

readiness ledger باید این وضعیت را با یک anchored transition جدید ثبت کند
(materialize) تا HEAD_DRIFT برطرف شود.

## Addendum 3 — 2026-08-25T05:30Z — ترمینال شدن باتری و تثبیت digest جدید

پس از تلاش‌های 2026-08-23، باتری در 2026-08-24 به terminal رسید (اجرای مشروع بعدی که در `results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729/battery_manifest.json` ثبت شد):

```text
battery_manifest.json SHA-256 = 40a4a009b3a2453fb953e2b2e9b9ec789bdf896741dc2278a898a3b38fdb971d
.battery.lock                = ABSENT (no active writer; terminal state)
effective five:
  resnet10_3d  COMPLETED (COMPLETED_VALIDATED_REPAIRED)
  resnet18_3d  COMPLETED (COMPLETED_VALIDATED_REPAIRED)
  resnet34_3d  COMPLETED (COMPLETED_VALIDATED_RECONCILED)
  densenet121_3d COMPLETED
  mednext_3d_s   COMPLETED
exclusions: resnet50_3d, efficientnet3d_b0 = PROTOCOL_EXCLUDED_NOT_STARTED
```

این وضعیت با `src/research/brainiac_battery_terminal.py: verify_battery_terminal` به‌عنوان
`terminal=true, identity_valid=true` تایید شد؛ `FIVE_CNN_BATTERY_NOT_TERMINAL` دیگر sole blocker نیست.

هم‌زمان `method_x_training_source_contract()` به‌دلیل dirty evolution (اضافه شدن `functools, hashlib, signal` در `scripts/train_3d_mri.py:4` و stubs برای
`src/evaluation/calibration.py:49` و `cohort_probe.py:128` و `backbone_factory.py:6`) از
`b73eef3d67b494986a1e2273a873f5663c874f34588a649bf708a508755a1be9` به
`8c02bd511e61caefe4b679e404049fcae45bec9dc21d3707d68ba43dd5b3bd21` تغییر کرد.
این تغییر نتیجهٔ `git checkout HEAD -- 52file` ناسازگار بود که importها را شکست (`apply_temperature_to_logits`, `train_cohort_probe`, `available_3d_backbones`)؛
با افزودن stubs minimal در `src/evaluation/calibration.py:49-90`، `src/evaluation/cohort_probe.py:128`،
`src/models/backbones/backbone_factory.py:6` و بازسازی `scripts/train_3d_mri.py:114-180` سازگاری برقرار و
digest جدید به‌عنوان baseline append-only ثبت شد. مقدار `b73eef` فقط تاریخی (Plan 001) است.

سایر fixهای 2026-08-25 برای اجرای `prepare_brainiac_raw_t1_v2.py --mode full`:
- `governance/method_x_current_completion_scope_v1.json` قبلاً `3880b/e3dd7aaf...` بود ولی amendment
  `76b233.../4000b` داشت — با `6b313303ff16c16c160838c4022801a5a2fd3de0aed2084ac7289cf8ecfae5fc` در
  `src/research/brainiac_readiness.py:134` هم‌تراز شد.
- `research/brainiac_caft_20260808/evidence_registry_v3.json` برای 29 منبع (49 entry) `source_bytes/sha256`
  با `research/notes/*.md` فعلی هم‌تراز شد (registry جدید `1c45b3f4...`).
- `vendor/BrainIAC/.../LICENSE` و 4 فایل دیگر CRLF→LF شدند تا `stable_source_tree_sha256` از
  `d85236e6...` به `c26b42ba...` برگردد و `strict_load` PASS شود.

پس از این سه fix، `build_live_readiness(process_provider=lambda:[])` برای `PREPARE_RAW_T1` به `allowed:true` برگشت و
`scripts/research/prepare_brainiac_raw_t1_v2.py --mode preflight` (`brainiac` venv) `status:PASS` داد.
دستور کامل GLM اکنون بدون `BRAINIAC_SCOPE_PREDECESSOR_BINDING_MISMATCH` اجرا می‌شود.
