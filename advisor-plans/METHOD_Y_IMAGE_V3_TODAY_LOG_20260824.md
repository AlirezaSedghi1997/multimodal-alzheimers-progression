# گزارش ثبت‌شدهٔ کار امروز — Method Y image training v3

**تاریخ محلی:** 2026-08-24 (Asia/Tehran)  
**مخزن زنده:** `H:\New project`  
**شاخه:** `method-y-post-revival-sanitized-20260812`  
**commit مبنا:** `54057162efe69556539d4c3e092b9e4871c4f161`  
**نوع ثبت:** evidence-backed work log؛ این فایل به‌تنهایی مجوز claim یا training نیست.

## 1. تصمیم علمی ثبت‌شده امروز

برای شاخهٔ تصویری Method Y دو gate مستقل تعریف شد:

```text
training_execution_allowed
  فقط integrity فنی/علمی نمونه، endpoint آینده‌نگر، split بدون leakage،
  tracer صحیح، geometry معتبر، finite بودن تصویر و event support را می‌سنجد.

positive_claim_allowed
  علاوه بر موارد بالا، quantitative equivalence، scanner robustness،
  incremental value و در صورت ادعای خارجی external validation قفل‌شده را می‌خواهد.
```

بنابراین نبودِ اثبات رسمی Centiloid/CenTauR، robustness اسکنر و cohort خارجی
جلوی **آموزش داخلی اکتشافی** را نمی‌گیرد، اما همچنان جلوی ادعای مثبت، کمی،
تشخیصی و تعمیم خارجی را می‌گیرد. سه scan دارای scaling نامعتبر، tracer اشتباه،
transform/geometry نامعتبر، NaN/Inf و leakage هرگز با آمار «اصلاح» نمی‌شوند و
از execution نیز حذف/مسدود می‌مانند.

تفسیر قفل‌شدهٔ ورودی تصویر:

```text
FBP: PATTERN_NORMALIZED_FBP_NOT_OFFICIAL_CENTILOID
FTP: PATTERN_NORMALIZED_FTP_NOT_CENTAUR_QUANTITATIVE
```

## 2. فایل‌های ساخته‌شده یا تکمیل‌شده امروز

| فایل | نقش | SHA-256 در snapshot امروز |
|---|---|---|
| `configs/method_y_image_training_v3.yaml` | قرارداد داده، مدل، HPO، robustness و runtime | `cbf8e7b53beb4345fffbe98a15b054a793c2578b35e285a6e98ca897cbfcf0b5` |
| `src/method_y_revival/image_training_policy.py` | جداسازی execution gate از claim gate | `32592c0f3615f6c3121dc17a0528b6b0df343b99947b1c03fcd7a79c791036f0` |
| `src/method_y_revival/image_training_manifest_v3.py` | ممیزی membership، tracer، split و event support | `6b56f5fae291b3cfd185ebb1a79d5b274abe52c5e00da70be182964518997fe1` |
| `scripts/build_method_y_image_training_readiness_v3.py` | ساخت membership خصوصی و readiness عمومی | `7d3441e47ad48abb5f2d79fc8c42ce85dd498c1e2d9e454c0da42f0d613d9a1f` |
| `src/method_y_revival/image_survival_v3.py` | normalization الگویی، discrete survival loss و شبکه‌های 3D/tabular | `e255adb297e487e7da15ffa88f0295656b4da8c779b985df432f0ccb997a0c33` |
| `src/method_y_revival/image_cache_v3.py` | cache رزوم‌پذیر per-subject روی H | `c5ce56368a7ae5723b71982bdf09d66bd52579aa95d09dbbb83ba6fb0a547d51` |
| `scripts/build_method_y_image_cache_v3.py` | launcher ساخت cacheهای دو شاخه | `4ca4104699ab18c3cab6183e7dd582e1e7a158c72fc62aa59349d07b82e6fee3` |
| `src/method_y_revival/image_training_runtime_v3.py` | training loop، checkpoint اتمیک، RNG و resume epoch-level | `42a039329cf3f052ce2b525187dc474c19f601aa7e20654046b9a5bf276b6271` |

تست‌های جدید:

```text
tests/method_y_revival/test_image_training_policy_v3.py
tests/method_y_revival/test_image_training_manifest_v3.py
tests/method_y_revival/test_image_survival_v3.py
tests/method_y_revival/test_image_cache_v3.py
tests/method_y_revival/test_image_training_runtime_v3.py
```

## 3. artifact زندهٔ readiness

مسیر canonical این extension:

```text
results/method_y_revival_v3_image_v1/
```

مسیر بدون suffix یعنی `results/method_y_revival_v3_image/` یک snapshot اولیهٔ
superseded است و نباید برای training مصرف، overwrite یا با v1 ادغام شود.

رسید اصلی:

```text
results/method_y_revival_v3_image_v1/readiness/FINAL_IMAGE_TRAINING_READINESS.json
```

مقادیر قفل‌شدهٔ فعلی:

```yaml
status: READY_FOR_FINAL_INTERNAL_IMAGE_TRAINING
training_execution_allowed: true
positive_claim_allowed: false
external_claim_allowed: false
run_contract_sha256: 19baa31d0acfbb15e8bcc11828bae8bee2b9e0ed3b1beea06f7a1119a74979c5

amyloid:
  development_subjects: 560
  events: 110
  internal_transport_LEARN_subjects: 335
  internal_transport_events: 23
  minimum_outer_test_events: 19
  minimum_inner_validation_events: 24
  scanner_support_available: true

tau:
  development_subjects: 306
  events: 75
  minimum_outer_test_events: 14
  minimum_inner_validation_events: 17
  scanner_support_available: false
```

Membership خصوصی amyloid شامل A4 development و LEARN internal transport است؛
آموزش/HPO اصلی فقط A4 را مصرف می‌کند. Membership tau شامل A4 placebo و A4
treatment است. هر outer-test membership در snapshot زنده دقیقاً یک‌بار دیده
می‌شود؛ inner train/validation نیز پوشش کامل و overlap صفر دارد.

## 4. آزمون‌های اجراشده امروز

### آزمون‌های متمرکز v3

```powershell
& 'H:\New project\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider `
  tests\method_y_revival\test_image_training_policy_v3.py `
  tests\method_y_revival\test_image_training_manifest_v3.py `
  tests\method_y_revival\test_image_survival_v3.py `
  tests\method_y_revival\test_image_cache_v3.py `
  tests\method_y_revival\test_image_training_runtime_v3.py -q
```

نتیجه: `23 passed in 7.91s`.

### کل suite مربوط به Method Y

```powershell
& 'H:\New project\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider `
  tests\method_y_revival -q
```

نتیجه: `363 passed in 190.84s`.

full-repository suite پس از آخرین تغییر runtime امروز دوباره اجرا نشده است؛
عدد تاریخی `1956 passed, 1 skipped` متعلق به finalization قبلی Method Y v2 است
و نباید به‌عنوان verification تازهٔ v3 معرفی شود.

## 5. runtime و storage snapshot

در زمان ثبت:

```text
active Method Y/PET writer: none
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM: 8188 MiB total, 7956 MiB free
Torch: 2.5.1+cu121
scikit-survival: 0.27.0
H free: 446.60 GiB
C free: 52.62 GiB
```

تمام output/tempهای launcher نهایی باید روی H باشند. هیچ training طولانی یا
ساخت کامل cache v3 در زمان این ثبت شروع نشده است.

## 6. کارهایی که هنوز انجام نشده‌اند

1. ADR جدید برای supersede کردن عبارت قدیمی «quantitative gate مانع آموزش».
2. جداسازی `scanner_support_available` از `scanner_robustness_proven`؛ کد فعلی
   در amyloid این دو را اشتباه یکی می‌گیرد.
3. hardening validator برای outer-test exactly-once و inner coverage کامل.
4. bind کردن cache build به hash منبع/readiness و بررسی attr هر HDF5 group.
5. بازیابی resume در crash بین terminal epoch و locked checkpoint/summary.
6. materialize کردن cacheهای v3 برای 560 amyloid و 306 tau scan.
7. ساخت high-level nested-CV/HPO trainer؛ low-level epoch runtime کافی نیست.
8. انتخاب winner فقط با mean inner IBS و قفل config/checkpoint.
9. ساخت evaluator censoring-aware و outer-test exactly-once.
10. matched comparison بین scalar+tabular و scalar+tabular+image.
11. scanner-balanced، leave-manufacturer-out، UNKNOWN-excluded و OOF probe.
12. calibration، subgroup CI، paired bootstrap و claim/privacy guard.
13. ساخت PS1 واحد، GPU smoke، integration interruption tests و full suite.
14. شروع آموزش واقعی؛ هنوز انجام نشده است.

جزئیات اجرای همهٔ موارد در Planهای 009 تا 013 همین پوشه آمده است.

## 7. مرز ادعا که نباید تغییر کند

- source-regional v2 همچنان نتیجهٔ اصلی محلی و image v3 تحلیل ثانویه است.
- target، future sustained CDR progression است؛ همان PET ورودی target نیست.
- PET pattern-normalized در v3 کمیِ رسمی نیست.
- هیچ claim تشخیص آلزایمر، causal localization، official Centiloid/CenTauR یا
  external generalization مجاز نیست.
- نبود cohort خارجی مانع اجرای داخلی نیست، ولی external claim را مسدود می‌کند.
- statistics فقط sensitivity/confounding را کمّی می‌کند؛ دادهٔ مستقل یا
  provenance کمیِ مفقود را ایجاد نمی‌کند.
