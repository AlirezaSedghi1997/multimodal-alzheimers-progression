# ADR-2026-08-24: تفکیک Dual-Gate برای آموزش داخلی اکتشافی تصویر Method Y v3

## وضعیت

Accepted — این ADR به‌طور مشخص بخش گیت آموزش تصویری در ADR-2026-07-19 و
`docs/PROGRAM_DEFAULT_SOURCE_OF_TRUTH_2026-07-19.md:183` را برای شاخهٔ تصویری
اکتشافی Method Y به‌صورت `SUPERSEDED_IN_PART` بازتعریف می‌کند.

## زمینه

در برنامهٔ قبلی (ADR-2026-07-19) قید شده بود که تا زمان حل قطعی کالیبراسیون
رسمی Centiloid/CenTauR و اعتبارسنجی خارجی، آموزش و ادعا شروع نمی‌شود. اما برای
تحلیل ثانویه و اکتشافی تصویر بر پایهٔ نرمال‌سازی الگویی (pattern-normalized)،
عدم وجود Centiloid رسمی یا کوهورت مستقل نباید مانع آموزش مدل داخلی در محیط
محافظت‌شده و ایزولهٔ A4 شود، مشروط بر اینکه ادعای کمی، بالینی یا تعمیم خارجی
به‌طور قطعی مسدود بماند.

## تصمیم

سیاست اعتبارسنجی به دو گیت مستقل و سلسله‌مراتبی تفکیک می‌شود:

1. **`training_execution_allowed` (گیت اجرای فنی و مهندسی):**
   - هویت دقیق ردیاب (FBP برای آمیلوئید، FTP برای تائو)
   - نقطهٔ پایانی آینده‌نگر و قفل‌شده (`future sustained CDR progression`)
   - تقسیم‌بندی قفل‌شدهٔ ۵ لایه بیرونی و ۳ لایه درونی گروه‌بندی‌شده بر اساس فرد
   - عدم نشت فرد بین لایه‌ها (`zero subject leakage`)
   - تبدیل فضایی و هندسهٔ معتبر تصویر (بدون zoom ناهماهنگ)
   - مقادیر عددی معتبر و متناهی (بدون NaN/Inf داخل مغز)
   - کفایت تعداد رخداد در تمام لایه‌ها (`adequate event support`)

2. **`positive_claim_allowed` (گیت ادعای علمی مثبت و کمی):**
   - اثبات رسمی هم‌ارزی کمی (Centiloid / CenTauR)
   - تحلیل‌های مستقل و قفل‌شدهٔ حساسیت و استحکام اسکنر (`scanner robustness`)
   - اثبات ارزش افزودهٔ خارج از نمونه (OOF incremental value) نسبت به مدل‌های پایه‌ای

3. **`external_claim_allowed` (گیت ادعای تعمیم خارجی):**
   - ارزیابی قفل‌شده بدون بازتنظیم (locked no-refit external evaluation) روی کوهورت مستقل

## نقش مدل‌ها و داده‌ها

```yaml
source_regional_v2_role: PRIMARY_LOCAL_INTERNAL_RESULT
image_v3_role: SECONDARY_FINAL_INTERNAL_EXPLORATORY_ANALYSIS
training_execution_requires:
  - exact tracer identity
  - future locked endpoint
  - locked subject-grouped 5x3 split
  - zero subject leakage
  - valid transform/geometry
  - finite voxels
  - adequate event support
claim_requires_additionally:
  - quantitative equivalence
  - actual scanner robustness analyses
  - OOF incremental value
external_claim_requires_additionally:
  - locked no-refit external evaluation
human_review_required: false
```

- تفسیر تصویر در v3:
  - FBP: `PATTERN_NORMALIZED_FBP_NOT_OFFICIAL_CENTILOID`
  - FTP: `PATTERN_NORMALIZED_FTP_NOT_CENTAUR_QUANTITATIVE`
- ابزارهای آماری، بازوزن‌دهی، یا تحلیل‌های حساسیت جایگزین شواهد کمی مستقل یا کوهورت خارجی نمی‌شوند.
- سه اسکن با مقیاس‌بندی نامعتبر منبع T1 برای همیشه از عضویت مدل خارج می‌مانند و با تقریب یا حدس اصلاح نمی‌شوند.

## پیامدها

- آموزش اکتشافی داخلی v3 در مخزن زنده مجاز است.
- تا زمان گذراندن گیت‌های ادعا، ادعاهای زیر اکیداً ممنوع هستند:
  - تشخیص بیماری آلزایمر (`ALZHEIMER_DISEASE_DIAGNOSIS`)
  - مکان‌یابی یا اثر علیتی (`CAUSAL_LOCALIZATION`)
  - هم‌ارزی با مقیاس رسمی Centiloid یا CenTauR
  - تعمیم خارجی بدون ارزیابی کوهورت مستقل قفل‌شده
- ADR قبلی (ADR-2026-07-19) حذف نمی‌شود و برای مابقی بخش‌های پروژه همچنان معتبر است.
