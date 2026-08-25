# Manifest بستهٔ handoff — Method Y image v3

**ساخته‌شده:** 2026-08-24  
**مخزن:** `H:\New project`  
**commit مبنا:** `5405716`  
**مخاطب:** Gemini، GLM یا هر ایجنت اجرایی با context صفر

## ترتیب مطالعه

1. `METHOD_Y_IMAGE_V3_TODAY_LOG_20260824.md`
2. `README.md`، بخش Method Y image v3
3. `GEMINI_GLM_METHOD_Y_IMAGE_V3_MASTER_PROMPT_FA.md`
4. فقط Plan تخصیص‌یافته از 009 تا 013

## فایل‌ها و hash

| فایل | بایت | SHA-256 |
|---|---:|---|
| `METHOD_Y_IMAGE_V3_TODAY_LOG_20260824.md` | 8779 | `11fb2716bb21a60129b4d133a761a9b6cdf41f38314946f7da8a9744d8757d75` |
| `009-method-y-v3-dual-gate-and-readiness-hardening.md` | 13282 | `de1451da90ffe8bbdc919dfbfd5050b93c673f0ee2708e2832d03bcf489dc69e` |
| `010-method-y-v3-cache-provenance-and-resume.md` | 11851 | `51cd611003c324563a52eec21b37a193563aa2b196d1552b1bdeecf6e13aa2e2` |
| `011-method-y-v3-nested-image-trainer.md` | 13704 | `26f76c2c3f2e6715623d51855a9b130082c49dea7ed9e811930660a0f3380948` |
| `012-method-y-v3-robustness-reports-and-claim-guard.md` | 10368 | `1fc13c43fc60bf2d8243a68388722de8433ce37abce0e150a01a70d170fa313c` |
| `013-method-y-v3-launcher-acceptance-and-training-run.md` | 11509 | `f9af1cab97766ef8b2b47d15d2b6392d80b145f5e29adc6c979d551aa138f5ea` |
| `GEMINI_GLM_METHOD_Y_IMAGE_V3_MASTER_PROMPT_FA.md` | 9123 | `7b312d279c3a0d19f138aa2a890e2e7c228e5a1faa08ef88090f014bbcbd1077` |

## dependency graph

```text
009 governance/readiness
  ↓
010 cache/resume
  ↓
011 nested trainer/exact outer
  ↓
012 robustness/report/claim/privacy
  ↓
013 launcher/acceptance/production run
```

## scope receipt

در زمان ساخت این بسته:

- هیچ source/config/test پروژه توسط مهارت `improve` تغییر نکرد؛
- فقط فایل‌های `advisor-plans/` ایجاد/به‌روزرسانی شدند؛
- هیچ cache، HPO، training یا outer evaluation اجرا نشد؛
- فقط آزمون‌های read-only اجرا شدند: `23 passed` متمرکز و `363 passed` کل
  `tests/method_y_revival`؛
- هیچ writer Method Y فعال نبود.

## hash verification command

```powershell
Set-Location -LiteralPath 'H:\New project\advisor-plans'
Get-FileHash -Algorithm SHA256 `
  METHOD_Y_IMAGE_V3_TODAY_LOG_20260824.md, `
  009-method-y-v3-dual-gate-and-readiness-hardening.md, `
  010-method-y-v3-cache-provenance-and-resume.md, `
  011-method-y-v3-nested-image-trainer.md, `
  012-method-y-v3-robustness-reports-and-claim-guard.md, `
  013-method-y-v3-launcher-acceptance-and-training-run.md, `
  GEMINI_GLM_METHOD_Y_IMAGE_V3_MASTER_PROMPT_FA.md
```

اگر hash فرق دارد، بسته drift کرده است؛ Plan را کور اجرا نکن.
