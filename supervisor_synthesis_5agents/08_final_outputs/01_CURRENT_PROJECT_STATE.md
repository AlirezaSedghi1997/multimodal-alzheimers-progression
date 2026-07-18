# وضعیت فعلی پروژه

Snapshot UTC: 2026-07-18T17:13:02+00:00

- Method Y amyloid endpoint (locked v1 evidence): 1,613 eligible subjects / 278 events; A4 development 1,106/240; LEARN transport 507/38.
- Tau and paired A/T endpoint (locked v1 evidence): 369 subjects / 91 events.
- Tau beyond amyloid: locked negative result; this is retained, not overwritten.
- External data: DUA/raw manifest unavailable; external claim is not permitted.
- FTP transform writer active at audit start: True; its count is intentionally read from the live snapshot and is not frozen here.

## baseline آزمون

```json
{
  "command": "H:\\New project\\.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider",
  "returncode": 1,
  "status": "PREEXISTING_COLLECTION_FAILURE",
  "summary": "8 import-file-mismatch collection errors caused by duplicate tests under archives/.../patch_work/old and new plus root tests.",
  "captured_utc": "2026-07-18T17:13:02+00:00"
}
```

Baseline کل مخزن در این snapshot به‌علت import-file mismatch ناشی از testهای duplicate در `archives/.../patch_work/old` و `new` سبز نبود؛ این خطای پیشین است و برای اصلاح آن هیچ فایل محافظت‌شده‌ای تغییر داده نشد.

## process و provenance

- Git HEAD: `480752dba55e1cd0bd992b420b502d91952904fe`
- Branch: `methody/pet-branch-implementation`
- Dirty file count snapshot: `454`
- Active synthesis writer targeting workspace: `False`
- Active transform writer observed: `True`
- External DUA/raw manifest: `PENDING_DUA_ACCESS`
