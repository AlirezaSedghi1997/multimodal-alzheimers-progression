# Agent Stage Execution Policy

## Stage Order

1. `bootstrap`
2. `inventory`
3. `catalog`
4. `preprocess`
5. `splits`
6. `baseline`
7. `optuna`
8. `fusion`
9. `external`
10. `xai`
11. `final`
12. `public_bundle`
13. `tau`

## Progress Contract

Each stage must update `.agent/PROGRESS.yaml` with status, next action, and artifact paths. A stage may be `COMPLETED`, `BLOCKED`, `FAILED`, or `PENDING`.

## Privacy Contract

Stage logs must use aggregate counts and hashed identifiers only.

