# Cognitive Feature Policy

## Feature Families

Core cognitive features:

- PACC.
- FCTOTAL96.
- LDELTOTAL.
- DIGITTOTAL.
- MMSCORE.

Clinical severity features:

- CDGLOBAL.
- CDRSB.
- MEMORY.
- ORIENT.
- JUDGE.
- COMMUN.
- HOME.
- CARE.

Cognitive complaint features:

- CFIPTTOTAL.
- CFSPTTOTAL.

Optional computerized cognition features:

- C3Comp.
- LearnWMComp.
- OCLONBComp.
- PsychAttnComp.
- AttnComp.
- C3AbrComp.

Mood/anxiety covariates:

- GDTOTAL.
- STAITOTAL.
- CADCNCRN.
- IESCORE.

## Temporal Rule

Prefer the nearest non-future visit relative to the Method X catalog target time. Do not use cognitive or psychological measurements after the PET/T1 target time unless a future-use sensitivity analysis is explicitly labeled and separated.

## Privacy Rule

Raw subject identifiers may be used only transiently in memory for joins. They must not be written to processed feature tables, model outputs, public bundles, logs, or reports.
