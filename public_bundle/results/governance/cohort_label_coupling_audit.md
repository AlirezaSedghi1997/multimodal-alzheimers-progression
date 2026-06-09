# Cohort-Label Coupling Audit (Before/After SF)

## Before SF inclusion (A4 + LEARN enrolled MRI)

- rows: `1699`
- Cramér's V: `1.0000`
- label_by_substudy: `{'A4': {'1': 1163}, 'LEARN': {'0': 536}}`

## After SF inclusion (Method X-Cog/SF catalog)

- rows: `4484`
- Cramér's V: `0.8996`
- label_by_substudy: `{'A4': {'1': 1163}, 'LEARN': {'0': 536}, 'SF': {'0': 2591, '1': 194}}`

## SF-only subset

- rows: `2785`
- Cramér's V: `1.0000`
- primary vs sensitivity label mismatches: `36`

## Interpretation

Before SF inclusion, binary A4/LEARN classification is mathematically close to a cohort classifier. After SF inclusion, label and cohort decouple within the SF screening context. SF remains an internal decoupling instrument, not external validation.
