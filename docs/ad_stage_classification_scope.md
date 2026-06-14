# AD Stage Classification Scope

## Current Scope: Binary Classification

The project performs **binary classification**: Amyloid-positive (AB+) vs Amyloid-negative (AB-).

### Why Binary Classification

1. **Proposal target**: "Early prediction of amyloid biomarker status" (AB+ vs AB-)
2. **Clinical relevance**: AB status determines eligibility for anti-amyloid therapy
3. **Label availability**: AB status from PET visual reads (imaging_PET_VA.csv)
4. **Cohort structure**: A4=AB+, LEARN=AB- (binary by design)

### Classes

| Class | Label | Count | Substudy |
|-------|-------|-------|----------|
| AB+ (Amyloid-positive) | 1 | 1,163 | A4 |
| AB- (Amyloid-negative) | 0 | 536 | LEARN |
| **Total** | | **1,699** | |

## Multi-Stage Classification (Future Work)

The proposal mentioned "classifying various stages of Alzheimer's disease." This would require:

### Potential Stages

| Stage | Description | Label Source |
|-------|-------------|--------------|
| Normal Cognition (NC) | No cognitive impairment | Clinical assessment |
| Mild Cognitive Impairment (MCI) | Intermediate impairment | Clinical assessment |
| Alzheimer's Disease (AD) | Clinical dementia | Clinical diagnosis |

### Why Not Implemented

1. **Label availability**: AB status is available; clinical AD diagnosis requires independent validation
2. **Cohort design**: A4/LEARN enrolled preclinical amyloid-positive individuals, not clinical AD patients
3. **Research focus**: Preclinical detection (before symptoms) is the proposal's primary objective

### Future Implementation

If multi-stage classification is needed:
- Use clinical assessment data (CDR, MMSE) as labels
- Require independent clinical validation
- Add MCI as intermediate class

## Conclusion

The current binary classification (AB+/AB-) is appropriate for the proposal's stated objective of "early prediction of amyloid biomarker status." Multi-stage classification is a natural extension but requires additional label sources.
