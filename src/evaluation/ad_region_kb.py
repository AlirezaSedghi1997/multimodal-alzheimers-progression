from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


@dataclass
class ADRegionInfo:
    """Alzheimer's disease clinical relevance for an AAL region."""

    aal_id: int
    region_name: str
    ad_relevance: str
    ad_evidence_level: str
    typical_ad_change: str
    clinical_notes: str
    braid_score: float


AD_RELEVANCE_DB: dict[int, ADRegionInfo] = {
    37: ADRegionInfo(37, "Hippocampus_L", "core", "pathognomonic", "atrophy", "Earliest tau tangle deposition site; most sensitive AD biomarker", 1.0),
    38: ADRegionInfo(38, "Hippocampus_R", "core", "pathognomonic", "atrophy", "Earliest tau tangle deposition site; most sensitive AD biomarker", 1.0),
    39: ADRegionInfo(39, "ParaHippocampal_L", "core", "well-established", "atrophy", "Entorhinal cortex extension; early atrophy in AD", 0.95),
    40: ADRegionInfo(40, "ParaHippocampal_R", "core", "well-established", "atrophy", "Entorhinal cortex extension; early atrophy in AD", 0.95),
    41: ADRegionInfo(41, "Amygdala_L", "core", "well-established", "atrophy", "Emotional processing center; early involvement in AD", 0.9),
    42: ADRegionInfo(42, "Amygdala_R", "core", "well-established", "atrophy", "Emotional processing center; early involvement in AD", 0.9),
    35: ADRegionInfo(35, "Cingulum_Post_L", "core", "well-established", "atrophy+plaques", "Posterior cingulate cortex; key node of default mode network; amyloid plaque deposition", 0.9),
    36: ADRegionInfo(36, "Cingulum_Post_R", "core", "well-established", "atrophy+plaques", "Posterior cingulate cortex; key node of default mode network; amyloid plaque deposition", 0.9),
    67: ADRegionInfo(67, "Precuneus_L", "core", "well-established", "atrophy", "Major hub of default mode network; early metabolic decline in AD", 0.85),
    68: ADRegionInfo(68, "Precuneus_R", "core", "well-established", "atrophy", "Major hub of default mode network; early metabolic decline in AD", 0.85),
    81: ADRegionInfo(81, "Temporal_Sup_L", "strong", "well-established", "atrophy", "Superior temporal gyrus; language and auditory processing decline in AD", 0.8),
    82: ADRegionInfo(82, "Temporal_Sup_R", "strong", "well-established", "atrophy", "Superior temporal gyrus; language and auditory processing decline in AD", 0.8),
    85: ADRegionInfo(85, "Temporal_Mid_L", "strong", "well-established", "atrophy", "Middle temporal gyrus; semantic memory impairment in AD", 0.8),
    86: ADRegionInfo(86, "Temporal_Mid_R", "strong", "well-established", "atrophy", "Middle temporal gyrus; semantic memory impairment in AD", 0.8),
    89: ADRegionInfo(89, "Temporal_Inf_L", "strong", "emerging", "atrophy", "Inferior temporal gyrus; visual object recognition deficit", 0.7),
    90: ADRegionInfo(90, "Temporal_Inf_R", "strong", "emerging", "atrophy", "Inferior temporal gyrus; visual object recognition deficit", 0.7),
    83: ADRegionInfo(83, "Temporal_Pole_Sup_L", "strong", "emerging", "atrophy", "Temporal pole; social cognition and semantic knowledge", 0.7),
    84: ADRegionInfo(84, "Temporal_Pole_Sup_R", "strong", "emerging", "atrophy", "Temporal pole; social cognition and semantic knowledge", 0.7),
    87: ADRegionInfo(87, "Temporal_Pole_Mid_L", "strong", "emerging", "atrophy", "Temporal pole; emotional and social processing", 0.65),
    88: ADRegionInfo(88, "Temporal_Pole_Mid_R", "strong", "emerging", "atrophy", "Temporal pole; emotional and social processing", 0.65),
    31: ADRegionInfo(31, "Cingulum_Ant_L", "moderate", "emerging", "atrophy", "Anterior cingulate; executive function decline", 0.6),
    32: ADRegionInfo(32, "Cingulum_Ant_R", "moderate", "emerging", "atrophy", "Anterior cingulate; executive function decline", 0.6),
    33: ADRegionInfo(33, "Cingulum_Mid_L", "moderate", "emerging", "atrophy", "Mid cingulate; attention and error monitoring", 0.55),
    34: ADRegionInfo(34, "Cingulum_Mid_R", "moderate", "emerging", "atrophy", "Mid cingulate; attention and error monitoring", 0.55),
    29: ADRegionInfo(29, "Insula_L", "moderate", "emerging", "atrophy", "Insular cortex; interoception and autonomic regulation", 0.55),
    30: ADRegionInfo(30, "Insula_R", "moderate", "emerging", "atrophy", "Insular cortex; interoception and autonomic regulation", 0.55),
    55: ADRegionInfo(55, "Fusiform_L", "moderate", "emerging", "atrophy", "Fusiform gyrus; face and object recognition", 0.5),
    56: ADRegionInfo(56, "Fusiform_R", "moderate", "emerging", "atrophy", "Fusiform gyrus; face and object recognition", 0.5),
    71: ADRegionInfo(71, "Caudate_L", "moderate", "emerging", "atrophy", "Caudate nucleus; striatal atrophy in moderate AD", 0.45),
    72: ADRegionInfo(72, "Caudate_R", "moderate", "emerging", "atrophy", "Caudate nucleus; striatal atrophy in moderate AD", 0.45),
    65: ADRegionInfo(65, "Angular_L", "moderate", "emerging", "atrophy", "Angular gyrus; semantic and spatial processing", 0.5),
    66: ADRegionInfo(66, "Angular_R", "moderate", "emerging", "atrophy", "Angular gyrus; semantic and spatial processing", 0.5),
    63: ADRegionInfo(63, "SupraMarginal_L", "weak", "minimal", "atrophy", "Supramarginal gyrus; attention and phonological processing", 0.35),
    64: ADRegionInfo(64, "SupraMarginal_R", "weak", "minimal", "atrophy", "Supramarginal gyrus; attention and phonological processing", 0.35),
    43: ADRegionInfo(43, "Calcarine_L", "weak", "minimal", "minimal", "Primary visual cortex; typically spared until late AD", 0.2),
    44: ADRegionInfo(44, "Calcarine_R", "weak", "minimal", "minimal", "Primary visual cortex; typically spared until late AD", 0.2),
    45: ADRegionInfo(45, "Cuneus_L", "weak", "minimal", "minimal", "Cuneus; visual processing; late involvement", 0.2),
    46: ADRegionInfo(46, "Cuneus_R", "weak", "minimal", "minimal", "Cuneus; visual processing; late involvement", 0.2),
    47: ADRegionInfo(47, "Lingual_L", "weak", "minimal", "minimal", "Lingual gyrus; visual processing", 0.2),
    48: ADRegionInfo(48, "Lingual_R", "weak", "minimal", "minimal", "Lingual gyrus; visual processing", 0.2),
    1: ADRegionInfo(1, "Precentral_L", "weak", "minimal", "minimal", "Primary motor cortex; typically spared in AD", 0.15),
    2: ADRegionInfo(2, "Precentral_R", "weak", "minimal", "minimal", "Primary motor cortex; typically spared in AD", 0.15),
    57: ADRegionInfo(57, "Postcentral_L", "weak", "minimal", "minimal", "Primary somatosensory cortex; typically spared", 0.15),
    58: ADRegionInfo(58, "Postcentral_R", "weak", "minimal", "minimal", "Primary somatosensory cortex; typically spared", 0.15),
    77: ADRegionInfo(77, "Thalamus_L", "weak", "minimal", "minimal", "Thalamus; relay center; less affected in early AD", 0.25),
    78: ADRegionInfo(78, "Thalamus_R", "weak", "minimal", "minimal", "Thalamus; relay center; less affected in early AD", 0.25),
    91: ADRegionInfo(91, "Cerebelum_Crus1_L", "weak", "minimal", "minimal", "Cerebellum; typically spared until very late AD", 0.1),
    92: ADRegionInfo(92, "Cerebelum_Crus1_R", "weak", "minimal", "minimal", "Cerebellum; typically spared until very late AD", 0.1),
    93: ADRegionInfo(93, "Cerebelum_Crus2_L", "weak", "minimal", "minimal", "Cerebellum; typically spared until very late AD", 0.1),
    94: ADRegionInfo(94, "Cerebelum_Crus2_R", "weak", "minimal", "minimal", "Cerebellum; typically spared until very late AD", 0.1),
}

AD_CORE_IDS = {37, 38, 39, 40, 41, 42, 35, 36, 67, 68}
AD_STRONG_IDS = {81, 82, 85, 86, 89, 90, 83, 84, 87, 88}
AD_MODERATE_IDS = {31, 32, 33, 34, 29, 30, 55, 56, 71, 72, 65, 66}


def get_ad_relevance(aal_id: int) -> ADRegionInfo | None:
    return AD_RELEVANCE_DB.get(aal_id)


def get_relevance_level(aal_id: int) -> str:
    info = AD_RELEVANCE_DB.get(aal_id)
    return info.ad_relevance if info else "minimal"


def compute_ad_pattern_score(region_attributions: Any) -> float:
    if np is None:
        raise RuntimeError("NumPy required")
    if not region_attributions:
        return 0.0
    total_score = 0.0
    total_weight = 0.0
    for attr in region_attributions:
        info = AD_RELEVANCE_DB.get(attr.region_id)
        if info is None:
            continue
        weight = info.braid_score
        contribution = float(np.abs(attr.mean_attribution)) * weight
        total_score += contribution
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return min(total_score / total_weight * 5.0, 1.0)


def rank_by_ad_importance(region_attributions: Any) -> list[tuple[Any, ADRegionInfo]]:
    pairs = []
    for attr in region_attributions:
        info = AD_RELEVANCE_DB.get(attr.region_id)
        if info is None:
            continue
        pairs.append((attr, info))
    pairs.sort(key=lambda p: float(np.abs(p[0].mean_attribution)) * p[1].braid_score if np is not None else 0, reverse=True)
    return pairs


def get_core_region_involvement(region_attributions: Any) -> dict[str, float]:
    result = {}
    for attr in region_attributions:
        if attr.region_id in AD_CORE_IDS:
            result[attr.region_name] = float(np.abs(attr.mean_attribution)) if np is not None else 0.0
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def get_spatial_pattern(region_attributions: Any) -> str:
    if np is None:
        return "unknown"
    core = [a for a in region_attributions if a.region_id in AD_CORE_IDS]
    temporal = [a for a in region_attributions if a.region_id in AD_STRONG_IDS]
    if not core and not temporal:
        return "no_ad_pattern"
    core_names = [a.region_name for a in core[:4]]
    has_hipp = any("Hippocampus" in n for n in core_names)
    has_temporal = any("Temporal" in n for n in core_names)
    has_bilateral_l = any("_L" in n for n in core_names)
    has_bilateral_r = any("_R" in n for n in core_names)
    bilateral = has_bilateral_l and has_bilateral_r
    if has_hipp and bilateral:
        return "bilateral_hippocampal"
    if has_hipp:
        return "unilateral_hippocampal"
    if has_temporal and bilateral:
        return "bilateral_temporal"
    if has_temporal:
        return "unilateral_temporal"
    return "diffuse_cortical"
