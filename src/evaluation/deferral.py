from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeferralResult:
    """Result of deferral decision."""

    should_defer: bool
    confidence: float
    threshold: float
    prediction: float
    reason: str


class DeferralPolicy:
    """Threshold-based agent deferral policy.

    Inspired by strategic decision support (arXiv:2606.12587) which shows
    that optimal agent behavior requires thresholding on the value of support.
    When model confidence is below threshold, prediction is deferred to a
    stronger model or clinician.
    """

    def __init__(self, confidence_threshold: float = 0.7, min_confidence: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.min_confidence = min_confidence

    def compute_confidence(self, probability: float) -> float:
        return abs(probability - 0.5) * 2.0

    def should_defer(self, probability: float) -> DeferralResult:
        confidence = self.compute_confidence(probability)
        should_defer = confidence < self.confidence_threshold
        if should_defer:
            if confidence < self.min_confidence:
                reason = "VERY_LOW_CONFIDENCE: prediction near chance level"
            else:
                reason = f"LOW_CONFIDENCE: {confidence:.2f} < threshold {self.confidence_threshold}"
        else:
            reason = f"HIGH_CONFIDENCE: {confidence:.2f} >= threshold {self.confidence_threshold}"
        return DeferralResult(
            should_defer=should_defer,
            confidence=confidence,
            threshold=self.confidence_threshold,
            prediction=probability,
            reason=reason,
        )

    def batch_deferral(self, probabilities: list[float]) -> dict:
        results = [self.should_defer(p) for p in probabilities]
        n_total = len(results)
        n_deferred = sum(1 for r in results if r.should_defer)
        return {
            "n_total": n_total,
            "n_deferred": n_deferred,
            "n_retained": n_total - n_deferred,
            "deferral_rate": n_deferred / max(n_total, 1),
            "results": results,
        }
