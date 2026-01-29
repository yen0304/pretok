"""Composite detector that combines multiple backends."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from pretok.detection import DetectionError, DetectionResult
from pretok.detection.base import BaseDetector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pretok.config import CompositeDetectorConfig


class CompositeDetector(BaseDetector):
    """Detector that combines multiple backends for improved accuracy.

    Supports multiple aggregation strategies:
    - voting: Use majority vote among detectors
    - weighted_average: Use weighted average of confidences
    - fallback_chain: Use first successful detection

    Example:
        >>> from pretok.detection.langdetect_backend import LangDetectDetector
        >>> detector = CompositeDetector([LangDetectDetector()])
        >>> result = detector.detect("Hello, world!")
        >>> result.language
        'en'
    """

    def __init__(
        self,
        detectors: Sequence[BaseDetector],
        config: CompositeDetectorConfig | None = None,
    ) -> None:
        """Initialize composite detector.

        Args:
            detectors: List of detector backends to combine
            config: Optional configuration

        Raises:
            ValueError: If no detectors provided
        """
        if not detectors:
            msg = "At least one detector must be provided"
            raise ValueError(msg)

        self._detectors = list(detectors)
        self._config = config
        self._strategy = config.strategy if config else "voting"
        self._weights = config.weights if config else {}

    @property
    def name(self) -> str:
        """Return detector name."""
        return "composite"

    @property
    def detectors(self) -> list[BaseDetector]:
        """Return list of backend detectors."""
        return self._detectors

    def detect(self, text: str) -> DetectionResult:
        """Detect language using combined backends.

        Args:
            text: Text to detect language for

        Returns:
            DetectionResult with detected language

        Raises:
            DetectionError: If all detectors fail
        """
        if self._strategy == "fallback_chain":
            return self._detect_fallback_chain(text)
        elif self._strategy == "weighted_average":
            return self._detect_weighted_average(text)
        else:  # voting (default)
            return self._detect_voting(text)

    def _detect_voting(self, text: str) -> DetectionResult:
        """Use majority voting strategy.

        Args:
            text: Text to detect

        Returns:
            DetectionResult based on majority vote
        """
        results: list[DetectionResult] = []
        errors: list[str] = []

        for detector in self._detectors:
            try:
                result = detector.detect(text)
                results.append(result)
            except DetectionError as e:
                errors.append(f"{detector.name}: {e}")

        if not results:
            raise DetectionError(
                f"All detectors failed: {'; '.join(errors)}",
                text=text,
                detector=self.name,
            )

        # Count votes for each language
        votes = Counter(r.language for r in results)
        winner, vote_count = votes.most_common(1)[0]

        # Calculate confidence as agreement ratio
        agreement = vote_count / len(results)

        # Get average confidence from detectors that voted for winner
        winning_confidences = [r.confidence for r in results if r.language == winner]
        avg_confidence = sum(winning_confidences) / len(winning_confidences)

        # Final confidence combines agreement and detector confidences
        final_confidence = agreement * avg_confidence

        return DetectionResult(
            language=winner,
            confidence=final_confidence,
            detector=self.name,
            raw_output={
                "votes": dict(votes),
                "agreement": agreement,
                "results": [
                    {"detector": r.detector, "language": r.language, "confidence": r.confidence}
                    for r in results
                ],
            },
        )

    def _detect_weighted_average(self, text: str) -> DetectionResult:
        """Use weighted average strategy.

        Args:
            text: Text to detect

        Returns:
            DetectionResult based on weighted confidence
        """
        results: list[DetectionResult] = []
        errors: list[str] = []

        for detector in self._detectors:
            try:
                result = detector.detect(text)
                results.append(result)
            except DetectionError as e:
                errors.append(f"{detector.name}: {e}")

        if not results:
            raise DetectionError(
                f"All detectors failed: {'; '.join(errors)}",
                text=text,
                detector=self.name,
            )

        # Calculate weighted scores for each language
        language_scores: dict[str, float] = {}
        total_weight = 0.0

        for result in results:
            weight = self._weights.get(result.detector, 1.0)
            score = weight * result.confidence
            total_weight += weight

            if result.language in language_scores:
                language_scores[result.language] += score
            else:
                language_scores[result.language] = score

        # Normalize scores
        if total_weight > 0:
            language_scores = {
                lang: score / total_weight for lang, score in language_scores.items()
            }

        # Get winner
        winner = max(language_scores, key=language_scores.get)  # type: ignore[arg-type]
        confidence = language_scores[winner]

        return DetectionResult(
            language=winner,
            confidence=min(confidence, 1.0),  # Cap at 1.0
            detector=self.name,
            raw_output={
                "scores": language_scores,
                "weights": self._weights,
                "results": [
                    {"detector": r.detector, "language": r.language, "confidence": r.confidence}
                    for r in results
                ],
            },
        )

    def _detect_fallback_chain(self, text: str) -> DetectionResult:
        """Use fallback chain strategy.

        Try each detector in order until one succeeds.

        Args:
            text: Text to detect

        Returns:
            DetectionResult from first successful detector
        """
        errors: list[str] = []

        for detector in self._detectors:
            try:
                return detector.detect(text)
            except DetectionError as e:
                errors.append(f"{detector.name}: {e}")

        raise DetectionError(
            f"All detectors in chain failed: {'; '.join(errors)}",
            text=text,
            detector=self.name,
        )


def create_detector(
    backend: str,
    config: dict[str, Any] | None = None,
) -> BaseDetector:
    """Factory function to create a detector by name.

    Args:
        backend: Name of the backend ('langdetect', 'fasttext', 'composite')
        config: Optional configuration dictionary

    Returns:
        Configured detector instance

    Raises:
        ValueError: If unknown backend specified
        ImportError: If required dependencies not installed
    """
    if backend == "langdetect":
        from pretok.config import LangDetectConfig
        from pretok.detection.langdetect_backend import LangDetectDetector

        langdetect_config = LangDetectConfig(**config) if config else None
        return LangDetectDetector(langdetect_config)

    elif backend == "fasttext":
        from pretok.config import FastTextConfig
        from pretok.detection.fasttext_backend import FastTextDetector

        fasttext_config = FastTextConfig(**config) if config else None
        return FastTextDetector(fasttext_config)

    elif backend == "composite":
        from pretok.config import CompositeDetectorConfig

        composite_config = CompositeDetectorConfig(**config) if config else None
        detector_names = composite_config.detectors if composite_config else ["langdetect"]

        # Recursively create child detectors
        child_detectors = [create_detector(name) for name in detector_names]
        return CompositeDetector(child_detectors, composite_config)

    else:
        msg = f"Unknown detector backend: {backend}"
        raise ValueError(msg)
