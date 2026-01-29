"""LangDetect backend for language detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pretok.detection import DetectionError, DetectionResult
from pretok.detection.base import BaseDetector

if TYPE_CHECKING:
    from pretok.config import LangDetectConfig


class LangDetectDetector(BaseDetector):
    """Language detector using langdetect library.

    langdetect is a port of Google's language-detection library to Python.
    It's lightweight and doesn't require model files.

    Example:
        >>> detector = LangDetectDetector()
        >>> result = detector.detect("Hello, world!")
        >>> result.language
        'en'
    """

    def __init__(self, config: LangDetectConfig | None = None) -> None:
        """Initialize the langdetect detector.

        Args:
            config: Optional configuration for langdetect

        Raises:
            ImportError: If langdetect is not installed
        """
        try:
            import langdetect
            from langdetect import DetectorFactory
        except ImportError as e:
            msg = (
                "langdetect is required for LangDetectDetector. "
                "Install it with: pip install langdetect"
            )
            raise ImportError(msg) from e

        self._langdetect = langdetect
        self._config = config

        # Set seed for reproducibility if configured
        if config and config.seed is not None:
            DetectorFactory.seed = config.seed

    @property
    def name(self) -> str:
        """Return detector name."""
        return "langdetect"

    def detect(self, text: str) -> DetectionResult:
        """Detect language using langdetect.

        Args:
            text: Text to detect language for

        Returns:
            DetectionResult with detected language

        Raises:
            DetectionError: If detection fails
        """
        if not text or not text.strip():
            raise DetectionError(
                "Cannot detect language of empty text",
                text=text,
                detector=self.name,
            )

        try:
            # Get probabilities for all detected languages
            probs = self._langdetect.detect_langs(text)

            if not probs:
                raise DetectionError(
                    "No language detected",
                    text=text,
                    detector=self.name,
                )

            # Get the top result
            top = probs[0]
            language = self._normalize_language_code(top.lang)

            return DetectionResult(
                language=language,
                confidence=top.prob,
                detector=self.name,
                raw_output={"all_probs": [(p.lang, p.prob) for p in probs]},
            )

        except self._langdetect.LangDetectException as e:
            raise DetectionError(
                f"Language detection failed: {e}",
                text=text,
                detector=self.name,
            ) from e

    def detect_with_alternatives(self, text: str, *, top_k: int = 3) -> list[DetectionResult]:
        """Detect language with alternative possibilities.

        Args:
            text: Text to detect language for
            top_k: Number of top results to return

        Returns:
            List of DetectionResult ordered by confidence
        """
        if not text or not text.strip():
            return []

        try:
            probs = self._langdetect.detect_langs(text)
            results = []

            for prob in probs[:top_k]:
                language = self._normalize_language_code(prob.lang)
                results.append(
                    DetectionResult(
                        language=language,
                        confidence=prob.prob,
                        detector=self.name,
                    )
                )

            return results

        except self._langdetect.LangDetectException:
            return []
