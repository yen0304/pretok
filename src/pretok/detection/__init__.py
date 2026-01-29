"""Language detection protocols and types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Result from language detection.

    Attributes:
        language: ISO 639-1 language code (e.g., 'en', 'zh', 'ja')
        confidence: Confidence score between 0.0 and 1.0
        detector: Name of the detector that produced this result
        raw_output: Optional raw output from the detector for debugging
    """

    language: str
    confidence: float
    detector: str
    raw_output: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            raise ValueError(msg)


@runtime_checkable
class LanguageDetector(Protocol):
    """Protocol for language detection backends.

    All language detectors must implement this protocol to be usable
    with pretok's detection system.

    Example:
        >>> class MyDetector:
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_detector"
        ...
        ...     def detect(self, text: str) -> DetectionResult:
        ...         # Detection logic here
        ...         return DetectionResult(
        ...             language="en",
        ...             confidence=0.95,
        ...             detector=self.name
        ...         )
        ...
        ...     def detect_batch(self, texts: Sequence[str]) -> list[DetectionResult]:
        ...         return [self.detect(t) for t in texts]
    """

    @property
    def name(self) -> str:
        """Return the detector's unique name identifier."""
        ...

    def detect(self, text: str) -> DetectionResult:
        """Detect the language of a single text.

        Args:
            text: The text to detect language for

        Returns:
            DetectionResult with language code and confidence
        """
        ...

    def detect_batch(self, texts: Sequence[str]) -> list[DetectionResult]:
        """Detect languages for multiple texts.

        Default implementation calls detect() for each text,
        but backends may override for batch optimization.

        Args:
            texts: Sequence of texts to detect

        Returns:
            List of DetectionResult for each input text
        """
        ...


class DetectionError(Exception):
    """Raised when language detection fails."""

    def __init__(
        self,
        message: str,
        *,
        text: str | None = None,
        detector: str | None = None,
    ) -> None:
        super().__init__(message)
        self.text = text
        self.detector = detector
