"""Base class for language detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pretok.detection import DetectionResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class BaseDetector(ABC):
    """Abstract base class for language detectors.

    Provides common functionality and default implementations
    for language detector backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the detector's unique name identifier."""
        ...

    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """Detect the language of a single text.

        Args:
            text: The text to detect language for

        Returns:
            DetectionResult with language code and confidence

        Raises:
            DetectionError: If detection fails
        """
        ...

    def detect_batch(self, texts: Sequence[str]) -> list[DetectionResult]:
        """Detect languages for multiple texts.

        Default implementation calls detect() for each text.
        Subclasses may override for batch optimization.

        Args:
            texts: Sequence of texts to detect

        Returns:
            List of DetectionResult for each input text
        """
        return [self.detect(text) for text in texts]

    def _normalize_language_code(self, code: str) -> str:
        """Normalize language code to ISO 639-1 format.

        Args:
            code: Raw language code from detector

        Returns:
            Normalized ISO 639-1 code (lowercase, 2 characters)
        """
        # Handle common variations
        code = code.lower().strip()

        # Map 3-letter codes to 2-letter codes
        iso_639_3_to_1 = {
            "eng": "en",
            "zho": "zh",
            "cmn": "zh",
            "jpn": "ja",
            "kor": "ko",
            "spa": "es",
            "fra": "fr",
            "deu": "de",
            "ita": "it",
            "por": "pt",
            "rus": "ru",
            "ara": "ar",
            "hin": "hi",
            "ben": "bn",
            "vie": "vi",
            "tha": "th",
            "ind": "id",
            "msa": "ms",
            "tur": "tr",
            "pol": "pl",
            "ukr": "uk",
            "nld": "nl",
            "ces": "cs",
            "ell": "el",
            "heb": "he",
            "fas": "fa",
            "swe": "sv",
            "dan": "da",
            "fin": "fi",
            "nor": "no",
            "hun": "hu",
            "ron": "ro",
            "cat": "ca",
            "srp": "sr",
            "hrv": "hr",
            "slk": "sk",
            "bul": "bg",
            "lit": "lt",
            "lav": "lv",
            "est": "et",
            "slv": "sl",
        }

        if code in iso_639_3_to_1:
            return iso_639_3_to_1[code]

        # Handle script variants (e.g., zh-Hans, zh-Hant)
        if "-" in code:
            code = code.split("-")[0]

        # Handle underscore variants (e.g., zh_CN)
        if "_" in code:
            code = code.split("_")[0]

        return code[:2] if len(code) >= 2 else code
