"""Base translator protocol and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Result from translation operation.

    Attributes:
        source_text: Original text
        translated_text: Translated text
        source_language: Detected or specified source language
        target_language: Target language
        translator: Name of the translator used
        confidence: Translation confidence (if available)
        metadata: Additional translation metadata
    """

    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    translator: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def was_translated(self) -> bool:
        """Check if text was actually translated."""
        return self.source_text != self.translated_text


class TranslationError(Exception):
    """Raised when translation fails."""

    def __init__(
        self,
        message: str,
        *,
        source_text: str | None = None,
        source_language: str | None = None,
        target_language: str | None = None,
        translator: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.source_text = source_text
        self.source_language = source_language
        self.target_language = target_language
        self.translator = translator
        self.cause = cause


@runtime_checkable
class Translator(Protocol):
    """Protocol for translation backends.

    All translators must implement this protocol to be usable
    with pretok's translation system.
    """

    @property
    def name(self) -> str:
        """Return the translator's unique name identifier."""
        ...

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        ...

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> TranslationResult:
        """Translate text to target language.

        Args:
            text: Text to translate
            target_language: Target language code (ISO 639-1)
            source_language: Source language code (optional, will auto-detect)

        Returns:
            TranslationResult with translated text
        """
        ...

    def translate_batch(
        self,
        texts: Sequence[str],
        target_language: str,
        source_language: str | None = None,
    ) -> list[TranslationResult]:
        """Translate multiple texts to target language.

        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code (optional)

        Returns:
            List of TranslationResult for each input
        """
        ...


class BaseTranslator(ABC):
    """Abstract base class for translators.

    Provides common functionality and default implementations
    for translator backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the translator's unique name identifier."""
        ...

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes.

        Override this in subclasses to provide actual supported languages.
        """
        return []

    @abstractmethod
    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> TranslationResult:
        """Translate text to target language."""
        ...

    def translate_batch(
        self,
        texts: Sequence[str],
        target_language: str,
        source_language: str | None = None,
    ) -> list[TranslationResult]:
        """Translate multiple texts to target language.

        Default implementation calls translate() for each text.
        Subclasses may override for batch optimization.
        """
        return [self.translate(text, target_language, source_language) for text in texts]

    def supports_language(self, language: str) -> bool:
        """Check if a language is supported.

        Args:
            language: Language code to check

        Returns:
            True if language is supported
        """
        if not self.supported_languages:
            return True  # If no list, assume all supported
        return language.lower() in [lang.lower() for lang in self.supported_languages]

    def _validate_languages(
        self,
        source_language: str | None,
        target_language: str,
    ) -> None:
        """Validate source and target languages are supported.

        Args:
            source_language: Source language (or None for auto-detect)
            target_language: Target language

        Raises:
            TranslationError: If language is not supported
        """
        if source_language and not self.supports_language(source_language):
            raise TranslationError(
                f"Source language '{source_language}' is not supported",
                source_language=source_language,
                translator=self.name,
            )

        if not self.supports_language(target_language):
            raise TranslationError(
                f"Target language '{target_language}' is not supported",
                target_language=target_language,
                translator=self.name,
            )
