"""Core pretok pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pretok.capability import get_default_registry
from pretok.detection import DetectionError, DetectionResult
from pretok.detection.base import BaseDetector
from pretok.detection.composite import create_detector
from pretok.pipeline.cache import Cache, MemoryCache, make_cache_key
from pretok.segment import Segment, SegmentType, lex_prompt
from pretok.segment.types import segments_to_text
from pretok.translation import TranslationError, TranslationResult
from pretok.translation.base import BaseTranslator

if TYPE_CHECKING:
    from pretok.config import PretokConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from pretok pipeline processing.

    Attributes:
        original_text: Original input text
        processed_text: Processed/translated text
        segments: List of segments with their details
        detections: Language detection results
        translations: Translation results
        from_cache: Whether result was from cache
        metadata: Additional processing metadata
    """

    original_text: str
    processed_text: str
    segments: list[Segment] = field(default_factory=list)
    detections: list[DetectionResult] = field(default_factory=list)
    translations: list[TranslationResult] = field(default_factory=list)
    from_cache: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def was_modified(self) -> bool:
        """Check if text was modified."""
        return self.original_text != self.processed_text


class Pretok:
    """Universal pre-token language adaptation layer.

    Pretok processes input text to optimize it for LLM tokenization:
    1. Parses prompt structure (roles, code, text)
    2. Detects languages in text segments
    3. Translates non-optimal languages to model's preferred language
    4. Reconstructs the prompt preserving structure

    Example:
        >>> from pretok import Pretok
        >>> pretok = Pretok(target_language="en")
        >>> result = pretok.process("Hello World")  # noqa: RUF002
        >>> result.processed_text
        'Hello World'

    With configuration:
        >>> from pretok import Pretok
        >>> from pretok.config import load_config
        >>> config = load_config("pretok.yaml")
        >>> pretok = Pretok(config=config)
    """

    def __init__(
        self,
        config: PretokConfig | None = None,
        *,
        target_language: str | None = None,
        model_id: str | None = None,
        detector: BaseDetector | None = None,
        translator: BaseTranslator | None = None,
        cache: Cache | None = None,
    ) -> None:
        """Initialize Pretok pipeline.

        Args:
            config: Full configuration object
            target_language: Target language for translation (overrides config)
            model_id: Model ID for capability lookup
            detector: Custom detector instance
            translator: Custom translator instance
            cache: Custom cache instance
        """
        self._config = config
        self._model_id = model_id

        # Determine target language
        if target_language:
            self._target_language = target_language
        elif model_id:
            capability = get_default_registry().get(model_id)
            self._target_language = capability.primary_language
        else:
            self._target_language = "en"

        # Initialize components
        self._detector = detector or self._create_detector()
        self._translator = translator
        self._cache = cache or self._create_cache()

        # Get model capability
        self._capability = get_default_registry().get(model_id) if model_id else None

        logger.info(
            "Initialized Pretok: target_language=%s, model_id=%s",
            self._target_language,
            model_id,
        )

    def _create_detector(self) -> BaseDetector:
        """Create detector from config or defaults."""
        if self._config:
            backend = self._config.pipeline.default_detector
            return create_detector(backend)

        # Default to langdetect
        return create_detector("langdetect")

    def _create_cache(self) -> Cache | None:
        """Create cache from config or defaults."""
        if self._config and not self._config.pipeline.cache_enabled:
            return None

        if self._config:
            cache_config = self._config.cache.memory
            return MemoryCache(
                max_size=cache_config.max_size,
                ttl=cache_config.ttl,
            )

        return MemoryCache()

    @property
    def target_language(self) -> str:
        """Return target language."""
        return self._target_language

    @property
    def detector(self) -> BaseDetector:
        """Return detector instance."""
        return self._detector

    @property
    def translator(self) -> BaseTranslator | None:
        """Return translator instance."""
        return self._translator

    def set_translator(self, translator: BaseTranslator) -> None:
        """Set translator instance.

        Args:
            translator: Translator to use
        """
        self._translator = translator

    def process(
        self,
        text: str,
        *,
        target_language: str | None = None,
        translate: bool = True,
        detect_only: bool = False,
    ) -> PipelineResult:
        """Process text through the pretok pipeline.

        Args:
            text: Input text to process
            target_language: Override target language for this call
            translate: Whether to perform translation
            detect_only: Only detect languages, don't translate

        Returns:
            PipelineResult with processed text and metadata
        """
        if not text:
            return PipelineResult(
                original_text=text,
                processed_text=text,
            )

        target = target_language or self._target_language

        # Check cache
        if self._cache and self._translator:
            cache_key = make_cache_key(text, None, target, self._translator.name)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for translation")
                return PipelineResult(
                    original_text=text,
                    processed_text=cached,
                    from_cache=True,
                )

        # Parse into segments
        segments = lex_prompt(text)

        # Detect languages for text segments
        detections: list[DetectionResult] = []
        for i, seg in enumerate(segments):
            if seg.type == SegmentType.TEXT and seg.content.strip():
                try:
                    detection = self._detector.detect(seg.content)
                    detections.append(detection)
                    segments[i] = seg.with_language(detection.language)
                except DetectionError as e:
                    logger.warning("Detection failed for segment: %s", e)

        if detect_only:
            return PipelineResult(
                original_text=text,
                processed_text=text,
                segments=segments,
                detections=detections,
            )

        # Translate segments if needed
        translations: list[TranslationResult] = []
        processed_segments = []

        for seg in segments:
            if self._should_translate(seg, target) and translate and self._translator:
                try:
                    result = self._translator.translate(
                        seg.content,
                        target,
                        seg.language,
                    )
                    translations.append(result)
                    processed_segments.append(seg.with_content(result.translated_text))
                except TranslationError as e:
                    logger.warning("Translation failed: %s", e)
                    processed_segments.append(seg)
            else:
                processed_segments.append(seg)

        # Reconstruct text
        processed_text = segments_to_text(processed_segments)

        # Cache result
        if self._cache and self._translator and translations:
            cache_key = make_cache_key(text, None, target, self._translator.name)
            self._cache.set(cache_key, processed_text)

        return PipelineResult(
            original_text=text,
            processed_text=processed_text,
            segments=processed_segments,
            detections=detections,
            translations=translations,
            metadata={
                "target_language": target,
                "segments_count": len(segments),
                "translated_count": len(translations),
            },
        )

    def _should_translate(self, segment: Segment, target_language: str) -> bool:
        """Determine if a segment should be translated.

        Args:
            segment: Segment to check
            target_language: Target language

        Returns:
            True if segment should be translated
        """
        # Non-translatable segments
        if not segment.translatable:
            return False

        # No language detected
        if not segment.language:
            return False

        # Already in target language
        if segment.language.lower() == target_language.lower():
            return False

        # Check model capability if available
        if self._capability:
            return self._capability.needs_translation(segment.language)

        return True

    def detect(self, text: str) -> DetectionResult:
        """Detect the language of text.

        Convenience method for single-text detection.

        Args:
            text: Text to detect

        Returns:
            DetectionResult with language and confidence
        """
        return self._detector.detect(text)

    def translate(
        self,
        text: str,
        target_language: str | None = None,
        source_language: str | None = None,
    ) -> TranslationResult:
        """Translate text to target language.

        Convenience method for single-text translation.

        Args:
            text: Text to translate
            target_language: Target language (uses default if not specified)
            source_language: Source language (auto-detect if not specified)

        Returns:
            TranslationResult with translated text

        Raises:
            ValueError: If no translator is configured
        """
        if not self._translator:
            raise ValueError("No translator configured")

        target = target_language or self._target_language
        return self._translator.translate(text, target, source_language)


def create_pretok(
    config: PretokConfig | None = None,
    *,
    target_language: str | None = None,
    model_id: str | None = None,
) -> Pretok:
    """Create a Pretok instance.

    Factory function for creating configured Pretok instances.

    Args:
        config: Configuration object
        target_language: Target language (uses model's primary if not specified)
        model_id: Model ID for capability lookup

    Returns:
        Configured Pretok instance
    """
    return Pretok(
        config=config,
        target_language=target_language,
        model_id=model_id,
    )
