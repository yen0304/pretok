"""Pretok - Universal pre-token language adaptation layer for LLMs.

Pretok optimizes multilingual input for LLM tokenization by:
1. Detecting languages in text segments
2. Translating to model-optimal languages
3. Preserving prompt structure (roles, code, formatting)

Example:
    >>> from pretok import Pretok
    >>> pretok = Pretok(target_language="en")
    >>> result = pretok.process("Hello World")  # noqa: RUF002
    >>> result.processed_text
    'Hello World'

With LLM translator:
    >>> from pretok import Pretok
    >>> from pretok.config import LLMTranslatorConfig
    >>> from pretok.translation.llm import LLMTranslator
    >>>
    >>> config = LLMTranslatorConfig(
    ...     base_url="https://api.openai.com/v1",
    ...     model="gpt-4o-mini"
    ... )
    >>> translator = LLMTranslator(config)
    >>> pretok = Pretok(target_language="en", translator=translator)
"""

from pretok._version import __version__
from pretok.capability import ModelCapability, ModelRegistry, get_default_registry
from pretok.config import (
    ConfigurationError,
    PretokConfig,
    get_default_config,
    load_config,
)
from pretok.detection import DetectionError, DetectionResult, LanguageDetector
from pretok.pipeline.core import PipelineResult, Pretok, create_pretok
from pretok.segment import Segment, SegmentType
from pretok.translation import TranslationError, TranslationResult, Translator

__all__ = [
    "ConfigurationError",
    "DetectionError",
    "DetectionResult",
    "LanguageDetector",
    "ModelCapability",
    "ModelRegistry",
    "PipelineResult",
    "Pretok",
    "PretokConfig",
    "Segment",
    "SegmentType",
    "TranslationError",
    "TranslationResult",
    "Translator",
    "__version__",
    "create_pretok",
    "get_default_config",
    "get_default_registry",
    "load_config",
]
