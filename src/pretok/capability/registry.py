"""Model capability registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelCapability:
    """Represents a model's language capabilities.

    Attributes:
        model_id: Unique identifier for the model (e.g., 'gpt-4', 'llama-3.2')
        supported_languages: List of ISO 639-1 language codes the model supports
        primary_language: The model's primary/best language
        fallback_language: Language to fall back to if primary not available
        token_efficiency: Relative efficiency map for languages (1.0 = baseline)
        metadata: Additional model metadata
    """

    model_id: str
    supported_languages: list[str] = field(default_factory=lambda: ["en"])
    primary_language: str = "en"
    fallback_language: str | None = None
    token_efficiency: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports_language(self, language: str) -> bool:
        """Check if the model supports a language.

        Args:
            language: ISO 639-1 language code

        Returns:
            True if the language is supported
        """
        return language.lower() in [lang.lower() for lang in self.supported_languages]

    def get_token_efficiency(self, language: str) -> float:
        """Get relative token efficiency for a language.

        Returns a multiplier indicating how efficiently the model
        tokenizes text in the given language compared to English.

        A value < 1.0 means more efficient (fewer tokens)
        A value > 1.0 means less efficient (more tokens)

        Args:
            language: ISO 639-1 language code

        Returns:
            Efficiency multiplier (default 1.0 for unknown languages)
        """
        return self.token_efficiency.get(language.lower(), 1.0)

    def needs_translation(self, source_language: str) -> bool:
        """Determine if text in source language needs translation.

        Args:
            source_language: ISO 639-1 code of the source text

        Returns:
            True if translation to primary language would be beneficial
        """
        source = source_language.lower()

        # Already in primary language
        if source == self.primary_language.lower():
            return False

        # Not supported - needs translation
        if not self.supports_language(source):
            return True

        # Supported but less efficient than primary
        source_efficiency = self.get_token_efficiency(source)
        primary_efficiency = self.get_token_efficiency(self.primary_language)

        # Only translate if it saves significant tokens (>20% improvement)
        return source_efficiency > primary_efficiency * 1.2

    def get_best_target_language(self, source_language: str) -> str:
        """Get the best target language for translation.

        Args:
            source_language: ISO 639-1 code of the source text

        Returns:
            Best target language code
        """
        # If source is supported and efficient, keep it
        if not self.needs_translation(source_language):
            return source_language

        # Otherwise, use primary language
        return self.primary_language


class ModelRegistry:
    """Registry for model capabilities.

    Manages a collection of model capabilities and provides
    lookup and matching functionality.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register(ModelCapability(
        ...     model_id="gpt-4",
        ...     supported_languages=["en", "zh", "ja", "ko", "es", "fr", "de"],
        ...     primary_language="en"
        ... ))
        >>> cap = registry.get("gpt-4")
        >>> cap.supports_language("zh")
        True
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._models: dict[str, ModelCapability] = {}
        self._patterns: list[tuple[str, ModelCapability]] = []

    def register(self, capability: ModelCapability) -> None:
        """Register a model capability.

        Args:
            capability: ModelCapability to register
        """
        self._models[capability.model_id.lower()] = capability
        logger.debug("Registered model capability: %s", capability.model_id)

    def register_pattern(self, pattern: str, capability: ModelCapability) -> None:
        """Register a capability pattern for model families.

        Patterns use glob-style matching (e.g., 'gpt-4*' matches 'gpt-4-turbo')

        Args:
            pattern: Glob pattern for model IDs
            capability: ModelCapability to apply to matching models
        """
        self._patterns.append((pattern.lower(), capability))
        logger.debug("Registered pattern capability: %s", pattern)

    def get(self, model_id: str) -> ModelCapability:
        """Get capability for a model.

        First checks exact matches, then pattern matches.
        Returns a default capability if no match found.

        Args:
            model_id: Model identifier

        Returns:
            ModelCapability for the model
        """
        model_id_lower = model_id.lower()

        # Check exact match
        if model_id_lower in self._models:
            return self._models[model_id_lower]

        # Check patterns
        import fnmatch

        for pattern, capability in self._patterns:
            if fnmatch.fnmatch(model_id_lower, pattern):
                return capability

        # Return default capability
        logger.debug("No capability found for %s, using default", model_id)
        return ModelCapability(model_id=model_id)

    def list_models(self) -> list[str]:
        """List all registered model IDs.

        Returns:
            List of model IDs
        """
        return list(self._models.keys())

    def load_from_yaml(self, path: str | Path) -> None:
        """Load model capabilities from YAML file.

        YAML format:
        ```yaml
        models:
          gpt-4:
            supported_languages: [en, zh, ja, ko, es, fr, de]
            primary_language: en
            token_efficiency:
              en: 1.0
              zh: 1.5
              ja: 1.8
        patterns:
          "gpt-4*":
            supported_languages: [en, zh, ja]
            primary_language: en
        ```

        Args:
            path: Path to YAML file
        """
        path = Path(path)

        with path.open() as f:
            data = yaml.safe_load(f)

        if not data:
            return

        # Load individual models
        for model_id, config in data.get("models", {}).items():
            capability = ModelCapability(
                model_id=model_id,
                supported_languages=config.get("supported_languages", ["en"]),
                primary_language=config.get("primary_language", "en"),
                fallback_language=config.get("fallback_language"),
                token_efficiency=config.get("token_efficiency", {}),
                metadata=config.get("metadata", {}),
            )
            self.register(capability)

        # Load patterns
        for pattern, config in data.get("patterns", {}).items():
            capability = ModelCapability(
                model_id=pattern,
                supported_languages=config.get("supported_languages", ["en"]),
                primary_language=config.get("primary_language", "en"),
                fallback_language=config.get("fallback_language"),
                token_efficiency=config.get("token_efficiency", {}),
                metadata=config.get("metadata", {}),
            )
            self.register_pattern(pattern, capability)

        logger.info(
            "Loaded %d models and %d patterns from %s",
            len(data.get("models", {})),
            len(data.get("patterns", {})),
            path,
        )

    def load_defaults(self) -> None:
        """Load default model capabilities."""
        # OpenAI models
        openai_multilingual = ModelCapability(
            model_id="openai-multilingual",
            supported_languages=[
                "en",
                "zh",
                "ja",
                "ko",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "ru",
                "ar",
                "hi",
                "th",
                "vi",
                "nl",
                "pl",
                "tr",
                "he",
                "id",
                "cs",
            ],
            primary_language="en",
            token_efficiency={
                "en": 1.0,
                "zh": 1.4,
                "ja": 1.8,
                "ko": 1.5,
                "es": 1.1,
                "fr": 1.1,
                "de": 1.2,
                "ru": 1.3,
                "ar": 1.4,
            },
        )

        # GPT-4 family
        self.register_pattern("gpt-4*", openai_multilingual)
        self.register_pattern("gpt-3.5*", openai_multilingual)
        self.register_pattern("gpt-4o*", openai_multilingual)

        # Claude models
        claude_multilingual = ModelCapability(
            model_id="claude-multilingual",
            supported_languages=[
                "en",
                "zh",
                "ja",
                "ko",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "ru",
                "ar",
                "hi",
                "th",
                "vi",
                "nl",
                "pl",
                "tr",
            ],
            primary_language="en",
            token_efficiency={
                "en": 1.0,
                "zh": 1.3,
                "ja": 1.6,
                "ko": 1.4,
            },
        )
        self.register_pattern("claude*", claude_multilingual)

        # Llama models (English-centric)
        llama_english = ModelCapability(
            model_id="llama-english",
            supported_languages=["en", "es", "fr", "de", "it", "pt"],
            primary_language="en",
            token_efficiency={
                "en": 1.0,
                "es": 1.2,
                "fr": 1.2,
                "de": 1.3,
            },
        )
        self.register_pattern("llama*", llama_english)
        self.register_pattern("meta-llama*", llama_english)

        # Qwen models (Chinese-optimized)
        qwen_multilingual = ModelCapability(
            model_id="qwen-multilingual",
            supported_languages=["en", "zh", "ja", "ko"],
            primary_language="zh",
            token_efficiency={
                "zh": 1.0,
                "en": 1.1,
                "ja": 1.3,
                "ko": 1.3,
            },
        )
        self.register_pattern("qwen*", qwen_multilingual)

        # Mistral models
        mistral_multilingual = ModelCapability(
            model_id="mistral-multilingual",
            supported_languages=["en", "fr", "es", "de", "it", "pt"],
            primary_language="en",
            token_efficiency={
                "en": 1.0,
                "fr": 1.1,
                "es": 1.1,
                "de": 1.2,
            },
        )
        self.register_pattern("mistral*", mistral_multilingual)

        logger.info("Loaded default model capabilities")


# Global registry instance
_default_registry: ModelRegistry | None = None


def get_default_registry() -> ModelRegistry:
    """Get the default model registry with built-in capabilities.

    The registry is lazily initialized and cached.

    Returns:
        Default ModelRegistry instance
    """
    global _default_registry

    if _default_registry is None:
        _default_registry = ModelRegistry()
        _default_registry.load_defaults()

    return _default_registry
