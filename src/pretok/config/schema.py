"""Configuration schema definitions using Pydantic models."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class PipelineConfig(BaseModel):
    """Configuration for the core pipeline."""

    default_detector: str = Field(
        default="langdetect",
        description="Default language detector backend",
    )
    default_translator: str | None = Field(
        default=None,
        description="Default translator backend",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable translation caching",
    )
    cache_backend: str = Field(
        default="memory",
        description="Cache backend to use",
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for language detection",
    )
    fallback_to_original: bool = Field(
        default=True,
        description="Return original text if translation fails",
    )
    strict_mode: bool = Field(
        default=False,
        description="Raise exceptions on failures instead of falling back",
    )


class FastTextConfig(BaseModel):
    """Configuration for FastText detector."""

    model_path: str | None = Field(
        default=None,
        description="Path to FastText language identification model",
    )
    k: int = Field(
        default=3,
        ge=1,
        description="Number of predictions to return",
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )


class LangDetectConfig(BaseModel):
    """Configuration for langdetect detector."""

    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )


class CompositeDetectorConfig(BaseModel):
    """Configuration for composite detector."""

    detectors: list[str] = Field(
        default_factory=lambda: ["langdetect"],
        description="List of detector backends to use",
    )
    strategy: str = Field(
        default="voting",
        description="Aggregation strategy: voting, weighted_average, fallback_chain",
    )
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Weights for weighted_average strategy",
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"voting", "weighted_average", "fallback_chain"}
        if v not in valid:
            msg = f"strategy must be one of {valid}"
            raise ValueError(msg)
        return v


class DetectionConfig(BaseModel):
    """Configuration for language detection."""

    fasttext: FastTextConfig = Field(default_factory=FastTextConfig)
    langdetect: LangDetectConfig = Field(default_factory=LangDetectConfig)
    composite: CompositeDetectorConfig = Field(default_factory=CompositeDetectorConfig)


class LLMTranslatorConfig(BaseModel):
    """Configuration for LLM-based translator (OpenAI-compatible APIs)."""

    base_url: str | None = Field(
        default=None,
        description="API base URL (OpenAI, OpenRouter, Ollama, vLLM, etc.)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (if not using env var)",
    )
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable name for API key",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Model name to use",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for translation",
    )
    user_prompt_template: str | None = Field(
        default=None,
        description="Custom user prompt template with {text}, {source}, {target}",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens for API response. If set, overrides max_tokens_multiplier. "
        "Recommended for thinking models (qwen3, DeepSeek-R1) that need more tokens.",
    )
    max_tokens_multiplier: int = Field(
        default=4,
        ge=1,
        description="Multiplier for calculating max_tokens from input length. "
        "max_tokens = len(text) * multiplier. Ignored if max_tokens is set.",
    )

    def get_api_key(self) -> str | None:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)


class HuggingFaceTranslatorConfig(BaseModel):
    """Configuration for HuggingFace translation models."""

    model_name: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="HuggingFace model ID",
    )
    device: str = Field(
        default="auto",
        description="Device to use: cpu, cuda, mps, auto",
    )
    max_length: int = Field(
        default=512,
        ge=1,
        description="Maximum sequence length",
    )
    num_beams: int = Field(
        default=4,
        ge=1,
        description="Number of beams for beam search",
    )
    language_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="ISO code to model-specific language code mapping",
    )


class GoogleTranslatorConfig(BaseModel):
    """Configuration for Google Cloud Translation."""

    project_id: str | None = Field(
        default=None,
        description="Google Cloud project ID",
    )
    location: str = Field(
        default="global",
        description="Google Cloud location",
    )
    credentials_path: str | None = Field(
        default=None,
        description="Path to service account credentials JSON",
    )


class DeepLTranslatorConfig(BaseModel):
    """Configuration for DeepL translator."""

    api_key: str | None = Field(
        default=None,
        description="DeepL API key",
    )
    api_key_env: str = Field(
        default="DEEPL_API_KEY",
        description="Environment variable name for API key",
    )
    formality: str = Field(
        default="default",
        description="Formality level: more, less, default",
    )
    use_free_api: bool = Field(
        default=False,
        description="Use DeepL free API endpoint",
    )

    @field_validator("formality")
    @classmethod
    def validate_formality(cls, v: str) -> str:
        valid = {"more", "less", "default"}
        if v not in valid:
            msg = f"formality must be one of {valid}"
            raise ValueError(msg)
        return v

    def get_api_key(self) -> str | None:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)


class TranslationConfig(BaseModel):
    """Configuration for translation engines."""

    llm: LLMTranslatorConfig = Field(default_factory=LLMTranslatorConfig)
    huggingface: HuggingFaceTranslatorConfig = Field(default_factory=HuggingFaceTranslatorConfig)
    google: GoogleTranslatorConfig = Field(default_factory=GoogleTranslatorConfig)
    deepl: DeepLTranslatorConfig = Field(default_factory=DeepLTranslatorConfig)


class MemoryCacheConfig(BaseModel):
    """Configuration for in-memory cache."""

    max_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of cached entries",
    )
    ttl: int = Field(
        default=3600,
        ge=0,
        description="Time-to-live in seconds (0 = no expiry)",
    )


class RedisCacheConfig(BaseModel):
    """Configuration for Redis cache."""

    url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
    )
    prefix: str = Field(
        default="pretok",
        description="Key prefix for cache entries",
    )
    ttl: int = Field(
        default=86400,
        ge=0,
        description="Time-to-live in seconds",
    )


class SQLiteCacheConfig(BaseModel):
    """Configuration for SQLite cache."""

    path: str = Field(
        default="~/.pretok/cache.db",
        description="Path to SQLite database file",
    )
    ttl: int = Field(
        default=604800,
        ge=0,
        description="Time-to-live in seconds",
    )


class CacheConfig(BaseModel):
    """Configuration for caching."""

    memory: MemoryCacheConfig = Field(default_factory=MemoryCacheConfig)
    redis: RedisCacheConfig = Field(default_factory=RedisCacheConfig)
    sqlite: SQLiteCacheConfig = Field(default_factory=SQLiteCacheConfig)


class CustomMarkerConfig(BaseModel):
    """Configuration for custom segment markers."""

    pattern: str = Field(description="Pattern to match (literal or regex)")
    type: str = Field(description="Segment type: ROLE_MARKER, CONTROL_TOKEN, DELIMITER")
    is_regex: bool = Field(default=False, description="Whether pattern is a regex")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid = {"ROLE_MARKER", "CONTROL_TOKEN", "DELIMITER"}
        if v not in valid:
            msg = f"type must be one of {valid}"
            raise ValueError(msg)
        return v


class SegmentConfig(BaseModel):
    """Configuration for segment processing."""

    preserve_code_blocks: bool = Field(
        default=True,
        description="Preserve code blocks without translation",
    )
    translate_code_comments: bool = Field(
        default=False,
        description="Translate comments within code blocks",
    )
    translate_json_strings: bool = Field(
        default=False,
        description="Translate string values in JSON",
    )
    custom_markers: list[CustomMarkerConfig] = Field(
        default_factory=list,
        description="Custom segment markers",
    )
    format_hint: str | None = Field(
        default=None,
        description="Force specific prompt format: chatml, llama, alpaca",
    )


class ModelCapabilityConfig(BaseModel):
    """Configuration for a single model's language capabilities."""

    supported_languages: list[str] = Field(
        default_factory=lambda: ["en"],
        description="List of supported language codes",
    )
    primary_language: str = Field(
        default="en",
        description="Primary/preferred language",
    )
    fallback_language: str | None = Field(
        default=None,
        description="Fallback language when primary not available",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model constraints",
    )

    @model_validator(mode="after")
    def validate_primary_in_supported(self) -> ModelCapabilityConfig:
        """Ensure primary language is in supported languages."""
        if self.primary_language not in self.supported_languages:
            self.supported_languages.append(self.primary_language)
        return self


class ModelsConfig(BaseModel):
    """Configuration for model capabilities."""

    default: ModelCapabilityConfig = Field(default_factory=ModelCapabilityConfig)

    model_config = {"extra": "allow"}

    def get_capability(self, model_id: str) -> ModelCapabilityConfig:
        """Get capability for a model, falling back to default."""
        if hasattr(self, model_id):
            cap = getattr(self, model_id)
            if isinstance(cap, dict):
                return ModelCapabilityConfig(**cap)
            if isinstance(cap, ModelCapabilityConfig):
                return cap
        return self.default


class PretokConfig(BaseModel):
    """Root configuration for pretok."""

    version: str = Field(
        default="1.0",
        description="Configuration schema version",
    )
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
