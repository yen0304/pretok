"""Tests for model capability module."""

from __future__ import annotations

from pathlib import Path

import yaml

from pretok.capability import (
    ModelCapability,
    ModelRegistry,
    get_default_registry,
)


class TestModelCapability:
    """Tests for ModelCapability dataclass."""

    def test_create_default(self) -> None:
        """Test creating capability with defaults."""
        cap = ModelCapability(model_id="test-model")
        assert cap.model_id == "test-model"
        assert cap.supported_languages == ["en"]
        assert cap.primary_language == "en"
        assert cap.fallback_language is None

    def test_create_custom(self) -> None:
        """Test creating capability with custom values."""
        cap = ModelCapability(
            model_id="multilingual-model",
            supported_languages=["en", "zh", "ja"],
            primary_language="en",
            fallback_language="zh",
        )
        assert cap.model_id == "multilingual-model"
        assert "zh" in cap.supported_languages
        assert cap.fallback_language == "zh"

    def test_supports_language(self) -> None:
        """Test language support check."""
        cap = ModelCapability(
            model_id="test",
            supported_languages=["en", "zh", "ja"],
        )
        assert cap.supports_language("en") is True
        assert cap.supports_language("EN") is True  # Case insensitive
        assert cap.supports_language("zh") is True
        assert cap.supports_language("ko") is False

    def test_token_efficiency(self) -> None:
        """Test token efficiency lookup."""
        cap = ModelCapability(
            model_id="test",
            token_efficiency={
                "en": 1.0,
                "zh": 1.5,
                "ja": 1.8,
            },
        )
        assert cap.get_token_efficiency("en") == 1.0
        assert cap.get_token_efficiency("zh") == 1.5
        assert cap.get_token_efficiency("ko") == 1.0  # Default

    def test_needs_translation_unsupported(self) -> None:
        """Test translation need for unsupported language."""
        cap = ModelCapability(
            model_id="test",
            supported_languages=["en"],
            primary_language="en",
        )
        # Korean not supported, needs translation
        assert cap.needs_translation("ko") is True

    def test_needs_translation_primary(self) -> None:
        """Test no translation needed for primary language."""
        cap = ModelCapability(
            model_id="test",
            supported_languages=["en", "zh"],
            primary_language="en",
        )
        assert cap.needs_translation("en") is False

    def test_needs_translation_efficiency(self) -> None:
        """Test translation based on efficiency."""
        cap = ModelCapability(
            model_id="test",
            supported_languages=["en", "zh", "ja"],
            primary_language="en",
            token_efficiency={
                "en": 1.0,
                "zh": 1.5,  # 50% more tokens
                "ja": 1.2,  # 20% more tokens
            },
        )
        # Chinese is supported but significantly less efficient
        assert cap.needs_translation("zh") is True
        # Japanese is supported and close enough in efficiency
        assert cap.needs_translation("ja") is False

    def test_get_best_target_language(self) -> None:
        """Test getting best target language."""
        cap = ModelCapability(
            model_id="test",
            supported_languages=["en", "zh"],
            primary_language="en",
        )
        # English stays English
        assert cap.get_best_target_language("en") == "en"
        # Unsupported goes to primary
        assert cap.get_best_target_language("ko") == "en"


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving models."""
        registry = ModelRegistry()
        cap = ModelCapability(
            model_id="my-model",
            supported_languages=["en", "zh"],
        )
        registry.register(cap)

        retrieved = registry.get("my-model")
        assert retrieved.model_id == "my-model"
        assert "zh" in retrieved.supported_languages

    def test_get_case_insensitive(self) -> None:
        """Test that model lookup is case insensitive."""
        registry = ModelRegistry()
        cap = ModelCapability(model_id="GPT-4")
        registry.register(cap)

        assert registry.get("gpt-4").model_id == "GPT-4"
        assert registry.get("GPT-4").model_id == "GPT-4"

    def test_get_unknown_returns_default(self) -> None:
        """Test that unknown model returns default capability."""
        registry = ModelRegistry()
        cap = registry.get("unknown-model")

        assert cap.model_id == "unknown-model"
        assert cap.supported_languages == ["en"]

    def test_register_pattern(self) -> None:
        """Test pattern-based registration."""
        registry = ModelRegistry()
        cap = ModelCapability(
            model_id="gpt-pattern",
            supported_languages=["en", "zh", "ja"],
        )
        registry.register_pattern("gpt-4*", cap)

        # Should match pattern
        result = registry.get("gpt-4-turbo")
        assert "zh" in result.supported_languages

        result = registry.get("gpt-4o-mini")
        assert "ja" in result.supported_languages

        # Should not match
        result = registry.get("gpt-3.5-turbo")
        assert result.supported_languages == ["en"]  # Default

    def test_exact_match_takes_priority(self) -> None:
        """Test that exact match takes priority over pattern."""
        registry = ModelRegistry()

        pattern_cap = ModelCapability(
            model_id="gpt-pattern",
            supported_languages=["en"],
        )
        registry.register_pattern("gpt-4*", pattern_cap)

        exact_cap = ModelCapability(
            model_id="gpt-4-turbo",
            supported_languages=["en", "zh", "ja", "ko"],
        )
        registry.register(exact_cap)

        result = registry.get("gpt-4-turbo")
        assert "ko" in result.supported_languages

    def test_list_models(self) -> None:
        """Test listing registered models."""
        registry = ModelRegistry()
        registry.register(ModelCapability(model_id="model-a"))
        registry.register(ModelCapability(model_id="model-b"))

        models = registry.list_models()
        assert "model-a" in models
        assert "model-b" in models
        assert len(models) == 2

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Test loading from YAML file."""
        yaml_content = {
            "models": {
                "test-model": {
                    "supported_languages": ["en", "zh", "ja"],
                    "primary_language": "en",
                    "token_efficiency": {"en": 1.0, "zh": 1.5},
                }
            },
            "patterns": {
                "test-*": {
                    "supported_languages": ["en", "fr"],
                    "primary_language": "en",
                }
            },
        }

        yaml_file = tmp_path / "models.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        registry = ModelRegistry()
        registry.load_from_yaml(yaml_file)

        # Check exact model
        cap = registry.get("test-model")
        assert "ja" in cap.supported_languages
        assert cap.get_token_efficiency("zh") == 1.5

        # Check pattern
        cap = registry.get("test-other")
        assert "fr" in cap.supported_languages


class TestDefaultRegistry:
    """Tests for default registry."""

    def test_get_default_registry(self) -> None:
        """Test getting default registry."""
        registry = get_default_registry()
        assert isinstance(registry, ModelRegistry)

    def test_gpt4_capability(self) -> None:
        """Test GPT-4 pattern capability."""
        registry = get_default_registry()
        cap = registry.get("gpt-4-turbo")

        assert cap.supports_language("en")
        assert cap.supports_language("zh")
        assert cap.supports_language("ja")
        assert cap.primary_language == "en"

    def test_claude_capability(self) -> None:
        """Test Claude pattern capability."""
        registry = get_default_registry()
        cap = registry.get("claude-3-opus")

        assert cap.supports_language("en")
        assert cap.supports_language("zh")

    def test_llama_capability(self) -> None:
        """Test Llama pattern capability."""
        registry = get_default_registry()
        cap = registry.get("llama-3.2-8b")

        assert cap.supports_language("en")
        assert cap.supports_language("es")
        # Llama is English-centric
        assert cap.primary_language == "en"

    def test_qwen_capability(self) -> None:
        """Test Qwen pattern capability."""
        registry = get_default_registry()
        cap = registry.get("qwen-2.5-72b")

        assert cap.supports_language("zh")
        assert cap.supports_language("en")
        # Qwen is Chinese-optimized
        assert cap.primary_language == "zh"

    def test_unknown_model(self) -> None:
        """Test unknown model returns default."""
        registry = get_default_registry()
        cap = registry.get("completely-unknown-model-xyz")

        assert cap.supports_language("en")
        assert cap.primary_language == "en"
