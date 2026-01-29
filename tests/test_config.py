"""Tests for configuration module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pretok.config import (
    CacheConfig,
    ConfigurationError,
    LLMTranslatorConfig,
    PipelineConfig,
    PretokConfig,
    TranslationConfig,
    find_config_file,
    get_default_config,
    load_config,
    load_config_file,
)


class TestPretokConfig:
    """Tests for PretokConfig schema."""

    def test_default_config(self) -> None:
        """Test that default config can be created."""
        config = PretokConfig()
        assert config.pipeline.default_detector == "langdetect"
        assert config.pipeline.cache_enabled is True

    def test_custom_target_language(self) -> None:
        """Test setting custom default detector."""
        config = PretokConfig(pipeline=PipelineConfig(default_detector="fasttext"))
        assert config.pipeline.default_detector == "fasttext"

    def test_detection_config_defaults(self) -> None:
        """Test detection config defaults."""
        config = PretokConfig()
        assert config.pipeline.confidence_threshold == 0.8

    def test_custom_confidence_threshold(self) -> None:
        """Test custom confidence threshold."""
        config = PretokConfig(pipeline=PipelineConfig(confidence_threshold=0.9))
        assert config.pipeline.confidence_threshold == 0.9

    def test_translation_config(self) -> None:
        """Test translation config."""
        config = PretokConfig(
            pipeline=PipelineConfig(default_translator="llm"),
            translation=TranslationConfig(
                llm=LLMTranslatorConfig(
                    base_url="https://api.openai.com/v1",
                    model="gpt-4o-mini",
                    api_key="test-key",
                ),
            ),
        )
        assert config.pipeline.default_translator == "llm"
        assert config.translation.llm is not None
        assert config.translation.llm.model == "gpt-4o-mini"

    def test_cache_config_memory(self) -> None:
        """Test memory cache config."""
        config = PretokConfig(pipeline=PipelineConfig(cache_enabled=True, cache_backend="memory"))
        assert config.pipeline.cache_enabled is True
        assert config.pipeline.cache_backend == "memory"

    def test_cache_config_redis(self) -> None:
        """Test redis cache config."""
        from pretok.config import RedisCacheConfig

        config = PretokConfig(
            pipeline=PipelineConfig(cache_enabled=True, cache_backend="redis"),
            cache=CacheConfig(
                redis=RedisCacheConfig(url="redis://localhost:6379/0"),
            ),
        )
        assert config.pipeline.cache_backend == "redis"
        assert config.cache.redis is not None
        assert config.cache.redis.url == "redis://localhost:6379/0"


class TestLLMTranslatorConfig:
    """Tests for LLM translator configuration."""

    def test_openai_config(self) -> None:
        """Test OpenAI configuration."""
        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="sk-xxx",
        )
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-4o-mini"

    def test_openrouter_config(self) -> None:
        """Test OpenRouter configuration."""
        config = LLMTranslatorConfig(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3-haiku",
            api_key="sk-or-xxx",
        )
        assert "openrouter" in config.base_url

    def test_ollama_config(self) -> None:
        """Test Ollama local configuration."""
        config = LLMTranslatorConfig(
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key=None,
        )
        assert config.base_url == "http://localhost:11434/v1"
        assert config.api_key is None

    def test_custom_prompts(self) -> None:
        """Test custom prompt templates."""
        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            system_prompt="You are a professional translator.",
            user_prompt_template="Translate this to {target_language}: {text}",
        )
        assert "professional translator" in config.system_prompt
        assert "{target_language}" in config.user_prompt_template


class TestLoadConfigFile:
    """Tests for loading configuration files."""

    def test_load_yaml(self, tmp_path: Path) -> None:
        """Test loading YAML config."""
        config_data = {
            "pipeline": {"default_detector": "fasttext", "cache_enabled": True},
            "detection": {"fasttext": {"k": 5}},
        }
        config_file = tmp_path / "pretok.yaml"
        config_file.write_text(yaml.dump(config_data))

        loaded = load_config_file(config_file)
        assert loaded["pipeline"]["default_detector"] == "fasttext"

    def test_load_json(self, tmp_path: Path) -> None:
        """Test loading JSON config."""
        config_data = {
            "pipeline": {"default_detector": "langdetect"},
        }
        config_file = tmp_path / "pretok.json"
        config_file.write_text(json.dumps(config_data))

        loaded = load_config_file(config_file)
        assert loaded["pipeline"]["default_detector"] == "langdetect"

    def test_load_toml(self, tmp_path: Path) -> None:
        """Test loading TOML config."""
        toml_content = """
[pipeline]
default_detector = "fasttext"
cache_enabled = true
"""
        config_file = tmp_path / "pretok.toml"
        config_file.write_text(toml_content)

        loaded = load_config_file(config_file)
        assert loaded["pipeline"]["default_detector"] == "fasttext"

    def test_file_not_found(self) -> None:
        """Test loading non-existent file."""
        with pytest.raises(ConfigurationError, match="not found"):
            load_config_file("/nonexistent/path/config.yaml")

    def test_invalid_format(self, tmp_path: Path) -> None:
        """Test loading unsupported format."""
        config_file = tmp_path / "config.xyz"
        config_file.write_text("data")

        with pytest.raises(ConfigurationError, match="Unsupported"):
            load_config_file(config_file)

    def test_env_var_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable resolution."""
        monkeypatch.setenv("PRETOK_API_KEY", "secret-key-123")

        config_data = {
            "translation": {
                "llm": {
                    "api_key": "${PRETOK_API_KEY}",
                    "model": "gpt-4o-mini",
                    "base_url": "https://api.openai.com/v1",
                }
            }
        }
        config_file = tmp_path / "pretok.yaml"
        config_file.write_text(yaml.dump(config_data))

        loaded = load_config_file(config_file)
        assert loaded["translation"]["llm"]["api_key"] == "secret-key-123"


class TestFindConfigFile:
    """Tests for config file discovery."""

    def test_find_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test finding pretok.yaml."""
        config_file = tmp_path / "pretok.yaml"
        config_file.write_text("pipeline:\n  default_detector: langdetect")
        monkeypatch.chdir(tmp_path)

        found = find_config_file()
        assert found is not None
        assert found.name == "pretok.yaml"

    def test_find_hidden_yaml(self, tmp_path: Path) -> None:
        """Test finding .pretok.yaml."""
        config_file = tmp_path / ".pretok.yaml"
        config_file.write_text("pipeline:\n  default_detector: langdetect")

        found = find_config_file(tmp_path)
        assert found is not None
        assert found.name == ".pretok.yaml"

    def test_priority_order(self, tmp_path: Path) -> None:
        """Test that pretok.yaml takes priority over .pretok.yaml."""
        (tmp_path / "pretok.yaml").write_text("# main")
        (tmp_path / ".pretok.yaml").write_text("# hidden")

        found = find_config_file(tmp_path)
        assert found is not None
        assert found.name == "pretok.yaml"

    def test_no_config_found(self, tmp_path: Path) -> None:
        """Test when no config file exists."""
        found = find_config_file(tmp_path)
        assert found is None


class TestLoadConfig:
    """Tests for the main load_config function."""

    def test_load_with_path(self, tmp_path: Path) -> None:
        """Test loading config from explicit path."""
        config_data = {"pipeline": {"default_detector": "fasttext"}}
        config_file = tmp_path / "custom.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        assert config.pipeline.default_detector == "fasttext"

    def test_load_with_overrides(self, tmp_path: Path) -> None:
        """Test loading config with runtime overrides."""
        config_data = {"pipeline": {"default_detector": "fasttext"}}
        config_file = tmp_path / "pretok.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(
            config_file,
            config_dict={"pipeline": {"default_detector": "langdetect"}},
        )
        assert config.pipeline.default_detector == "langdetect"

    def test_load_defaults_only(self) -> None:
        """Test loading with no file and no overrides."""
        config = load_config(auto_discover=False)
        assert config.pipeline.default_detector == "langdetect"

    def test_auto_discover(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test auto-discovery of config file."""
        config_data = {"pipeline": {"default_detector": "fasttext"}}
        config_file = tmp_path / "pretok.yaml"
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.chdir(tmp_path)

        config = load_config()
        assert config.pipeline.default_detector == "fasttext"

    def test_invalid_config_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid config raises ConfigurationError."""
        config_data = {"detection": {"composite": {"strategy": "invalid"}}}
        config_file = tmp_path / "pretok.yaml"
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            load_config(config_file)

    def test_deep_merge(self, tmp_path: Path) -> None:
        """Test deep merging of config dictionaries."""
        config_data = {
            "pipeline": {"default_detector": "fasttext", "cache_enabled": True},
            "detection": {"fasttext": {"k": 5}},
        }
        config_file = tmp_path / "pretok.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_config(
            config_file,
            config_dict={"pipeline": {"cache_enabled": False}},
        )
        # Original value preserved
        assert config.pipeline.default_detector == "fasttext"
        # Override applied
        assert config.pipeline.cache_enabled is False


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_valid_config(self) -> None:
        """Test that default config is valid."""
        config = get_default_config()
        assert isinstance(config, PretokConfig)
        assert config.pipeline.default_detector == "langdetect"
        assert config.pipeline.cache_enabled is True


class TestDeepLTranslatorConfig:
    """Tests for DeepL translator configuration."""

    def test_default_config(self) -> None:
        """Test default DeepL config."""
        from pretok.config.schema import DeepLTranslatorConfig

        config = DeepLTranslatorConfig()
        assert config.formality == "default"
        assert config.use_free_api is False

    def test_formality_validation(self) -> None:
        """Test formality validation."""
        from pretok.config.schema import DeepLTranslatorConfig

        # Valid values
        DeepLTranslatorConfig(formality="more")
        DeepLTranslatorConfig(formality="less")
        DeepLTranslatorConfig(formality="default")

        # Invalid value
        with pytest.raises(ValueError, match="formality must be one of"):
            DeepLTranslatorConfig(formality="invalid")

    def test_get_api_key_from_config(self) -> None:
        """Test getting API key from config."""
        from pretok.config.schema import DeepLTranslatorConfig

        config = DeepLTranslatorConfig(api_key="test-key")
        assert config.get_api_key() == "test-key"

    def test_get_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting API key from environment."""
        from pretok.config.schema import DeepLTranslatorConfig

        monkeypatch.setenv("DEEPL_API_KEY", "env-key")
        config = DeepLTranslatorConfig()
        assert config.get_api_key() == "env-key"


class TestLLMTranslatorConfigApiKey:
    """Tests for LLM translator API key handling."""

    def test_get_api_key_from_config(self) -> None:
        """Test getting API key from config."""
        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4",
            api_key="config-key",
        )
        assert config.get_api_key() == "config-key"

    def test_get_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4",
        )
        assert config.get_api_key() == "env-key"


class TestModelsConfig:
    """Tests for models configuration."""

    def test_get_capability_default(self) -> None:
        """Test getting default model capability."""
        from pretok.config import ModelsConfig

        config = ModelsConfig()
        cap = config.get_capability("unknown-model")
        assert cap.primary_language == "en"

    def test_get_capability_custom(self) -> None:
        """Test getting custom model capability."""
        from pretok.config import ModelCapabilityConfig, ModelsConfig

        config = ModelsConfig(
            default=ModelCapabilityConfig(),
            **{"gpt-4": {"supported_languages": ["en", "zh"], "primary_language": "en"}},
        )
        cap = config.get_capability("gpt-4")
        assert "zh" in cap.supported_languages


class TestModelCapabilityConfig:
    """Tests for model capability configuration."""

    def test_primary_added_to_supported(self) -> None:
        """Test that primary language is added to supported if not present."""
        from pretok.config import ModelCapabilityConfig

        cap = ModelCapabilityConfig(supported_languages=["zh"], primary_language="en")
        # Primary should be auto-added
        assert "en" in cap.supported_languages
        assert "zh" in cap.supported_languages


class TestPretokConfigToDict:
    """Tests for PretokConfig.to_dict method."""

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = PretokConfig()
        result = config.to_dict()
        assert isinstance(result, dict)
        assert "pipeline" in result
        assert "detection" in result
        assert "translation" in result
