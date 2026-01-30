"""Tests for translation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pretok.translation import TranslationError, TranslationResult, Translator
from pretok.translation.base import BaseTranslator


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a translation result."""
        result = TranslationResult(
            source_text="Hello",
            translated_text="你好",
            source_language="en",
            target_language="zh",
            translator="test",
        )
        assert result.source_text == "Hello"
        assert result.translated_text == "你好"
        assert result.source_language == "en"
        assert result.target_language == "zh"

    def test_was_translated_true(self) -> None:
        """Test was_translated when text was translated."""
        result = TranslationResult(
            source_text="Hello",
            translated_text="你好",
            source_language="en",
            target_language="zh",
            translator="test",
        )
        assert result.was_translated is True

    def test_was_translated_false(self) -> None:
        """Test was_translated when text was not translated."""
        result = TranslationResult(
            source_text="Hello",
            translated_text="Hello",
            source_language="en",
            target_language="en",
            translator="test",
        )
        assert result.was_translated is False

    def test_result_with_metadata(self) -> None:
        """Test result with metadata."""
        result = TranslationResult(
            source_text="Hello",
            translated_text="你好",
            source_language="en",
            target_language="zh",
            translator="test",
            confidence=0.95,
            metadata={"model": "gpt-4"},
        )
        assert result.confidence == 0.95
        assert result.metadata["model"] == "gpt-4"


class TestTranslationError:
    """Tests for TranslationError exception."""

    def test_basic_error(self) -> None:
        """Test basic error."""
        error = TranslationError("Translation failed")
        assert str(error) == "Translation failed"

    def test_error_with_context(self) -> None:
        """Test error with full context."""
        cause = ValueError("API error")
        error = TranslationError(
            "Translation failed",
            source_text="Hello",
            source_language="en",
            target_language="zh",
            translator="llm",
            cause=cause,
        )
        assert error.source_text == "Hello"
        assert error.source_language == "en"
        assert error.target_language == "zh"
        assert error.translator == "llm"
        assert error.cause is cause


class TestTranslatorProtocol:
    """Tests for Translator protocol."""

    def test_protocol_compliance(self) -> None:
        """Test that a class implementing the protocol is recognized."""

        class MockTranslator:
            @property
            def name(self) -> str:
                return "mock"

            @property
            def supported_languages(self) -> list[str]:
                return ["en", "zh"]

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_language or "en",
                    target_language=target_language,
                    translator="mock",
                )

            def translate_batch(
                self,
                texts: list[str],
                target_language: str,
                source_language: str | None = None,
            ) -> list[TranslationResult]:
                return [self.translate(t, target_language, source_language) for t in texts]

        translator = MockTranslator()
        assert isinstance(translator, Translator)


class TestBaseTranslator:
    """Tests for BaseTranslator base class."""

    def test_supports_language_with_list(self) -> None:
        """Test supports_language with explicit list."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def supported_languages(self) -> list[str]:
                return ["en", "zh", "ja"]

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_language or "en",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        assert translator.supports_language("en") is True
        assert translator.supports_language("EN") is True  # Case insensitive
        assert translator.supports_language("zh") is True
        assert translator.supports_language("ko") is False

    def test_supports_language_empty_list(self) -> None:
        """Test that empty list means all supported."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def supported_languages(self) -> list[str]:
                return []

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_language or "en",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        assert translator.supports_language("any") is True

    def test_translate_batch_default(self) -> None:
        """Test default translate_batch implementation."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=f"[{text}]",
                    source_language=source_language or "en",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        results = translator.translate_batch(["a", "b", "c"], "zh")

        assert len(results) == 3
        assert results[0].translated_text == "[a]"
        assert results[1].translated_text == "[b]"
        assert results[2].translated_text == "[c]"


class TestLLMTranslator:
    """Tests for LLMTranslator."""

    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        with patch("openai.OpenAI") as mock:
            # Create mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "你好"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock.return_value = mock_client

            yield mock

    def test_translate(self, mock_openai) -> None:
        """Test basic translation."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)
        result = translator.translate("Hello", "zh", "en")

        assert result.translated_text == "你好"
        assert result.source_language == "en"
        assert result.target_language == "zh"

    def test_translate_empty_text(self, mock_openai) -> None:
        """Test translating empty text."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)
        result = translator.translate("", "zh")

        # Empty text should return as-is
        assert result.translated_text == ""
        assert result.was_translated is False

    def test_translator_name(self, mock_openai) -> None:
        """Test translator name includes model."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        assert translator.name == "llm:gpt-4o-mini"


class TestTranslatorFactory:
    """Tests for translator factory."""

    def test_create_unknown_backend(self) -> None:
        """Test creating unknown backend raises error."""
        from pretok.translation.factory import create_translator

        with pytest.raises(ValueError, match="Unknown translator backend"):
            create_translator("unknown")

    def test_create_llm_without_config(self) -> None:
        """Test creating LLM translator without config raises error."""
        from pretok.translation.factory import create_translator

        with pytest.raises(ValueError, match="requires configuration"):
            create_translator("llm")

    def test_create_llm_with_empty_config(self) -> None:
        """Test creating LLM translator with empty TranslationConfig raises error."""
        from pretok.config import TranslationConfig
        from pretok.translation.factory import create_translator

        # TranslationConfig without llm field set properly
        config = TranslationConfig()
        config.llm = None  # type: ignore[assignment]
        with pytest.raises(ValueError, match="requires configuration"):
            create_translator("llm", config)


class TestLLMTranslatorAdvanced:
    """Advanced tests for LLMTranslator."""

    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        with patch("openai.OpenAI") as mock:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "翻譯結果"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock.return_value = mock_client

            yield mock

    def test_translate_without_source_language(self, mock_openai) -> None:
        """Test translation without specifying source language."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)
        result = translator.translate("Hello world", "zh")

        assert result.target_language == "zh"
        assert result.source_language == "auto"

    def test_translate_batch(self, mock_openai) -> None:
        """Test batch translation."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)
        results = translator.translate_batch(["Hello", "World"], "zh", "en")

        assert len(results) == 2
        for result in results:
            assert result.target_language == "zh"

    def test_supported_languages_empty(self, mock_openai) -> None:
        """Test that LLM translator supports all languages (empty list)."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)
        assert translator.supported_languages == []
        assert translator.supports_language("any_language") is True

    def test_custom_prompts(self, mock_openai) -> None:
        """Test translator with custom prompts."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
            system_prompt="You are a translator.",
            user_prompt_template="Translate: {text}",
        )
        translator = LLMTranslator(config)
        result = translator.translate("Hello", "zh")
        assert result is not None

    def test_api_error_handling(self, mock_openai) -> None:
        """Test handling of API errors."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        with pytest.raises(TranslationError):
            translator.translate("Hello", "zh")

    def test_no_api_key_local_api(self, mock_openai) -> None:
        """Test translator with no API key (for local APIs like Ollama)."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="http://localhost:11434/v1",
            model="llama3",
            api_key=None,
        )
        translator = LLMTranslator(config)
        assert translator.name == "llm:llama3"

    def test_default_max_tokens_multiplier(self, mock_openai) -> None:
        """Test default max_tokens calculation uses multiplier of 4."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)
        translator.translate("Hello", "zh", "en")

        # Verify max_tokens = len("Hello") * 4 = 20
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 20

    def test_explicit_max_tokens(self, mock_openai) -> None:
        """Test explicit max_tokens overrides multiplier."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
            max_tokens=500,
        )
        translator = LLMTranslator(config)
        translator.translate("Hello", "zh", "en")

        call_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500

    def test_custom_max_tokens_multiplier(self, mock_openai) -> None:
        """Test custom max_tokens_multiplier for thinking models."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="qwen3:latest",
            api_key="test-key",
            max_tokens_multiplier=15,
        )
        translator = LLMTranslator(config)
        translator.translate("Hello", "zh", "en")

        # Verify max_tokens = len("Hello") * 15 = 75
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 75

    def test_max_tokens_takes_precedence_over_multiplier(self, mock_openai) -> None:
        """Test that max_tokens takes precedence when both are set."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
            max_tokens=300,
            max_tokens_multiplier=100,  # Would be 500 for "Hello"
        )
        translator = LLMTranslator(config)
        translator.translate("Hello", "zh", "en")

        # max_tokens should be 300, not 500
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 300


class TestBaseTranslatorValidation:
    """Tests for BaseTranslator validation methods."""

    def test_validate_languages_success(self) -> None:
        """Test successful language validation."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def supported_languages(self) -> list[str]:
                return ["en", "zh"]

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_language or "en",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        # Should not raise
        translator._validate_languages("en", "zh")

    def test_validate_languages_unsupported_source(self) -> None:
        """Test validation fails for unsupported source language."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def supported_languages(self) -> list[str]:
                return ["en", "zh"]

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_language or "en",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        with pytest.raises(TranslationError, match="not supported"):
            translator._validate_languages("ko", "zh")

    def test_validate_languages_unsupported_target(self) -> None:
        """Test validation fails for unsupported target language."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def supported_languages(self) -> list[str]:
                return ["en", "zh"]

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_language or "en",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        with pytest.raises(TranslationError, match="not supported"):
            translator._validate_languages("en", "ko")


class TestTranslatorFactoryAdvanced:
    """Additional tests for translator factory."""

    def test_create_llm_translator_without_config(self) -> None:
        """Test creating LLM translator without config raises error."""
        from pretok.translation.factory import create_translator

        with pytest.raises(ValueError, match="LLM translator requires configuration"):
            create_translator("llm", config=None)

    def test_create_llm_translator_with_empty_llm_config(self) -> None:
        """Test creating LLM translator with config but no llm section."""
        from unittest.mock import MagicMock

        from pretok.config import TranslationConfig
        from pretok.translation.factory import create_translator

        config = MagicMock(spec=TranslationConfig)
        config.llm = None
        with pytest.raises(ValueError, match="LLM translator requires configuration"):
            create_translator("llm", config=config)

    def test_create_llm_translator_success(self) -> None:
        """Test successfully creating LLM translator."""
        from pretok.config import LLMTranslatorConfig, TranslationConfig
        from pretok.translation.factory import create_translator
        from pretok.translation.llm import LLMTranslator

        llm_config = LLMTranslatorConfig(api_key="test-key")
        config = TranslationConfig(llm=llm_config)
        translator = create_translator("llm", config=config)
        assert isinstance(translator, LLMTranslator)

    def test_create_unknown_backend(self) -> None:
        """Test creating unknown backend raises error."""
        from pretok.translation.factory import create_translator

        with pytest.raises(ValueError, match="Unknown translator backend"):
            create_translator("unknown_backend", config=None)


class TestBaseTranslatorAdvanced:
    """Advanced tests for BaseTranslator."""

    def test_translate_batch_default_implementation(self) -> None:
        """Test translate_batch default implementation."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=f"[{target_language}] {text}",
                    source_language=source_language or "en",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        results = translator.translate_batch(["Hello", "World"], "zh")
        assert len(results) == 2
        assert results[0].translated_text == "[zh] Hello"
        assert results[1].translated_text == "[zh] World"

    def test_supports_language_empty_list(self) -> None:
        """Test supports_language when supported_languages is empty returns True."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def supported_languages(self) -> list[str]:
                return []  # Empty means all supported

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_language or "auto",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        assert translator.supports_language("en") is True
        assert translator.supports_language("zh") is True
        assert translator.supports_language("any_random") is True

    def test_supports_language_with_list(self) -> None:
        """Test supports_language with specific language list."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def supported_languages(self) -> list[str]:
                return ["en", "zh", "ja"]

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_language or "auto",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        assert translator.supports_language("en") is True
        assert translator.supports_language("EN") is True  # Case insensitive
        assert translator.supports_language("ko") is False

    def test_supports_language_case_insensitive(self) -> None:
        """Test supports_language is case insensitive."""

        class TestTranslator(BaseTranslator):
            @property
            def name(self) -> str:
                return "test"

            @property
            def supported_languages(self) -> list[str]:
                return ["EN", "ZH"]

            def translate(
                self,
                text: str,
                target_language: str,
                source_language: str | None = None,
            ) -> TranslationResult:
                return TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_language=source_language or "auto",
                    target_language=target_language,
                    translator=self.name,
                )

        translator = TestTranslator()
        assert translator.supports_language("en") is True
        assert translator.supports_language("zh") is True


class TestLLMTranslatorCleanTranslation:
    """Tests for LLMTranslator._clean_translation method."""

    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        with patch("openai.OpenAI") as mock:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "翻譯結果"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock.return_value = mock_client

            yield mock

    def test_clean_removes_preamble_heres_translation(self, mock_openai) -> None:
        """Test removing 'Here is the translation:' preamble."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        result = translator._clean_translation("Here is the translation: 你好", "Hello")
        assert result == "你好"

        result = translator._clean_translation("Here's the translation: 你好", "Hello")
        assert result == "你好"

    def test_clean_removes_translation_prefix(self, mock_openai) -> None:
        """Test removing 'Translation:' prefix."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        result = translator._clean_translation("Translation: 你好", "Hello")
        assert result == "你好"

        result = translator._clean_translation("Translated text: 你好世界", "Hello world")
        assert result == "你好世界"

    def test_clean_removes_in_english_prefix(self, mock_openai) -> None:
        """Test removing 'In English:' prefix."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        result = translator._clean_translation("In English: Hello", "你好")
        assert result == "Hello"

    def test_clean_removes_trailing_notes_parentheses(self, mock_openai) -> None:
        """Test removing trailing notes in parentheses."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        result = translator._clean_translation("你好 (Note: I translated this literally)", "Hello")
        assert result == "你好"

        result = translator._clean_translation(
            "你好世界 (I translated this as a greeting)", "Hello world"
        )
        assert result == "你好世界"

        result = translator._clean_translation("你好 (Translator's note: formal version)", "Hello")
        assert result == "你好"

    def test_clean_removes_trailing_notes_newline(self, mock_openai) -> None:
        """Test removing trailing notes after newline."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        result = translator._clean_translation("你好\n\nNote: This is a simple greeting", "Hello")
        assert result == "你好"

        result = translator._clean_translation("你好\n\nP.S. This is formal", "Hello")
        assert result == "你好"

    def test_clean_preserves_original_leading_newline(self, mock_openai) -> None:
        """Test preserving leading newline from original."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        result = translator._clean_translation("你好", "\nHello")
        assert result == "\n你好"

    def test_clean_preserves_original_trailing_newline(self, mock_openai) -> None:
        """Test preserving trailing newline from original."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        result = translator._clean_translation("你好", "Hello\n")
        assert result == "你好\n"

    def test_clean_preserves_both_newlines(self, mock_openai) -> None:
        """Test preserving both leading and trailing newlines."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        result = translator._clean_translation("你好", "\nHello\n")
        assert result == "\n你好\n"


class TestLLMTranslatorBatchErrorHandling:
    """Tests for LLMTranslator.translate_batch error handling."""

    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        with patch("openai.OpenAI") as mock:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "翻譯結果"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock.return_value = mock_client

            yield mock

    def test_translate_batch_with_error(self, mock_openai) -> None:
        """Test translate_batch handles errors gracefully."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        # Make the second call fail
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [
            mock_client.chat.completions.create.return_value,  # First succeeds
            Exception("API Error"),  # Second fails
            mock_client.chat.completions.create.return_value,  # Third succeeds
        ]

        results = translator.translate_batch(["Hello", "World", "Test"], "zh", "en")

        assert len(results) == 3
        # First and third should be translated
        assert results[0].translated_text == "翻譯結果"
        # Second should be original (due to error)
        assert results[1].translated_text == "World"
        assert results[1].source_language == "en"
        assert results[2].translated_text == "翻譯結果"

    def test_translate_batch_all_errors(self, mock_openai) -> None:
        """Test translate_batch when all translations fail."""
        from pretok.config import LLMTranslatorConfig
        from pretok.translation.llm import LLMTranslator

        config = LLMTranslatorConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="test-key",
        )
        translator = LLMTranslator(config)

        # Make all calls fail
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")

        results = translator.translate_batch(["Hello", "World"], "zh")

        assert len(results) == 2
        # All should be original text
        assert results[0].translated_text == "Hello"
        assert results[0].source_language == "unknown"
        assert results[1].translated_text == "World"
