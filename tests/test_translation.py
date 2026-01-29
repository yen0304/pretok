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
