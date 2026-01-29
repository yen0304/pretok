"""Integration tests for pretok pipeline.

These tests verify that all components work together correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pretok import Pretok, create_pretok
from pretok.detection import DetectionResult
from pretok.detection.base import BaseDetector
from pretok.pipeline.cache import MemoryCache
from pretok.translation import TranslationResult
from pretok.translation.base import BaseTranslator


class MockDetector(BaseDetector):
    """Mock detector for integration tests."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        """Initialize with predefined responses."""
        self._responses = responses or {}
        self._default = "en"

    @property
    def name(self) -> str:
        return "mock_detector"

    def detect(self, text: str) -> DetectionResult:
        """Return predefined or default detection."""
        language = self._responses.get(text, self._default)
        return DetectionResult(
            language=language,
            confidence=0.95,
            detector=self.name,
        )

    def detect_batch(self, texts: list[str]) -> list[DetectionResult]:
        """Detect language for multiple texts."""
        return [self.detect(text) for text in texts]


class MockTranslator(BaseTranslator):
    """Mock translator for integration tests."""

    def __init__(
        self,
        responses: dict[tuple[str, str, str], str] | None = None,
        name: str = "mock_translator",
    ) -> None:
        """Initialize with predefined responses."""
        self._responses = responses or {}
        self._name = name
        self._supported_languages: list[str] = []
        self.translate_called: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_languages(self) -> list[str]:
        return self._supported_languages

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> TranslationResult:
        """Return predefined or echo translation."""
        self.translate_called.append(
            {
                "text": text,
                "target": target_language,
                "source": source_language,
            }
        )
        key = (text, target_language, source_language or "auto")
        translated = self._responses.get(key, f"[{target_language}]{text}")
        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_language=source_language or "auto",
            target_language=target_language,
            translator=self.name,
        )


class TestEndToEndPipeline:
    """End-to-end integration tests for the pretok pipeline."""

    def test_full_pipeline_with_translation(self) -> None:
        """Test complete flow: detection -> translation -> output."""
        detector = MockDetector({"Hello world": "en", "Bonjour le monde": "fr"})
        translator = MockTranslator(
            {
                ("Bonjour le monde", "en", "fr"): "Hello world",
            }
        )

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        result = pretok.process("Bonjour le monde")

        assert result.processed_text == "Hello world"
        assert result.was_modified
        # Check detections contain French
        assert any(d.language == "fr" for d in result.detections)

    def test_pipeline_no_translation_needed(self) -> None:
        """Test that text in target language is not translated."""
        detector = MockDetector({"Hello world": "en"})
        translator = MockTranslator()

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        result = pretok.process("Hello world")

        assert result.processed_text == "Hello world"
        assert not result.was_modified
        assert len(translator.translate_called) == 0

    def test_pipeline_multi_segment_prompt(self) -> None:
        """Test processing prompt with multiple segments."""
        detector = MockDetector(
            {
                "system": "en",
                "assistant": "en",
                "user": "en",
                "You are a helpful assistant.": "en",
                "Translate this text.": "en",
            }
        )
        translator = MockTranslator()

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Translate this text.<|im_end|>"""

        result = pretok.process(prompt)

        # Structure should be preserved
        assert "<|im_start|>" in result.processed_text
        assert "<|im_end|>" in result.processed_text

    def test_pipeline_preserves_code_blocks(self) -> None:
        """Test that code blocks are not translated."""
        detector = MockDetector()
        translator = MockTranslator()

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        text = """Here is some code:
```python
def hello():
    print("Hello, world!")
```
The end."""

        result = pretok.process(text)

        # Code block should be preserved exactly
        assert 'print("Hello, world!")' in result.processed_text
        assert "```python" in result.processed_text

    def test_pipeline_with_caching(self) -> None:
        """Test that caching works correctly."""
        detector = MockDetector({"Test text": "en"})
        translator = MockTranslator(
            {
                ("Test text", "zh", "en"): "Test translation",
            }
        )
        cache = MemoryCache(max_size=100)

        pretok = Pretok(
            target_language="zh",
            detector=detector,
            translator=translator,
            cache=cache,
        )

        # First call
        result1 = pretok.process("Test text")
        translate_count1 = len(translator.translate_called)

        # Second call - should use cache
        result2 = pretok.process("Test text")
        translate_count2 = len(translator.translate_called)

        assert result1.processed_text == result2.processed_text
        # Should not call translate again due to cache
        # (detection might still be called)
        assert translate_count1 == translate_count2


class TestCreatePretokIntegration:
    """Integration tests for create_pretok factory."""

    def test_create_with_model_uses_primary_language(self) -> None:
        """Test that create_pretok uses model's primary language."""
        pretok = create_pretok(model_id="gpt-4")

        # GPT-4 has English as primary language
        assert pretok.target_language == "en"

    def test_create_with_explicit_target_overrides_model(self) -> None:
        """Test explicit target_language overrides model setting."""
        pretok = create_pretok(model_id="gpt-4", target_language="zh")

        assert pretok.target_language == "zh"

    def test_create_with_qwen_uses_chinese(self) -> None:
        """Test Qwen model defaults to Chinese."""
        pretok = create_pretok(model_id="qwen-7b")

        assert pretok.target_language == "zh"


class TestMultilingualScenarios:
    """Tests for multilingual text scenarios."""

    def test_mixed_language_text(self) -> None:
        """Test handling of text with mixed languages."""
        # This is a complex scenario - we test that it doesn't crash
        detector = MockDetector()
        translator = MockTranslator()

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        # Mixed English and French (simulated)
        text = "Hello world, bonjour le monde"
        result = pretok.process(text)

        # Should complete without error
        assert result.processed_text is not None

    def test_unicode_preservation(self) -> None:
        """Test that Unicode characters are preserved."""
        detector = MockDetector()
        translator = MockTranslator()

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        # Text with various Unicode characters
        text = "Hello 👋 world 🌍"
        result = pretok.process(text)

        assert "👋" in result.processed_text
        assert "🌍" in result.processed_text

    def test_empty_and_whitespace(self) -> None:
        """Test handling of empty and whitespace-only text."""
        detector = MockDetector()
        translator = MockTranslator()

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        # Empty text
        result_empty = pretok.process("")
        assert result_empty.processed_text == ""
        assert not result_empty.was_modified

        # Whitespace only
        result_ws = pretok.process("   \n\t  ")
        assert result_ws.processed_text == "   \n\t  "


class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_detector_error_propagates(self) -> None:
        """Test that detector errors are properly handled."""
        detector = MagicMock(spec=BaseDetector)
        detector.name = "failing_detector"
        detector.detect.side_effect = ValueError("Detection failed")

        pretok = Pretok(
            target_language="en",
            detector=detector,
        )

        with pytest.raises(ValueError, match="Detection failed"):
            pretok.detect("test")

    def test_translator_error_propagates(self) -> None:
        """Test that translator errors are properly handled."""
        detector = MockDetector({"test text": "fr"})
        translator = MagicMock(spec=BaseTranslator)
        translator.name = "failing_translator"
        translator.translate.side_effect = RuntimeError("Translation failed")

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        with pytest.raises(RuntimeError, match="Translation failed"):
            pretok.translate("test text", "en", "fr")

    def test_graceful_handling_without_translator(self) -> None:
        """Test that pipeline works without translator (detection only)."""
        detector = MockDetector({"test": "fr"})

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=None,
        )

        # Should not crash, just detect
        result = pretok.process("test")
        assert result.processed_text == "test"


class TestPromptFormatIntegration:
    """Tests for different prompt format handling."""

    def test_chatml_format_preserved(self) -> None:
        """Test ChatML format is preserved through pipeline."""
        detector = MockDetector()
        translator = MockTranslator()

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        prompt = """<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Help me.<|im_end|>"""

        result = pretok.process(prompt)

        assert "<|im_start|>system" in result.processed_text
        assert "<|im_end|>" in result.processed_text
        assert "<|im_start|>user" in result.processed_text

    def test_llama_format_preserved(self) -> None:
        """Test Llama format is preserved through pipeline."""
        detector = MockDetector()
        translator = MockTranslator()

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        prompt = """[INST] <<SYS>>
You are helpful.
<</SYS>>

Help me. [/INST]"""

        result = pretok.process(prompt)

        assert "[INST]" in result.processed_text
        assert "[/INST]" in result.processed_text
        assert "<<SYS>>" in result.processed_text

    def test_alpaca_format_preserved(self) -> None:
        """Test Alpaca format is preserved through pipeline."""
        detector = MockDetector()
        translator = MockTranslator()

        pretok = Pretok(
            target_language="en",
            detector=detector,
            translator=translator,
        )

        prompt = """### Instruction:
Tell me about AI.

### Response:
AI is fascinating."""

        result = pretok.process(prompt)

        assert "### Instruction:" in result.processed_text
        assert "### Response:" in result.processed_text
