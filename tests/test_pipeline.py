"""Tests for pipeline module."""

from __future__ import annotations

import pytest

from pretok.detection import DetectionResult
from pretok.detection.base import BaseDetector
from pretok.pipeline.cache import MemoryCache, make_cache_key
from pretok.pipeline.core import PipelineResult, Pretok, create_pretok
from pretok.translation import TranslationResult
from pretok.translation.base import BaseTranslator


class TestMemoryCache:
    """Tests for MemoryCache."""

    def test_get_set(self) -> None:
        """Test basic get and set."""
        cache = MemoryCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_get_missing(self) -> None:
        """Test getting missing key."""
        cache = MemoryCache()
        assert cache.get("missing") is None

    def test_delete(self) -> None:
        """Test deleting key."""
        cache = MemoryCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None

    def test_delete_missing(self) -> None:
        """Test deleting missing key."""
        cache = MemoryCache()
        assert cache.delete("missing") is False

    def test_clear(self) -> None:
        """Test clearing cache."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size() == 0

    def test_max_size(self) -> None:
        """Test max size eviction."""
        cache = MemoryCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict 'a'

        assert cache.size() == 3
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_lru_eviction(self) -> None:
        """Test LRU eviction order."""
        cache = MemoryCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access 'a' to make it recently used
        cache.get("a")

        cache.set("d", 4)  # Should evict 'b' (least recently used)

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_stats(self) -> None:
        """Test cache statistics."""
        cache = MemoryCache(max_size=10)
        cache.set("a", 1)
        cache.get("a")  # Hit
        cache.get("b")  # Miss

        stats = cache.stats
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestMakeCacheKey:
    """Tests for cache key generation."""

    def test_deterministic(self) -> None:
        """Test that same inputs give same key."""
        key1 = make_cache_key("hello", "en", "zh", "test")
        key2 = make_cache_key("hello", "en", "zh", "test")
        assert key1 == key2

    def test_different_inputs(self) -> None:
        """Test that different inputs give different keys."""
        key1 = make_cache_key("hello", "en", "zh", "test")
        key2 = make_cache_key("world", "en", "zh", "test")
        assert key1 != key2

    def test_auto_source_language(self) -> None:
        """Test with None source language."""
        key = make_cache_key("hello", None, "zh", "test")
        assert len(key) == 32


class MockDetector(BaseDetector):
    """Mock detector for testing."""

    @property
    def name(self) -> str:
        return "mock"

    def detect(self, text: str) -> DetectionResult:
        # Simple mock: assume Chinese if contains Chinese chars
        if any("\u4e00" <= c <= "\u9fff" for c in text):
            return DetectionResult(language="zh", confidence=0.95, detector="mock")
        return DetectionResult(language="en", confidence=0.95, detector="mock")


class MockTranslator(BaseTranslator):
    """Mock translator for testing."""

    @property
    def name(self) -> str:
        return "mock"

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> TranslationResult:
        # Simple mock: return bracketed text as "translation"
        translated = f"[translated:{target_language}]{text}"
        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_language=source_language or "auto",
            target_language=target_language,
            translator=self.name,
        )


class TestPretok:
    """Tests for Pretok class."""

    @pytest.fixture
    def pretok(self) -> Pretok:
        """Create Pretok instance with mock components."""
        return Pretok(
            target_language="en",
            detector=MockDetector(),
            translator=MockTranslator(),
        )

    def test_create_default(self) -> None:
        """Test creating Pretok with defaults."""
        pretok = Pretok()
        assert pretok.target_language == "en"

    def test_create_with_target_language(self) -> None:
        """Test creating Pretok with custom target language."""
        pretok = Pretok(target_language="zh")
        assert pretok.target_language == "zh"

    def test_process_empty_text(self, pretok: Pretok) -> None:
        """Test processing empty text."""
        result = pretok.process("")
        assert result.original_text == ""
        assert result.processed_text == ""
        assert result.was_modified is False

    def test_process_same_language(self, pretok: Pretok) -> None:
        """Test processing text already in target language."""
        result = pretok.process("Hello, world!")

        assert result.original_text == "Hello, world!"
        # English to English, no translation needed
        assert result.processed_text == "Hello, world!"
        assert len(result.translations) == 0

    def test_process_different_language(self, pretok: Pretok) -> None:
        """Test processing text in different language."""
        result = pretok.process("你好世界")

        assert result.original_text == "你好世界"
        # Chinese should be translated to English
        assert "[translated:en]" in result.processed_text
        assert len(result.translations) > 0

    def test_process_detect_only(self, pretok: Pretok) -> None:
        """Test processing with detect_only flag."""
        result = pretok.process("你好世界", detect_only=True)

        assert result.processed_text == "你好世界"  # Unchanged
        assert len(result.detections) > 0
        assert len(result.translations) == 0

    def test_process_preserves_structure(self, pretok: Pretok) -> None:
        """Test that prompt structure is preserved."""
        text = "<|im_start|>user\n你好<|im_end|>"
        result = pretok.process(text)

        # Should still have markers
        assert "<|im_start|>" in result.processed_text
        assert "<|im_end|>" in result.processed_text

    def test_detect(self, pretok: Pretok) -> None:
        """Test detect convenience method."""
        result = pretok.detect("Hello, world!")
        assert result.language == "en"
        assert result.confidence > 0.5

    def test_translate(self, pretok: Pretok) -> None:
        """Test translate convenience method."""
        result = pretok.translate("Hello", "zh")
        assert "[translated:zh]" in result.translated_text

    def test_translate_no_translator(self) -> None:
        """Test translate raises error without translator."""
        pretok = Pretok(detector=MockDetector())
        with pytest.raises(ValueError, match="No translator"):
            pretok.translate("Hello", "zh")

    def test_cache_hit(self, pretok: Pretok) -> None:
        """Test that cache is used."""
        # First call - cache miss
        result1 = pretok.process("你好")
        assert result1.from_cache is False

        # Second call - cache hit
        result2 = pretok.process("你好")
        assert result2.from_cache is True
        assert result2.processed_text == result1.processed_text

    def test_set_translator(self) -> None:
        """Test setting translator after creation."""
        pretok = Pretok(detector=MockDetector())
        assert pretok.translator is None

        translator = MockTranslator()
        pretok.set_translator(translator)
        assert pretok.translator is translator


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_was_modified_true(self) -> None:
        """Test was_modified when text changed."""
        result = PipelineResult(
            original_text="Hello",
            processed_text="你好",
        )
        assert result.was_modified is True

    def test_was_modified_false(self) -> None:
        """Test was_modified when text unchanged."""
        result = PipelineResult(
            original_text="Hello",
            processed_text="Hello",
        )
        assert result.was_modified is False


class TestCreatePretok:
    """Tests for create_pretok factory function."""

    def test_create_with_defaults(self) -> None:
        """Test creating with defaults."""
        pretok = create_pretok()
        assert pretok.target_language == "en"

    def test_create_with_model_id(self) -> None:
        """Test creating with model ID."""
        pretok = create_pretok(model_id="gpt-4-turbo")
        assert pretok.target_language == "en"

    def test_create_with_qwen(self) -> None:
        """Test creating with Qwen model (Chinese-optimized)."""
        pretok = create_pretok(model_id="qwen-2.5-72b")
        # Qwen's primary language is Chinese
        assert pretok.target_language == "zh"
