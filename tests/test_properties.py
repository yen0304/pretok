"""Property-based tests using Hypothesis.

These tests verify invariants that should hold for any input.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pretok.detection import DetectionResult
from pretok.pipeline.cache import MemoryCache, make_cache_key
from pretok.segment import Segment, SegmentType
from pretok.segment.lexer import PromptLexer
from pretok.segment.types import segments_to_text
from pretok.translation import TranslationResult

# =============================================================================
# Detection Result Properties
# =============================================================================


class TestDetectionResultProperties:
    """Property-based tests for DetectionResult."""

    @given(
        language=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        detector=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    def test_valid_detection_result_creation(
        self,
        language: str,
        confidence: float,
        detector: str,
    ) -> None:
        """Valid inputs should always create a valid DetectionResult."""
        result = DetectionResult(
            language=language,
            confidence=confidence,
            detector=detector,
        )
        assert result.language == language
        assert result.confidence == confidence
        assert result.detector == detector

    @given(
        language=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
        confidence=st.floats(
            min_value=1.01,
            max_value=100.0,
            allow_nan=False,
        ),
        detector=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    def test_invalid_high_confidence_rejected(
        self,
        language: str,
        confidence: float,
        detector: str,
    ) -> None:
        """Confidence > 1.0 should be rejected."""
        import pytest

        with pytest.raises(ValueError):
            DetectionResult(
                language=language,
                confidence=confidence,
                detector=detector,
            )

    @given(
        language=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
        confidence=st.floats(
            min_value=-100.0,
            max_value=-0.01,
            allow_nan=False,
        ),
        detector=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    def test_invalid_negative_confidence_rejected(
        self,
        language: str,
        confidence: float,
        detector: str,
    ) -> None:
        """Negative confidence should be rejected."""
        import pytest

        with pytest.raises(ValueError):
            DetectionResult(
                language=language,
                confidence=confidence,
                detector=detector,
            )


# =============================================================================
# Translation Result Properties
# =============================================================================


class TestTranslationResultProperties:
    """Property-based tests for TranslationResult."""

    @given(
        source_text=st.text(max_size=1000),
        translated_text=st.text(max_size=1000),
        source_lang=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
        target_lang=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
        translator=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    def test_translation_result_creation(
        self,
        source_text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
        translator: str,
    ) -> None:
        """Any valid strings should create a valid TranslationResult."""
        result = TranslationResult(
            source_text=source_text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            translator=translator,
        )
        assert result.source_text == source_text
        assert result.translated_text == translated_text
        assert result.source_language == source_lang
        assert result.target_language == target_lang

    @given(
        text=st.text(max_size=500),
        lang=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
        translator=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    def test_was_translated_false_when_same(
        self,
        text: str,
        lang: str,
        translator: str,
    ) -> None:
        """was_translated should be False when source equals translated."""
        result = TranslationResult(
            source_text=text,
            translated_text=text,
            source_language=lang,
            target_language=lang,
            translator=translator,
        )
        assert not result.was_translated

    @given(
        source_text=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        translated_text=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        source_lang=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
        target_lang=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
        translator=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    def test_was_translated_true_when_different(
        self,
        source_text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
        translator: str,
    ) -> None:
        """was_translated should be True when texts differ."""
        assume(source_text != translated_text)
        result = TranslationResult(
            source_text=source_text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            translator=translator,
        )
        assert result.was_translated


# =============================================================================
# Segment Properties
# =============================================================================


class TestSegmentProperties:
    """Property-based tests for Segment."""

    @given(
        content=st.text(max_size=1000),
        start=st.integers(min_value=0, max_value=10000),
    )
    def test_segment_length_equals_content_length(
        self,
        content: str,
        start: int,
    ) -> None:
        """Segment length should equal content length."""
        segment = Segment(
            content=content,
            type=SegmentType.TEXT,
            start=start,
            end=start + len(content),
        )
        assert len(segment) == len(content)

    @given(
        content=st.text(max_size=1000),
        start=st.integers(min_value=0, max_value=10000),
    )
    def test_segment_span_correct(
        self,
        content: str,
        start: int,
    ) -> None:
        """Segment span should be (start, start + len(content))."""
        end = start + len(content)
        segment = Segment(
            content=content,
            type=SegmentType.TEXT,
            start=start,
            end=end,
        )
        assert segment.span == (start, end)

    @given(
        old_content=st.text(max_size=500),
        new_content=st.text(max_size=500),
        start=st.integers(min_value=0, max_value=10000),
    )
    def test_with_content_preserves_other_fields(
        self,
        old_content: str,
        new_content: str,
        start: int,
    ) -> None:
        """with_content should preserve type and metadata."""
        original = Segment(
            content=old_content,
            type=SegmentType.CODE,
            start=start,
            end=start + len(old_content),
            language="python",
        )
        updated = original.with_content(new_content)

        assert updated.content == new_content
        assert updated.type == original.type
        assert updated.language == original.language

    @given(
        content=st.text(max_size=500),
        start=st.integers(min_value=0, max_value=10000),
        language=st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
    )
    def test_with_language_preserves_other_fields(
        self,
        content: str,
        start: int,
        language: str,
    ) -> None:
        """with_language should only change the language field."""
        end = start + len(content)
        original = Segment(
            content=content,
            type=SegmentType.TEXT,
            start=start,
            end=end,
        )
        updated = original.with_language(language)

        assert updated.content == original.content
        assert updated.type == original.type
        assert updated.start == original.start
        assert updated.end == original.end
        assert updated.language == language


# =============================================================================
# Lexer Properties
# =============================================================================


class TestLexerProperties:
    """Property-based tests for PromptLexer."""

    @given(text=st.text(max_size=2000))
    @settings(max_examples=100)
    def test_segments_reconstruct_original(self, text: str) -> None:
        """Segments should always reconstruct the original text."""
        lexer = PromptLexer()
        segments = lexer.lex(text)
        reconstructed = segments_to_text(segments)

        assert reconstructed == text

    @given(text=st.text(max_size=2000))
    @settings(max_examples=100)
    def test_segments_are_contiguous(self, text: str) -> None:
        """Segments should cover the entire text without gaps or overlaps."""
        lexer = PromptLexer()
        segments = lexer.lex(text)

        if not segments:
            assert text == ""
            return

        # First segment starts at 0
        assert segments[0].start == 0

        # Each segment's end is next segment's start
        for i in range(len(segments) - 1):
            current_end = segments[i].start + len(segments[i].content)
            next_start = segments[i + 1].start
            assert current_end == next_start

        # Last segment ends at text length
        last_segment = segments[-1]
        assert last_segment.start + len(last_segment.content) == len(text)

    @given(text=st.text(max_size=2000))
    @settings(max_examples=100)
    def test_no_empty_segments(self, text: str) -> None:
        """Lexer should not produce empty segments."""
        lexer = PromptLexer()
        segments = lexer.lex(text)

        for segment in segments:
            # Each segment should have content
            assert len(segment.content) > 0


# =============================================================================
# Cache Properties
# =============================================================================


class TestCacheProperties:
    """Property-based tests for caching."""

    @given(
        text=st.text(max_size=500),
        source_lang=st.one_of(st.none(), st.text(min_size=1, max_size=5)),
        target_lang=st.text(min_size=1, max_size=5).filter(lambda x: x.strip()),
        translator=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    def test_cache_key_deterministic(
        self,
        text: str,
        source_lang: str | None,
        target_lang: str,
        translator: str,
    ) -> None:
        """Same inputs should always produce the same cache key."""
        key1 = make_cache_key(text, source_lang, target_lang, translator)
        key2 = make_cache_key(text, source_lang, target_lang, translator)

        assert key1 == key2

    @given(
        text1=st.text(min_size=1, max_size=500),
        text2=st.text(min_size=1, max_size=500),
        target_lang=st.text(min_size=1, max_size=5).filter(lambda x: x.strip()),
        translator=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
    )
    def test_different_texts_different_keys(
        self,
        text1: str,
        text2: str,
        target_lang: str,
        translator: str,
    ) -> None:
        """Different texts should produce different cache keys."""
        assume(text1 != text2)
        key1 = make_cache_key(text1, None, target_lang, translator)
        key2 = make_cache_key(text2, None, target_lang, translator)

        assert key1 != key2

    @given(
        key=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        value=st.text(max_size=500),
    )
    def test_cache_set_get_roundtrip(self, key: str, value: str) -> None:
        """Value stored in cache should be retrievable."""
        cache = MemoryCache(max_size=100)
        cache.set(key, value)
        retrieved = cache.get(key)

        assert retrieved == value

    @given(
        keys=st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            min_size=1,
            max_size=20,
            unique=True,
        ),
    )
    def test_cache_clear_removes_all(self, keys: list[str]) -> None:
        """Clear should remove all entries."""
        cache = MemoryCache(max_size=100)

        for key in keys:
            cache.set(key, f"value_{key}")

        cache.clear()

        for key in keys:
            assert cache.get(key) is None


# =============================================================================
# Code Block Properties
# =============================================================================


class TestCodeBlockProperties:
    """Property-based tests for code block handling."""

    @given(
        code=st.text(min_size=1, max_size=500).filter(lambda x: "```" not in x),
        language=st.sampled_from(["python", "javascript", "rust", "go", ""]),
    )
    def test_code_blocks_preserved(self, code: str, language: str) -> None:
        """Code blocks should be preserved exactly."""
        lexer = PromptLexer()

        text = f"```{language}\n{code}\n```"
        segments = lexer.lex(text)

        # Code block content should be preserved
        reconstructed = segments_to_text(segments)
        assert reconstructed == text

        # Should have at least one code segment
        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) >= 1


# =============================================================================
# Prompt Format Properties
# =============================================================================


class TestPromptFormatProperties:
    """Property-based tests for prompt format detection."""

    @given(role=st.sampled_from(["system", "user", "assistant"]))
    def test_chatml_markers_preserved(self, role: str) -> None:
        """ChatML markers should be preserved."""
        lexer = PromptLexer()

        text = f"<|im_start|>{role}\nContent<|im_end|>"
        segments = lexer.lex(text)

        reconstructed = segments_to_text(segments)
        assert reconstructed == text
        assert f"<|im_start|>{role}" in reconstructed
        assert "<|im_end|>" in reconstructed

    @given(
        content=st.text(min_size=1, max_size=200).filter(
            lambda x: "[INST]" not in x and "[/INST]" not in x
        ),
    )
    def test_llama_format_preserved(self, content: str) -> None:
        """Llama format markers should be preserved."""
        lexer = PromptLexer()

        text = f"[INST] {content} [/INST]"
        segments = lexer.lex(text)

        reconstructed = segments_to_text(segments)
        assert reconstructed == text
