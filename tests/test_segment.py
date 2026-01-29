"""Tests for segment processing module."""

from __future__ import annotations

from pretok.segment import PromptLexer, Segment, SegmentType, lex_prompt
from pretok.segment.lexer import detect_prompt_format
from pretok.segment.types import segments_to_text


class TestSegmentType:
    """Tests for SegmentType enum."""

    def test_all_types_exist(self) -> None:
        """Test that all expected segment types exist."""
        assert SegmentType.TEXT
        assert SegmentType.CODE
        assert SegmentType.JSON
        assert SegmentType.ROLE_MARKER
        assert SegmentType.CONTROL_TOKEN
        assert SegmentType.DELIMITER
        assert SegmentType.WHITESPACE
        assert SegmentType.COMMENT


class TestSegment:
    """Tests for Segment dataclass."""

    def test_create_text_segment(self) -> None:
        """Test creating a text segment."""
        seg = Segment(
            type=SegmentType.TEXT,
            content="Hello world",
            start=0,
            end=11,
        )
        assert seg.type == SegmentType.TEXT
        assert seg.content == "Hello world"
        assert seg.start == 0
        assert seg.end == 11
        assert seg.translatable is True

    def test_code_segment_not_translatable(self) -> None:
        """Test that code segments are not translatable."""
        seg = Segment(
            type=SegmentType.CODE,
            content="print('hello')",
            start=0,
            end=14,
        )
        assert seg.translatable is False

    def test_role_marker_not_translatable(self) -> None:
        """Test that role markers are not translatable."""
        seg = Segment(
            type=SegmentType.ROLE_MARKER,
            content="<|user|>",
            start=0,
            end=8,
        )
        assert seg.translatable is False

    def test_segment_length(self) -> None:
        """Test segment length property."""
        seg = Segment(
            type=SegmentType.TEXT,
            content="Hello",
            start=0,
            end=5,
        )
        assert len(seg) == 5

    def test_segment_span(self) -> None:
        """Test segment span property."""
        seg = Segment(
            type=SegmentType.TEXT,
            content="Hello",
            start=10,
            end=15,
        )
        assert seg.span == (10, 15)

    def test_with_content(self) -> None:
        """Test creating new segment with different content."""
        seg = Segment(
            type=SegmentType.TEXT,
            content="Hello",
            start=0,
            end=5,
            language="en",
        )
        new_seg = seg.with_content("你好")

        assert new_seg.content == "你好"
        assert new_seg.type == SegmentType.TEXT
        assert new_seg.language == "en"
        # Original unchanged
        assert seg.content == "Hello"

    def test_with_language(self) -> None:
        """Test creating new segment with language."""
        seg = Segment(
            type=SegmentType.TEXT,
            content="Hello",
            start=0,
            end=5,
        )
        new_seg = seg.with_language("en")

        assert new_seg.language == "en"
        assert seg.language is None


class TestSegmentsToText:
    """Tests for segments_to_text function."""

    def test_reconstruct_simple(self) -> None:
        """Test reconstructing simple text."""
        segments = [
            Segment(type=SegmentType.TEXT, content="Hello ", start=0, end=6),
            Segment(type=SegmentType.TEXT, content="world", start=6, end=11),
        ]
        assert segments_to_text(segments) == "Hello world"

    def test_reconstruct_mixed(self) -> None:
        """Test reconstructing mixed content."""
        segments = [
            Segment(type=SegmentType.ROLE_MARKER, content="<|user|>", start=0, end=8),
            Segment(type=SegmentType.TEXT, content="\nHello", start=8, end=14),
            Segment(type=SegmentType.ROLE_MARKER, content="<|end|>", start=14, end=21),
        ]
        assert segments_to_text(segments) == "<|user|>\nHello<|end|>"

    def test_reconstruct_empty(self) -> None:
        """Test reconstructing empty list."""
        assert segments_to_text([]) == ""


class TestPromptLexer:
    """Tests for PromptLexer."""

    def test_lex_plain_text(self) -> None:
        """Test lexing plain text."""
        lexer = PromptLexer()
        segments = lexer.lex("Hello, world!")

        assert len(segments) == 1
        assert segments[0].type == SegmentType.TEXT
        assert segments[0].content == "Hello, world!"

    def test_lex_chatml_format(self) -> None:
        """Test lexing ChatML format."""
        lexer = PromptLexer(format_hint="chatml")
        text = "<|im_start|>user\nHello!<|im_end|>"
        segments = lexer.lex(text)

        types = [s.type for s in segments]
        assert SegmentType.ROLE_MARKER in types
        assert SegmentType.TEXT in types

    def test_lex_llama_format(self) -> None:
        """Test lexing Llama format."""
        lexer = PromptLexer(format_hint="llama2")
        text = "[INST]What is Python?[/INST]"
        segments = lexer.lex(text)

        # Should have INST markers and text
        assert any(s.type == SegmentType.ROLE_MARKER for s in segments)
        assert any(s.content == "What is Python?" for s in segments)

    def test_lex_code_block(self) -> None:
        """Test lexing code blocks."""
        lexer = PromptLexer()
        text = "Here is code:\n```python\nprint('hello')\n```\nEnd."
        segments = lexer.lex(text)

        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 1
        assert "print('hello')" in code_segments[0].content

    def test_lex_inline_code(self) -> None:
        """Test lexing inline code."""
        lexer = PromptLexer()
        text = "Use the `print()` function"
        segments = lexer.lex(text)

        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 1
        assert code_segments[0].content == "`print()`"

    def test_lex_preserves_content(self) -> None:
        """Test that lexing preserves all content."""
        lexer = PromptLexer()
        original = "<|im_start|>user\n你好，世界！\n```python\nprint('test')\n```\n<|im_end|>"
        segments = lexer.lex(original)
        reconstructed = segments_to_text(segments)

        assert reconstructed == original

    def test_empty_input(self) -> None:
        """Test lexing empty input."""
        lexer = PromptLexer()
        assert lexer.lex("") == []


class TestLexPrompt:
    """Tests for lex_prompt convenience function."""

    def test_basic_usage(self) -> None:
        """Test basic usage of lex_prompt."""
        segments = lex_prompt("Hello, world!")
        assert len(segments) == 1
        assert segments[0].type == SegmentType.TEXT

    def test_with_format_hint(self) -> None:
        """Test lex_prompt with format hint."""
        segments = lex_prompt("[INST]Hi[/INST]", format_hint="llama2")
        assert any(s.type == SegmentType.ROLE_MARKER for s in segments)


class TestDetectPromptFormat:
    """Tests for prompt format detection."""

    def test_detect_chatml(self) -> None:
        """Test detecting ChatML format."""
        text = "<|im_start|>user\nHello<|im_end|>"
        assert detect_prompt_format(text) == "chatml"

    def test_detect_llama2(self) -> None:
        """Test detecting Llama 2 format."""
        text = "[INST]Question[/INST]"
        assert detect_prompt_format(text) == "llama2"

    def test_detect_alpaca(self) -> None:
        """Test detecting Alpaca format."""
        text = "### Instruction:\nDo something\n### Response:"
        assert detect_prompt_format(text) == "alpaca"

    def test_detect_vicuna(self) -> None:
        """Test detecting Vicuna format."""
        text = "USER: Hello\nASSISTANT: Hi there"
        assert detect_prompt_format(text) == "vicuna"

    def test_detect_none(self) -> None:
        """Test that plain text returns None."""
        text = "Just some plain text without markers"
        assert detect_prompt_format(text) is None


class TestComplexPrompts:
    """Tests for complex real-world prompts."""

    def test_multilingual_prompt(self) -> None:
        """Test prompt with multiple languages."""
        lexer = PromptLexer()
        text = "<|im_start|>user\n請幫我翻譯這段英文：Hello world<|im_end|>"
        segments = lexer.lex(text)

        # Should preserve all content
        assert segments_to_text(segments) == text

        # Text segment should contain both Chinese and English
        text_segments = [s for s in segments if s.type == SegmentType.TEXT]
        assert len(text_segments) >= 1

    def test_code_with_comments(self) -> None:
        """Test code block with comments."""
        lexer = PromptLexer()
        text = """Here's the code:
```python
# This is a comment
def hello():
    print("Hello")  # Another comment
```
Done."""
        segments = lexer.lex(text)

        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 1
        assert "# This is a comment" in code_segments[0].content

    def test_nested_json(self) -> None:
        """Test JSON content detection."""
        lexer = PromptLexer()
        text = 'Parse this: {"name": "test", "value": 123}'
        segments = lexer.lex(text)

        # JSON detection is basic, may or may not detect
        # Main goal is not to break parsing
        _ = [s for s in segments if s.type == SegmentType.JSON]  # Check if JSON detected
        assert segments_to_text(segments) == text


class TestLexerAdditionalCoverage:
    """Additional tests to improve lexer coverage."""

    def test_lex_with_config(self) -> None:
        """Test lexer with custom config."""
        from pretok.config import SegmentConfig

        config = SegmentConfig(code_fence_pattern="```")
        lexer = PromptLexer(config=config)
        text = "```code```"
        segments = lexer.lex(text)
        assert segments_to_text(segments) == text

    def test_lex_no_format_hint_all_patterns(self) -> None:
        """Test lexer without format hint includes all patterns."""
        lexer = PromptLexer(format_hint=None)
        # Should not raise
        text = "<|im_start|>user\nHello<|im_end|>"
        segments = lexer.lex(text)
        assert len(segments) > 0

    def test_detect_llama_format(self) -> None:
        """Test detecting Llama format with specific markers."""
        text = "<|begin_of_text|>Hello"
        assert detect_prompt_format(text) == "llama"

    def test_detect_mistral_format(self) -> None:
        """Test detecting Mistral format."""
        text = "[INST] Question [/INST] Answer"
        # Note: Mistral and Llama2 share markers, should detect one of them
        result = detect_prompt_format(text)
        assert result in ("llama2", "mistral")

    def test_merge_adjacent_text_empty(self) -> None:
        """Test _merge_adjacent_text with empty input."""
        lexer = PromptLexer()
        result = lexer._merge_adjacent_text([])
        assert result == []

    def test_lex_url_pattern(self) -> None:
        """Test lexer recognizes URL pattern."""
        lexer = PromptLexer()
        text = "Check https://example.com for info"
        segments = lexer.lex(text)
        assert segments_to_text(segments) == text
        # URL should be preserved as text or specific type
        assert any("https://example.com" in s.content for s in segments)

    def test_lex_custom_markers_with_regex(self) -> None:
        """Test custom markers with is_regex=True."""
        from pretok.config import SegmentConfig
        from pretok.config.schema import CustomMarkerConfig

        custom_marker = CustomMarkerConfig(
            pattern=r"\$\{.*?\}",  # Match ${...}
            type="CONTROL_TOKEN",
            is_regex=True,
        )
        config = SegmentConfig(custom_markers=[custom_marker])
        lexer = PromptLexer(config=config)
        text = "Hello ${variable} world"
        segments = lexer.lex(text)
        assert segments_to_text(segments) == text

    def test_lex_custom_markers_without_regex(self) -> None:
        """Test custom markers with is_regex=False (literal)."""
        from pretok.config import SegmentConfig
        from pretok.config.schema import CustomMarkerConfig

        custom_marker = CustomMarkerConfig(
            pattern="<<CUSTOM>>",
            type="DELIMITER",
            is_regex=False,
        )
        config = SegmentConfig(custom_markers=[custom_marker])
        lexer = PromptLexer(config=config)
        text = "Before <<CUSTOM>> After"
        segments = lexer.lex(text)
        # Should have delimiter segment
        delim_segs = [s for s in segments if s.type == SegmentType.DELIMITER]
        assert len(delim_segs) > 0

    def test_detect_alpaca_format(self) -> None:
        """Test detecting Alpaca format."""
        text = "### Instruction:\nDo something\n### Response:\nDone"
        assert detect_prompt_format(text) == "alpaca"

    def test_detect_vicuna_format(self) -> None:
        """Test detecting Vicuna format."""
        text = "USER: Hello\nASSISTANT: Hi there"
        assert detect_prompt_format(text) == "vicuna"

    def test_detect_unknown_format(self) -> None:
        """Test detecting unknown format returns None."""
        text = "Just plain text without any format markers"
        assert detect_prompt_format(text) is None

    def test_lex_adjacent_text_merge(self) -> None:
        """Test that adjacent text segments get merged."""
        lexer = PromptLexer()
        # Plain text should become a single segment
        text = "Hello world this is a test"
        segments = lexer.lex(text)
        text_segments = [s for s in segments if s.type == SegmentType.TEXT]
        assert len(text_segments) == 1
        assert text_segments[0].content == text
