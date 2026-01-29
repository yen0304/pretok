"""Prompt lexer for segment parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pretok.segment.types import Segment, SegmentType

if TYPE_CHECKING:
    from pretok.config import SegmentConfig


@dataclass
class PromptFormat:
    """Definition of a prompt format with its markers.

    Attributes:
        name: Format name (e.g., 'chatml', 'llama', 'alpaca')
        role_patterns: Regex patterns for role markers
        control_patterns: Regex patterns for control tokens
        delimiter_patterns: Regex patterns for delimiters
    """

    name: str
    role_patterns: list[str] = field(default_factory=list)
    control_patterns: list[str] = field(default_factory=list)
    delimiter_patterns: list[str] = field(default_factory=list)


# Predefined prompt formats
PROMPT_FORMATS: dict[str, PromptFormat] = {
    "chatml": PromptFormat(
        name="chatml",
        role_patterns=[
            r"<\|im_start\|>(?:system|user|assistant)",
            r"<\|im_end\|>",
        ],
        control_patterns=[
            r"<\|endoftext\|>",
        ],
    ),
    "llama": PromptFormat(
        name="llama",
        role_patterns=[
            r"<\|begin_of_text\|>",
            r"<\|end_of_text\|>",
            r"<\|start_header_id\|>(?:system|user|assistant)<\|end_header_id\|>",
            r"<\|eot_id\|>",
        ],
        control_patterns=[
            r"<s>",
            r"</s>",
        ],
    ),
    "llama2": PromptFormat(
        name="llama2",
        role_patterns=[
            r"\[INST\]",
            r"\[/INST\]",
            r"<<SYS>>",
            r"<</SYS>>",
        ],
        control_patterns=[
            r"<s>",
            r"</s>",
        ],
    ),
    "alpaca": PromptFormat(
        name="alpaca",
        role_patterns=[],
        control_patterns=[],
        delimiter_patterns=[
            r"### (?:Instruction|Input|Response):",
        ],
    ),
    "vicuna": PromptFormat(
        name="vicuna",
        role_patterns=[],
        control_patterns=[],
        delimiter_patterns=[
            r"USER:",
            r"ASSISTANT:",
            r"SYSTEM:",
        ],
    ),
    "mistral": PromptFormat(
        name="mistral",
        role_patterns=[
            r"\[INST\]",
            r"\[/INST\]",
        ],
        control_patterns=[
            r"<s>",
            r"</s>",
        ],
    ),
}


class PromptLexer:
    """Lexer for parsing prompts into segments.

    The lexer identifies different types of content in a prompt:
    - Role markers (e.g., <|user|>, [INST])
    - Control tokens (e.g., <s>, </s>)
    - Code blocks
    - JSON content
    - Regular text

    Example:
        >>> lexer = PromptLexer()
        >>> segments = lexer.lex("<|im_start|>user\\nHello!<|im_end|>")
        >>> [s.type for s in segments]
        [SegmentType.ROLE_MARKER, SegmentType.TEXT, SegmentType.ROLE_MARKER]
    """

    def __init__(
        self,
        config: SegmentConfig | None = None,
        format_hint: str | None = None,
    ) -> None:
        """Initialize the lexer.

        Args:
            config: Optional segment configuration
            format_hint: Hint for prompt format ('chatml', 'llama', etc.)
        """
        self._config = config
        self._format_hint = format_hint or (config.format_hint if config else None)
        self._patterns = self._build_patterns()

    def _build_patterns(self) -> list[tuple[re.Pattern[str], SegmentType]]:
        """Build compiled regex patterns for tokenization."""
        patterns: list[tuple[re.Pattern[str], SegmentType]] = []

        # Add format-specific patterns if format is specified
        if self._format_hint and self._format_hint in PROMPT_FORMATS:
            fmt = PROMPT_FORMATS[self._format_hint]
            for p in fmt.role_patterns:
                patterns.append((re.compile(p), SegmentType.ROLE_MARKER))
            for p in fmt.control_patterns:
                patterns.append((re.compile(p), SegmentType.CONTROL_TOKEN))
            for p in fmt.delimiter_patterns:
                patterns.append((re.compile(p), SegmentType.DELIMITER))
        else:
            # Add all known patterns if no format specified
            for fmt in PROMPT_FORMATS.values():
                for p in fmt.role_patterns:
                    patterns.append((re.compile(p), SegmentType.ROLE_MARKER))
                for p in fmt.control_patterns:
                    patterns.append((re.compile(p), SegmentType.CONTROL_TOKEN))
                for p in fmt.delimiter_patterns:
                    patterns.append((re.compile(p), SegmentType.DELIMITER))

        # Add custom markers from config
        if self._config and self._config.custom_markers:
            for marker in self._config.custom_markers:
                pattern = marker.pattern
                if not marker.is_regex:
                    pattern = re.escape(pattern)

                seg_type = SegmentType[marker.type]
                patterns.append((re.compile(pattern), seg_type))

        # Code block patterns (markdown-style)
        patterns.extend(
            [
                # Fenced code blocks with language
                (re.compile(r"```[\w]*\n[\s\S]*?```", re.MULTILINE), SegmentType.CODE),
                # Fenced code blocks without language
                (re.compile(r"```[\s\S]*?```", re.MULTILINE), SegmentType.CODE),
                # Inline code
                (re.compile(r"`[^`\n]+`"), SegmentType.CODE),
            ]
        )

        # JSON detection (simple heuristic)
        patterns.append((re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"), SegmentType.JSON))

        return patterns

    def lex(self, text: str) -> list[Segment]:
        """Parse text into segments.

        Args:
            text: Input text to parse

        Returns:
            List of Segment objects
        """
        if not text:
            return []

        segments: list[Segment] = []
        position = 0

        while position < len(text):
            # Try to match special patterns
            match_found = False

            for pattern, seg_type in self._patterns:
                match = pattern.match(text, position)
                if match:
                    # Add any preceding text
                    if match.start() > position:
                        segments.append(
                            Segment(
                                type=SegmentType.TEXT,
                                content=text[position : match.start()],
                                start=position,
                                end=match.start(),
                            )
                        )

                    # Add the matched segment
                    segments.append(
                        Segment(
                            type=seg_type,
                            content=match.group(),
                            start=match.start(),
                            end=match.end(),
                        )
                    )

                    position = match.end()
                    match_found = True
                    break

            if not match_found:
                # Find next potential match start
                next_match_start = len(text)
                for pattern, _ in self._patterns:
                    search = pattern.search(text, position)
                    if search and search.start() < next_match_start:
                        next_match_start = search.start()

                # Add text segment up to next match (or end)
                if next_match_start > position:
                    segments.append(
                        Segment(
                            type=SegmentType.TEXT,
                            content=text[position:next_match_start],
                            start=position,
                            end=next_match_start,
                        )
                    )
                    position = next_match_start

        return self._merge_adjacent_text(segments)

    def _merge_adjacent_text(self, segments: list[Segment]) -> list[Segment]:
        """Merge adjacent TEXT segments.

        Args:
            segments: List of segments

        Returns:
            Merged segment list
        """
        if not segments:
            return segments

        merged: list[Segment] = []

        for seg in segments:
            if merged and merged[-1].type == SegmentType.TEXT and seg.type == SegmentType.TEXT:
                # Merge with previous
                prev = merged[-1]
                merged[-1] = Segment(
                    type=SegmentType.TEXT,
                    content=prev.content + seg.content,
                    start=prev.start,
                    end=seg.end,
                )
            else:
                merged.append(seg)

        return merged


def lex_prompt(
    text: str,
    *,
    format_hint: str | None = None,
    config: SegmentConfig | None = None,
) -> list[Segment]:
    """Convenience function to lex a prompt.

    Args:
        text: Input text to parse
        format_hint: Hint for prompt format
        config: Optional segment configuration

    Returns:
        List of Segment objects
    """
    lexer = PromptLexer(config=config, format_hint=format_hint)
    return lexer.lex(text)


def detect_prompt_format(text: str) -> str | None:
    """Attempt to detect the prompt format from text.

    Args:
        text: Input text

    Returns:
        Format name or None if not detected
    """
    # Check for format-specific markers
    format_indicators = {
        "chatml": ["<|im_start|>", "<|im_end|>"],
        "llama": ["<|begin_of_text|>", "<|start_header_id|>"],
        "llama2": ["[INST]", "<<SYS>>"],
        "alpaca": ["### Instruction:", "### Response:"],
        "vicuna": ["USER:", "ASSISTANT:"],
        "mistral": ["[INST]", "[/INST]"],
    }

    for fmt_name, indicators in format_indicators.items():
        if any(ind in text for ind in indicators):
            return fmt_name

    return None
