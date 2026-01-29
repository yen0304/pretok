"""Segment types and data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SegmentType(Enum):
    """Types of segments in a prompt.

    Different segment types have different handling rules:
    - TEXT: Regular text content that should be translated
    - CODE: Code blocks that should NOT be translated (except comments)
    - JSON: JSON content where only string values may be translated
    - ROLE_MARKER: Chat format markers (e.g., <|user|>, [INST])
    - CONTROL_TOKEN: Special tokens (e.g., <|endoftext|>, <s>)
    - DELIMITER: Format delimiters (e.g., ###, ---)
    - WHITESPACE: Significant whitespace that should be preserved
    - COMMENT: Code comments (may be translated based on config)
    """

    TEXT = auto()
    CODE = auto()
    JSON = auto()
    ROLE_MARKER = auto()
    CONTROL_TOKEN = auto()
    DELIMITER = auto()
    WHITESPACE = auto()
    COMMENT = auto()


@dataclass
class Segment:
    """A segment of text within a prompt.

    Segments represent distinct parts of a prompt that may need
    different handling during translation.

    Attributes:
        type: The type of segment
        content: The text content of the segment
        start: Starting character index in original text
        end: Ending character index in original text
        language: Detected language code (if applicable)
        metadata: Additional segment metadata
        translatable: Whether this segment should be translated
    """

    type: SegmentType
    content: str
    start: int
    end: int
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    translatable: bool = True

    def __post_init__(self) -> None:
        """Set default translatability based on type."""
        # These types are never translated
        non_translatable = {
            SegmentType.CODE,
            SegmentType.ROLE_MARKER,
            SegmentType.CONTROL_TOKEN,
            SegmentType.DELIMITER,
            SegmentType.WHITESPACE,
        }
        if self.type in non_translatable:
            self.translatable = False

    def __len__(self) -> int:
        """Return length of content."""
        return len(self.content)

    @property
    def span(self) -> tuple[int, int]:
        """Return (start, end) tuple."""
        return (self.start, self.end)

    def with_content(self, new_content: str) -> Segment:
        """Create new segment with different content.

        Args:
            new_content: New content string

        Returns:
            New Segment with updated content
        """
        return Segment(
            type=self.type,
            content=new_content,
            start=self.start,
            end=self.end,
            language=self.language,
            metadata=self.metadata.copy(),
            translatable=self.translatable,
        )

    def with_language(self, language: str) -> Segment:
        """Create new segment with detected language.

        Args:
            language: ISO 639-1 language code

        Returns:
            New Segment with language set
        """
        return Segment(
            type=self.type,
            content=self.content,
            start=self.start,
            end=self.end,
            language=language,
            metadata=self.metadata.copy(),
            translatable=self.translatable,
        )


def segments_to_text(segments: list[Segment]) -> str:
    """Reconstruct text from segments.

    Args:
        segments: List of segments in order

    Returns:
        Reconstructed text string
    """
    return "".join(seg.content for seg in segments)
