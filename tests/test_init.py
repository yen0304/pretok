"""Tests for the pretok package."""

from pretok import __version__


def test_version() -> None:
    """Test that version is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"
