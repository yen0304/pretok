"""Tests for the pretok package."""

from pretok import __version__


def test_version() -> None:
    """Test that version is defined and follows semantic versioning."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    # Check semantic versioning format (e.g., "0.1.0", "1.2.3")
    parts = __version__.split(".")
    assert len(parts) >= 2, "Version should have at least major.minor"
    assert all(part.isdigit() for part in parts[:2]), "Major and minor should be numeric"
