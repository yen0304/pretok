"""Model capability management module."""

from pretok.capability.registry import (
    ModelCapability,
    ModelRegistry,
    get_default_registry,
)

__all__ = [
    "ModelCapability",
    "ModelRegistry",
    "get_default_registry",
]
