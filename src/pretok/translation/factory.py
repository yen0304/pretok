"""Translator factory for creating translator instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pretok.translation.base import BaseTranslator

if TYPE_CHECKING:
    from pretok.config import TranslationConfig


def create_translator(
    backend: str,
    config: TranslationConfig | None = None,
) -> BaseTranslator:
    """Create a translator instance by backend name.

    Args:
        backend: Backend name ('llm', 'huggingface', 'google', 'deepl')
        config: Translation configuration

    Returns:
        Configured translator instance

    Raises:
        ValueError: If unknown backend specified
        ImportError: If required dependencies not installed
    """
    if backend == "llm":
        from pretok.translation.llm import LLMTranslator

        if config is None or config.llm is None:
            raise ValueError("LLM translator requires configuration")

        return LLMTranslator(config.llm)

    elif backend == "huggingface":
        from pretok.translation.huggingface import HuggingFaceTranslator

        if config is None:
            from pretok.config import HuggingFaceTranslatorConfig

            hf_config = HuggingFaceTranslatorConfig()
        else:
            hf_config = config.huggingface

        return HuggingFaceTranslator(hf_config)

    else:
        raise ValueError(f"Unknown translator backend: {backend}")
