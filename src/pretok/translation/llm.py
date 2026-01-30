"""LLM-based translator using OpenAI-compatible APIs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pretok.translation.base import BaseTranslator, TranslationError, TranslationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pretok.config import LLMTranslatorConfig

logger = logging.getLogger(__name__)

# Default system prompt for translation
DEFAULT_SYSTEM_PROMPT = """You are a translation engine. Output ONLY the translated text, nothing else.

Rules:
- Translate accurately while preserving meaning and tone
- Keep all formatting: newlines, spaces, punctuation exactly as they appear
- Do NOT add any notes, explanations, comments, or metadata
- Do NOT add phrases like "Here is the translation" or "Note:"
- If the input has leading/trailing whitespace, preserve it in the output
- Output the translation directly, with no preamble"""

# Default user prompt template
DEFAULT_USER_PROMPT = """Translate from {source_language} to {target_language}:

{text}"""

# Default user prompt when source language is unknown
DEFAULT_USER_PROMPT_AUTO = """Translate to {target_language}:

{text}"""


class LLMTranslator(BaseTranslator):
    """Translator using OpenAI-compatible APIs.

    This translator works with any API that implements the OpenAI
    chat completions interface, including:
    - OpenAI API
    - OpenRouter
    - Ollama
    - vLLM
    - LM Studio
    - Azure OpenAI
    - Together AI
    - Groq
    - etc.

    Example:
        >>> from pretok.config import LLMTranslatorConfig
        >>> config = LLMTranslatorConfig(
        ...     base_url="https://api.openai.com/v1",
        ...     model="gpt-4o-mini",
        ...     api_key="sk-xxx"
        ... )
        >>> translator = LLMTranslator(config)
        >>> result = translator.translate("Hello", "zh")
        >>> result.translated_text
        '你好'

    With Ollama:
        >>> config = LLMTranslatorConfig(
        ...     base_url="http://localhost:11434/v1",
        ...     model="llama3.2",
        ... )
        >>> translator = LLMTranslator(config)
    """

    def __init__(self, config: LLMTranslatorConfig) -> None:
        """Initialize LLM translator.

        Args:
            config: LLM translator configuration

        Raises:
            ImportError: If openai package is not installed
            TranslationError: If configuration is invalid
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            msg = (
                "openai package is required for LLMTranslator. Install it with: pip install openai"
            )
            raise ImportError(msg) from e

        self._config = config
        self._api_key = config.get_api_key()

        # Initialize OpenAI client
        client_kwargs: dict[str, Any] = {}

        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        else:
            # Some local APIs don't require an API key
            client_kwargs["api_key"] = "not-required"

        self._client = OpenAI(**client_kwargs)

        # Prompts
        self._system_prompt = config.system_prompt or DEFAULT_SYSTEM_PROMPT
        self._user_prompt_template = config.user_prompt_template

        logger.info(
            "Initialized LLMTranslator with model=%s, base_url=%s",
            config.model,
            config.base_url or "default",
        )

    @property
    def name(self) -> str:
        """Return translator name."""
        return f"llm:{self._config.model}"

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages.

        LLM translators generally support many languages,
        so we return an empty list to indicate "all supported".
        """
        return []

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> TranslationResult:
        """Translate text using LLM.

        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (optional)

        Returns:
            TranslationResult with translated text
        """
        if not text or not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_language=source_language or "unknown",
                target_language=target_language,
                translator=self.name,
            )

        # Build user prompt
        if self._user_prompt_template:
            user_prompt = self._user_prompt_template.format(
                text=text,
                source_language=source_language or "auto-detect",
                target_language=target_language,
            )
        elif source_language:
            user_prompt = DEFAULT_USER_PROMPT.format(
                text=text,
                source_language=self._get_language_name(source_language),
                target_language=self._get_language_name(target_language),
            )
        else:
            user_prompt = DEFAULT_USER_PROMPT_AUTO.format(
                text=text,
                target_language=self._get_language_name(target_language),
            )

        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._config.temperature,
                max_tokens=len(text) * 4,  # Allow for expansion
            )

            translated_text = response.choices[0].message.content or ""

            # Clean up the response - remove common LLM artifacts
            translated_text = self._clean_translation(translated_text, text)

            return TranslationResult(
                source_text=text,
                translated_text=translated_text,
                source_language=source_language or "auto",
                target_language=target_language,
                translator=self.name,
                metadata={
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else 0,
                    },
                },
            )

        except Exception as e:
            raise TranslationError(
                f"LLM translation failed: {e}",
                source_text=text,
                source_language=source_language,
                target_language=target_language,
                translator=self.name,
                cause=e,
            ) from e

    def translate_batch(
        self,
        texts: Sequence[str],
        target_language: str,
        source_language: str | None = None,
    ) -> list[TranslationResult]:
        """Translate multiple texts.

        For LLM translators, we translate texts one by one
        to maintain quality and handle errors individually.
        """
        results = []
        for text in texts:
            try:
                result = self.translate(text, target_language, source_language)
                results.append(result)
            except TranslationError:
                # Return original on failure
                results.append(
                    TranslationResult(
                        source_text=text,
                        translated_text=text,
                        source_language=source_language or "unknown",
                        target_language=target_language,
                        translator=self.name,
                    )
                )
        return results

    def _get_language_name(self, code: str) -> str:
        """Convert language code to full name for prompts.

        Args:
            code: ISO 639-1 language code

        Returns:
            Full language name
        """
        language_names = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "th": "Thai",
            "vi": "Vietnamese",
            "nl": "Dutch",
            "pl": "Polish",
            "tr": "Turkish",
            "he": "Hebrew",
            "id": "Indonesian",
            "cs": "Czech",
            "sv": "Swedish",
            "da": "Danish",
            "fi": "Finnish",
            "no": "Norwegian",
            "el": "Greek",
            "hu": "Hungarian",
            "ro": "Romanian",
            "uk": "Ukrainian",
            "bg": "Bulgarian",
            "hr": "Croatian",
            "sk": "Slovak",
            "sl": "Slovenian",
            "lt": "Lithuanian",
            "lv": "Latvian",
            "et": "Estonian",
            "ms": "Malay",
            "bn": "Bengali",
            "ta": "Tamil",
            "te": "Telugu",
            "mr": "Marathi",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "pa": "Punjabi",
            "fa": "Persian",
            "ur": "Urdu",
            "sw": "Swahili",
            "af": "Afrikaans",
            "ca": "Catalan",
            "eu": "Basque",
            "gl": "Galician",
        }
        return language_names.get(code.lower(), code)

    def _clean_translation(self, translated: str, original: str) -> str:
        """Clean up LLM translation output.

        Removes common artifacts like notes, explanations, and preambles
        that LLMs sometimes add despite instructions.

        Args:
            translated: Raw translation from LLM
            original: Original text (to preserve whitespace patterns)

        Returns:
            Cleaned translation text
        """
        import re

        result = translated

        # Remove common preamble patterns
        preamble_patterns = [
            r"^(?:Here(?:'s| is) (?:the )?translation:?\s*)",
            r"^(?:Translation:?\s*)",
            r"^(?:Translated(?: text)?:?\s*)",
            r"^(?:In English:?\s*)",
        ]
        for pattern in preamble_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        # Remove trailing notes/explanations in parentheses
        # Match patterns like "(Note: ...)" or "(I translated ...)" at the end
        result = re.sub(
            r"\s*\((?:Note|I translated|Translation note|Translator'?s? note)[^)]*\)\s*$",
            "",
            result,
            flags=re.IGNORECASE,
        )

        # Remove trailing notes after newlines
        # Handles cases like "\n\nNote: I translated..."
        result = re.sub(
            r"\n+(?:Note|N\.B\.|PS|P\.S\.):?\s+.*$",
            "",
            result,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Preserve original whitespace structure
        # If original started with newline, ensure result does too
        original_starts_with_newline = original.startswith("\n")
        original_ends_with_newline = original.endswith("\n")

        result = result.strip()

        if original_starts_with_newline and not result.startswith("\n"):
            result = "\n" + result
        if original_ends_with_newline and not result.endswith("\n"):
            result = result + "\n"

        return result
