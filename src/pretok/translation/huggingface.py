"""HuggingFace-based translator using transformers models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pretok.translation.base import BaseTranslator, TranslationError, TranslationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pretok.config import HuggingFaceTranslatorConfig

logger = logging.getLogger(__name__)


# Language code mappings for NLLB models
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "he": "heb_Hebr",
    "id": "ind_Latn",
    "cs": "ces_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "el": "ell_Grek",
    "hu": "hun_Latn",
    "ro": "ron_Latn",
    "uk": "ukr_Cyrl",
    "bg": "bul_Cyrl",
    "hr": "hrv_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "et": "est_Latn",
    "ms": "zsm_Latn",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
    "fa": "pes_Arab",
    "sw": "swh_Latn",
}

# M2M100 uses ISO codes directly
M2M100_LANGUAGE_CODES = {
    "en": "en",
    "zh": "zh",
    "ja": "ja",
    "ko": "ko",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "ar": "ar",
    "hi": "hi",
    "th": "th",
    "vi": "vi",
    "nl": "nl",
    "pl": "pl",
    "tr": "tr",
    "he": "he",
    "id": "id",
    "cs": "cs",
}


class HuggingFaceTranslator(BaseTranslator):
    """Translator using HuggingFace transformers models.

    Supports various translation models including:
    - NLLB (No Language Left Behind) - facebook/nllb-200-*
    - M2M100 - facebook/m2m100_*
    - MarianMT - Helsinki-NLP/opus-mt-*
    - Any Seq2Seq translation model

    Example:
        >>> from pretok.config import HuggingFaceTranslatorConfig
        >>> config = HuggingFaceTranslatorConfig(
        ...     model_name="facebook/nllb-200-distilled-600M"
        ... )
        >>> translator = HuggingFaceTranslator(config)
        >>> result = translator.translate("Hello", "zh")
    """

    def __init__(self, config: HuggingFaceTranslatorConfig) -> None:
        """Initialize HuggingFace translator.

        Args:
            config: HuggingFace translator configuration

        Raises:
            ImportError: If transformers package is not installed
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            msg = (
                "transformers package is required for HuggingFaceTranslator. "
                "Install it with: pip install transformers torch"
            )
            raise ImportError(msg) from e

        self._config = config
        self._model_name = config.model_name

        # Determine device
        self._device = self._get_device(config.device)

        # Load model and tokenizer
        logger.info("Loading HuggingFace model: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._model.to(self._device)

        # Determine language code mapping
        self._language_mapping = self._detect_language_mapping()

        logger.info(
            "Initialized HuggingFaceTranslator with model=%s, device=%s",
            self._model_name,
            self._device,
        )

    def _get_device(self, device_str: str) -> str:
        """Determine the device to use.

        Args:
            device_str: Device configuration string

        Returns:
            Device string for PyTorch
        """
        if device_str != "auto":
            return device_str

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def _detect_language_mapping(self) -> dict[str, str]:
        """Detect the appropriate language code mapping for the model.

        Returns:
            Dictionary mapping ISO codes to model-specific codes
        """
        # Use custom mapping if provided
        if self._config.language_mapping:
            return self._config.language_mapping

        model_lower = self._model_name.lower()

        if "nllb" in model_lower:
            return NLLB_LANGUAGE_CODES
        elif "m2m100" in model_lower:
            return M2M100_LANGUAGE_CODES
        else:
            # Default: assume ISO codes work
            return {}

    @property
    def name(self) -> str:
        """Return translator name."""
        return f"huggingface:{self._model_name}"

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        if self._language_mapping:
            return list(self._language_mapping.keys())
        return []

    def _get_model_lang_code(self, iso_code: str) -> str:
        """Convert ISO language code to model-specific code.

        Args:
            iso_code: ISO 639-1 language code

        Returns:
            Model-specific language code
        """
        return self._language_mapping.get(iso_code.lower(), iso_code)

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> TranslationResult:
        """Translate text using HuggingFace model.

        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (required for most models)

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

        # Convert language codes
        target_code = self._get_model_lang_code(target_language)
        source_code = self._get_model_lang_code(source_language) if source_language else None

        try:
            # Set source language for tokenizer if supported
            if source_code and hasattr(self._tokenizer, "src_lang"):
                self._tokenizer.src_lang = source_code

            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=self._config.max_length,
                truncation=True,
            ).to(self._device)

            # Generate translation
            generated_kwargs: dict[str, Any] = {
                "max_length": self._config.max_length,
                "num_beams": self._config.num_beams,
            }

            # Set target language if supported
            if (
                hasattr(self._tokenizer, "lang_code_to_id")
                and target_code in self._tokenizer.lang_code_to_id
            ):
                generated_kwargs["forced_bos_token_id"] = self._tokenizer.lang_code_to_id[
                    target_code
                ]

            outputs = self._model.generate(**inputs, **generated_kwargs)

            # Decode
            translated_text = self._tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            return TranslationResult(
                source_text=text,
                translated_text=translated_text,
                source_language=source_language or "auto",
                target_language=target_language,
                translator=self.name,
                metadata={
                    "model": self._model_name,
                    "device": self._device,
                },
            )

        except Exception as e:
            raise TranslationError(
                f"HuggingFace translation failed: {e}",
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
        """Translate multiple texts efficiently in batch.

        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code

        Returns:
            List of TranslationResult for each input
        """
        if not texts:
            return []

        target_code = self._get_model_lang_code(target_language)
        source_code = self._get_model_lang_code(source_language) if source_language else None

        try:
            # Set source language for tokenizer if supported
            if source_code and hasattr(self._tokenizer, "src_lang"):
                self._tokenizer.src_lang = source_code

            # Tokenize batch
            inputs = self._tokenizer(
                list(texts),
                return_tensors="pt",
                max_length=self._config.max_length,
                truncation=True,
                padding=True,
            ).to(self._device)

            # Generate translations
            generated_kwargs: dict[str, Any] = {
                "max_length": self._config.max_length,
                "num_beams": self._config.num_beams,
            }

            if (
                hasattr(self._tokenizer, "lang_code_to_id")
                and target_code in self._tokenizer.lang_code_to_id
            ):
                generated_kwargs["forced_bos_token_id"] = self._tokenizer.lang_code_to_id[
                    target_code
                ]

            outputs = self._model.generate(**inputs, **generated_kwargs)

            # Decode all
            translations = self._tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )

            results = []
            for text, translated in zip(texts, translations, strict=False):
                results.append(
                    TranslationResult(
                        source_text=text,
                        translated_text=translated,
                        source_language=source_language or "auto",
                        target_language=target_language,
                        translator=self.name,
                    )
                )

            return results

        except Exception as e:
            # Fall back to individual translation on error
            logger.warning("Batch translation failed, falling back to individual: %s", e)
            return [self.translate(text, target_language, source_language) for text in texts]
