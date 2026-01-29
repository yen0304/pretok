"""FastText backend for language detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pretok.detection import DetectionError, DetectionResult
from pretok.detection.base import BaseDetector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pretok.config import FastTextConfig

logger = logging.getLogger(__name__)


class FastTextDetector(BaseDetector):
    """Language detector using FastText library.

    FastText provides fast and accurate language identification.
    Requires a pretrained language identification model.

    Example:
        >>> detector = FastTextDetector()
        >>> result = detector.detect("Bonjour le monde!")
        >>> result.language
        'fr'
    """

    # Default model URL from Facebook/Meta
    DEFAULT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self, config: FastTextConfig | None = None) -> None:
        """Initialize the FastText detector.

        Args:
            config: Optional configuration for FastText

        Raises:
            ImportError: If fasttext is not installed
        """
        try:
            import fasttext
        except ImportError as e:
            msg = (
                "fasttext is required for FastTextDetector. "
                "Install it with: pip install fasttext-wheel"
            )
            raise ImportError(msg) from e

        self._fasttext = fasttext
        self._config = config
        self._model = None

        # Disable FastText's own warning messages
        fasttext.FastText.eprint = lambda *_args, **_kwargs: None

    @property
    def name(self) -> str:
        """Return detector name."""
        return "fasttext"

    def _ensure_model(self) -> None:
        """Ensure the model is loaded, downloading if necessary."""
        if self._model is not None:
            return

        model_path = self._get_model_path()

        if not model_path.exists():
            self._download_model(model_path)

        self._model = self._fasttext.load_model(str(model_path))
        logger.info("Loaded FastText model from %s", model_path)

    def _get_model_path(self) -> Path:
        """Get the path to the FastText model."""
        if self._config and self._config.model_path:
            return Path(self._config.model_path).expanduser()

        # Default path in user's cache directory
        cache_dir = Path.home() / ".cache" / "pretok" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "lid.176.bin"

    def _download_model(self, path: Path) -> None:
        """Download the FastText language identification model.

        Args:
            path: Path to save the model to
        """
        import urllib.request

        logger.info("Downloading FastText model to %s...", path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(self.DEFAULT_MODEL_URL, path)
            logger.info("FastText model downloaded successfully")
        except Exception as e:
            msg = f"Failed to download FastText model: {e}"
            raise DetectionError(msg, detector=self.name) from e

    def detect(self, text: str) -> DetectionResult:
        """Detect language using FastText.

        Args:
            text: Text to detect language for

        Returns:
            DetectionResult with detected language

        Raises:
            DetectionError: If detection fails
        """
        if not text or not text.strip():
            raise DetectionError(
                "Cannot detect language of empty text",
                text=text,
                detector=self.name,
            )

        self._ensure_model()

        # FastText expects single-line text
        text = text.replace("\n", " ").strip()

        try:
            k = self._config.k if self._config else 1
            threshold = self._config.threshold if self._config else 0.0

            # Model is guaranteed to be loaded after _ensure_model()
            assert self._model is not None
            labels, probs = self._model.predict(text, k=k, threshold=threshold)

            if not labels:
                raise DetectionError(
                    "No language detected",
                    text=text,
                    detector=self.name,
                )

            # FastText returns labels like "__label__en"
            language = self._parse_fasttext_label(labels[0])
            confidence = float(probs[0])

            return DetectionResult(
                language=language,
                confidence=confidence,
                detector=self.name,
                raw_output={
                    "all_labels": list(labels),
                    "all_probs": [float(p) for p in probs],
                },
            )

        except Exception as e:
            if isinstance(e, DetectionError):
                raise
            raise DetectionError(
                f"Language detection failed: {e}",
                text=text,
                detector=self.name,
            ) from e

    def detect_batch(self, texts: Sequence[str]) -> list[DetectionResult]:
        """Detect languages for multiple texts efficiently.

        FastText can process multiple texts more efficiently in batch.

        Args:
            texts: Sequence of texts to detect

        Returns:
            List of DetectionResult for each input text
        """
        if not texts:
            return []

        self._ensure_model()

        results = []
        for text in texts:
            try:
                result = self.detect(text)
                results.append(result)
            except DetectionError:
                # Return unknown for failed detections
                results.append(
                    DetectionResult(
                        language="unknown",
                        confidence=0.0,
                        detector=self.name,
                    )
                )

        return results

    def _parse_fasttext_label(self, label: str) -> str:
        """Parse FastText label format to language code.

        Args:
            label: FastText label like "__label__en"

        Returns:
            Normalized language code
        """
        # Remove __label__ prefix
        code = label.replace("__label__", "")
        return self._normalize_language_code(code)
