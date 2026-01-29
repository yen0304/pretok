"""Tests for language detection module."""

from __future__ import annotations

import pytest

from pretok.detection import DetectionError, DetectionResult, LanguageDetector
from pretok.detection.base import BaseDetector


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_create_valid_result(self) -> None:
        """Test creating a valid detection result."""
        result = DetectionResult(
            language="en",
            confidence=0.95,
            detector="test",
        )
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.detector == "test"
        assert result.raw_output is None

    def test_create_with_raw_output(self) -> None:
        """Test creating result with raw output."""
        result = DetectionResult(
            language="zh",
            confidence=0.88,
            detector="test",
            raw_output={"extra": "data"},
        )
        assert result.raw_output == {"extra": "data"}

    def test_invalid_confidence_too_high(self) -> None:
        """Test that confidence > 1.0 raises error."""
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            DetectionResult(
                language="en",
                confidence=1.5,
                detector="test",
            )

    def test_invalid_confidence_negative(self) -> None:
        """Test that negative confidence raises error."""
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            DetectionResult(
                language="en",
                confidence=-0.1,
                detector="test",
            )

    def test_frozen_result(self) -> None:
        """Test that result is immutable."""
        result = DetectionResult(
            language="en",
            confidence=0.9,
            detector="test",
        )
        with pytest.raises(AttributeError):
            result.language = "fr"  # type: ignore[misc]


class TestDetectionError:
    """Tests for DetectionError exception."""

    def test_basic_error(self) -> None:
        """Test basic error message."""
        error = DetectionError("Detection failed")
        assert str(error) == "Detection failed"
        assert error.text is None
        assert error.detector is None

    def test_error_with_context(self) -> None:
        """Test error with context."""
        error = DetectionError(
            "Detection failed",
            text="Some text",
            detector="langdetect",
        )
        assert error.text == "Some text"
        assert error.detector == "langdetect"


class TestLanguageDetectorProtocol:
    """Tests for LanguageDetector protocol."""

    def test_protocol_compliance(self) -> None:
        """Test that a class implementing the protocol is recognized."""

        class MockDetector:
            @property
            def name(self) -> str:
                return "mock"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="en", confidence=1.0, detector="mock")

            def detect_batch(self, texts: list[str]) -> list[DetectionResult]:
                return [self.detect(t) for t in texts]

        detector = MockDetector()
        assert isinstance(detector, LanguageDetector)


class TestBaseDetector:
    """Tests for BaseDetector base class."""

    def test_normalize_language_code_iso_639_1(self) -> None:
        """Test normalization of already valid codes."""

        class TestDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "test"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="en", confidence=1.0, detector="test")

        detector = TestDetector()
        assert detector._normalize_language_code("en") == "en"
        assert detector._normalize_language_code("EN") == "en"
        assert detector._normalize_language_code("  zh  ") == "zh"

    def test_normalize_iso_639_3(self) -> None:
        """Test normalization of ISO 639-3 codes."""

        class TestDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "test"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="en", confidence=1.0, detector="test")

        detector = TestDetector()
        assert detector._normalize_language_code("eng") == "en"
        assert detector._normalize_language_code("zho") == "zh"
        assert detector._normalize_language_code("jpn") == "ja"

    def test_normalize_with_script(self) -> None:
        """Test normalization of codes with script tags."""

        class TestDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "test"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="en", confidence=1.0, detector="test")

        detector = TestDetector()
        assert detector._normalize_language_code("zh-Hans") == "zh"
        assert detector._normalize_language_code("zh-Hant") == "zh"
        assert detector._normalize_language_code("zh_CN") == "zh"


class TestLangDetectDetector:
    """Tests for LangDetect backend."""

    @pytest.fixture
    def detector(self):
        """Create a LangDetect detector."""
        pytest.importorskip("langdetect")
        from pretok.detection.langdetect_backend import LangDetectDetector

        return LangDetectDetector()

    def test_detect_english(self, detector) -> None:
        """Test detecting English text."""
        result = detector.detect(
            "The quick brown fox jumps over the lazy dog. This is a longer sentence "
            "to help the detector correctly identify the language of this text."
        )
        assert result.language == "en"
        assert result.confidence > 0.5
        assert result.detector == "langdetect"

    def test_detect_chinese(self, detector) -> None:
        """Test detecting Chinese text."""
        result = detector.detect("你好，今天天氣很好")
        assert result.language == "zh"
        assert result.confidence > 0.5

    def test_detect_japanese(self, detector) -> None:
        """Test detecting Japanese text."""
        result = detector.detect("こんにちは、お元気ですか")
        assert result.language == "ja"
        assert result.confidence > 0.5

    def test_detect_empty_text_raises(self, detector) -> None:
        """Test that empty text raises error."""
        with pytest.raises(DetectionError, match="empty"):
            detector.detect("")

    def test_detect_whitespace_only_raises(self, detector) -> None:
        """Test that whitespace-only text raises error."""
        with pytest.raises(DetectionError, match="empty"):
            detector.detect("   \n\t   ")

    def test_detect_batch(self, detector) -> None:
        """Test batch detection."""
        texts = [
            "Hello, world!",
            "Bonjour le monde",
            "Hola mundo",
        ]
        results = detector.detect_batch(texts)
        assert len(results) == 3
        assert results[0].language == "en"
        assert results[1].language == "fr"
        assert results[2].language == "es"


class TestCompositeDetector:
    """Tests for Composite detector."""

    @pytest.fixture
    def mock_detectors(self) -> list[BaseDetector]:
        """Create mock detectors for testing."""

        class MockDetectorA(BaseDetector):
            @property
            def name(self) -> str:
                return "mock_a"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="en", confidence=0.9, detector="mock_a")

        class MockDetectorB(BaseDetector):
            @property
            def name(self) -> str:
                return "mock_b"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="en", confidence=0.8, detector="mock_b")

        return [MockDetectorA(), MockDetectorB()]

    def test_voting_unanimous(self, mock_detectors) -> None:
        """Test voting with unanimous agreement."""
        from pretok.detection.composite import CompositeDetector

        detector = CompositeDetector(mock_detectors)
        result = detector.detect("test")
        assert result.language == "en"
        assert result.confidence > 0.8
        assert result.detector == "composite"

    def test_voting_disagreement(self) -> None:
        """Test voting with disagreement."""
        from pretok.detection.composite import CompositeDetector

        class MockDetectorA(BaseDetector):
            @property
            def name(self) -> str:
                return "mock_a"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="en", confidence=0.9, detector="mock_a")

        class MockDetectorB(BaseDetector):
            @property
            def name(self) -> str:
                return "mock_b"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="fr", confidence=0.8, detector="mock_b")

        class MockDetectorC(BaseDetector):
            @property
            def name(self) -> str:
                return "mock_c"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="en", confidence=0.7, detector="mock_c")

        detector = CompositeDetector([MockDetectorA(), MockDetectorB(), MockDetectorC()])
        result = detector.detect("test")
        # English should win with 2 votes
        assert result.language == "en"

    def test_fallback_chain(self) -> None:
        """Test fallback chain strategy."""
        from pretok.config import CompositeDetectorConfig
        from pretok.detection.composite import CompositeDetector

        class FailingDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "failing"

            def detect(self, text: str) -> DetectionResult:
                raise DetectionError("Always fails", detector=self.name)

        class SuccessDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "success"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="de", confidence=0.85, detector="success")

        config = CompositeDetectorConfig(strategy="fallback_chain")
        detector = CompositeDetector(
            [FailingDetector(), SuccessDetector()],
            config=config,
        )
        result = detector.detect("test")
        assert result.language == "de"
        assert result.detector == "success"

    def test_all_fail_raises(self) -> None:
        """Test that error is raised when all detectors fail."""
        from pretok.detection.composite import CompositeDetector

        class FailingDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "failing"

            def detect(self, text: str) -> DetectionResult:
                raise DetectionError("Always fails", detector=self.name)

        detector = CompositeDetector([FailingDetector()])
        with pytest.raises(DetectionError, match="All detectors failed"):
            detector.detect("test")

    def test_empty_detectors_raises(self) -> None:
        """Test that empty detector list raises error."""
        from pretok.detection.composite import CompositeDetector

        with pytest.raises(ValueError, match="At least one detector"):
            CompositeDetector([])


class TestCreateDetector:
    """Tests for create_detector factory function."""

    def test_create_langdetect(self) -> None:
        """Test creating langdetect detector."""
        pytest.importorskip("langdetect")
        from pretok.detection.composite import create_detector
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = create_detector("langdetect")
        assert isinstance(detector, LangDetectDetector)

    def test_unknown_backend_raises(self) -> None:
        """Test that unknown backend raises error."""
        from pretok.detection.composite import create_detector

        with pytest.raises(ValueError, match="Unknown detector backend"):
            create_detector("unknown_backend")
