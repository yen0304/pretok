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
            "Guten Tag, wie geht es Ihnen?",  # German - more distinct than Spanish
        ]
        results = detector.detect_batch(texts)
        assert len(results) == 3
        assert results[0].language == "en"
        assert results[1].language == "fr"
        assert results[2].language == "de"


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

    def test_create_langdetect_with_config(self) -> None:
        """Test creating langdetect detector with config."""
        pytest.importorskip("langdetect")
        from pretok.detection.composite import create_detector
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = create_detector("langdetect", {"seed": 42})
        assert isinstance(detector, LangDetectDetector)

    def test_create_composite(self) -> None:
        """Test creating composite detector."""
        pytest.importorskip("langdetect")
        from pretok.detection.composite import CompositeDetector, create_detector

        detector = create_detector("composite", {"detectors": ["langdetect"]})
        assert isinstance(detector, CompositeDetector)

    def test_unknown_backend_raises(self) -> None:
        """Test that unknown backend raises error."""
        from pretok.detection.composite import create_detector

        with pytest.raises(ValueError, match="Unknown detector backend"):
            create_detector("unknown_backend")


class TestCompositeWeightedAverage:
    """Tests for weighted average strategy."""

    def test_weighted_average_basic(self) -> None:
        """Test basic weighted average."""
        from pretok.config import CompositeDetectorConfig
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
                return DetectionResult(language="en", confidence=0.8, detector="mock_b")

        config = CompositeDetectorConfig(
            strategy="weighted_average",
            weights={"mock_a": 2.0, "mock_b": 1.0},
        )
        detector = CompositeDetector([MockDetectorA(), MockDetectorB()], config=config)
        result = detector.detect("test")
        assert result.language == "en"
        assert result.detector == "composite"

    def test_weighted_average_different_languages(self) -> None:
        """Test weighted average with different language votes."""
        from pretok.config import CompositeDetectorConfig
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
                return DetectionResult(language="fr", confidence=0.7, detector="mock_b")

        config = CompositeDetectorConfig(
            strategy="weighted_average",
            weights={"mock_a": 3.0, "mock_b": 1.0},
        )
        detector = CompositeDetector([MockDetectorA(), MockDetectorB()], config=config)
        result = detector.detect("test")
        # mock_a has higher weight, so en should win
        assert result.language == "en"

    def test_weighted_average_all_fail(self) -> None:
        """Test weighted average when all detectors fail."""
        from pretok.config import CompositeDetectorConfig
        from pretok.detection.composite import CompositeDetector

        class FailingDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "failing"

            def detect(self, text: str) -> DetectionResult:
                raise DetectionError("Always fails", detector=self.name)

        config = CompositeDetectorConfig(strategy="weighted_average")
        detector = CompositeDetector([FailingDetector()], config=config)
        with pytest.raises(DetectionError, match="All detectors failed"):
            detector.detect("test")


class TestCompositeDetectorProperty:
    """Tests for CompositeDetector properties."""

    def test_detectors_property(self) -> None:
        """Test detectors property returns list of backends."""
        from pretok.detection.composite import CompositeDetector

        class MockDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "mock"

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(language="en", confidence=0.9, detector="mock")

        mock = MockDetector()
        detector = CompositeDetector([mock])
        assert detector.detectors == [mock]

    def test_fallback_chain_all_fail(self) -> None:
        """Test fallback chain when all detectors fail."""
        from pretok.config import CompositeDetectorConfig
        from pretok.detection.composite import CompositeDetector

        class FailingDetector(BaseDetector):
            @property
            def name(self) -> str:
                return "failing"

            def detect(self, text: str) -> DetectionResult:
                raise DetectionError("Always fails", detector=self.name)

        config = CompositeDetectorConfig(strategy="fallback_chain")
        detector = CompositeDetector([FailingDetector()], config=config)
        with pytest.raises(DetectionError, match="All detectors in chain failed"):
            detector.detect("test")


class TestLangDetectBackend:
    """Tests for LangDetectDetector backend."""

    def test_langdetect_name(self) -> None:
        """Test langdetect detector name property."""
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = LangDetectDetector()
        assert detector.name == "langdetect"

    def test_langdetect_detect_english(self) -> None:
        """Test detecting English text."""
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = LangDetectDetector()
        result = detector.detect("This is a test sentence in English.")
        assert result.language == "en"
        assert result.confidence > 0.5

    def test_langdetect_detect_chinese(self) -> None:
        """Test detecting Chinese text."""
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = LangDetectDetector()
        result = detector.detect("這是一個中文測試句子")
        assert result.language in ("zh-cn", "zh-tw", "zh")
        assert result.confidence > 0.5

    def test_langdetect_detect_empty_raises_error(self) -> None:
        """Test that empty text raises DetectionError."""
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = LangDetectDetector()
        with pytest.raises(DetectionError, match="empty"):
            detector.detect("")

    def test_langdetect_detect_whitespace_raises_error(self) -> None:
        """Test that whitespace-only text raises DetectionError."""
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = LangDetectDetector()
        with pytest.raises(DetectionError, match="empty"):
            detector.detect("   \n\t  ")

    def test_langdetect_with_config_seed(self) -> None:
        """Test langdetect with seed config for reproducibility."""
        from pretok.config import LangDetectConfig
        from pretok.detection.langdetect_backend import LangDetectDetector

        config = LangDetectConfig(seed=42)
        detector = LangDetectDetector(config=config)
        # Use longer text for reliable detection
        result = detector.detect(
            "This is a longer test sentence in English to ensure proper detection."
        )
        assert result.language == "en"

    def test_langdetect_raw_output_has_all_probs(self) -> None:
        """Test that raw_output contains all_probs."""
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = LangDetectDetector()
        result = detector.detect("Hello world this is a test")
        assert result.raw_output is not None
        assert "all_probs" in result.raw_output
        assert len(result.raw_output["all_probs"]) > 0

    def test_langdetect_detect_with_alternatives(self) -> None:
        """Test detect_with_alternatives method."""
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = LangDetectDetector()
        results = detector.detect_with_alternatives(
            "Bonjour le monde, this is a test",
            top_k=3,
        )
        assert len(results) > 0
        assert len(results) <= 3
        # All results should have language and confidence
        for r in results:
            assert r.language
            assert r.confidence > 0

    def test_langdetect_detect_with_alternatives_empty_text(self) -> None:
        """Test detect_with_alternatives with empty text returns empty list."""
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = LangDetectDetector()
        results = detector.detect_with_alternatives("", top_k=3)
        assert results == []

    def test_langdetect_detect_with_alternatives_whitespace(self) -> None:
        """Test detect_with_alternatives with whitespace returns empty list."""
        from pretok.detection.langdetect_backend import LangDetectDetector

        detector = LangDetectDetector()
        results = detector.detect_with_alternatives("   \n ", top_k=3)
        assert results == []
