# Language Detection

pretok supports multiple language detection backends.

## Available Detectors

### LangDetect (Default)

Pure Python implementation, no external dependencies.

```python
from pretok.detection import LangDetectDetector

detector = LangDetectDetector(seed=42)
result = detector.detect("Bonjour le monde")
print(result.language)  # "fr"
print(result.confidence)  # 0.99
```

### FastText

High accuracy, requires FastText model file.

```python
from pretok.detection import FastTextDetector

detector = FastTextDetector(model_path="/path/to/lid.176.bin")
result = detector.detect("Hello world")
```

### Composite Detector

Combine multiple detectors for better accuracy.

```python
from pretok.detection import CompositeDetector

detector = CompositeDetector(
    detectors=["fasttext", "langdetect"],
    strategy="voting",
)
```

## Detection Result

```python
@dataclass
class DetectionResult:
    language: str        # ISO 639-1 code
    confidence: float    # 0.0 to 1.0
    alternatives: list[tuple[str, float]]
```

## Batch Detection

```python
results = detector.detect_batch([
    "Hello world",
    "Bonjour le monde",
    "Hallo Welt",
])
```

## Language Utilities

```python
from pretok.utils.languages import (
    to_iso639_1,
    to_iso639_3,
    get_language_name,
)

print(to_iso639_3("en"))  # "eng"
print(get_language_name("fr"))  # "French"
```
