# Pipeline API

## Pretok

The main pipeline class for processing text.

### Constructor

```python
Pretok(
    config: PretokConfig | None = None,
    *,
    target_language: str | None = None,
    model_id: str | None = None,
    detector: BaseDetector | None = None,
    translator: BaseTranslator | None = None,
    cache: Cache | None = None,
)
```

**Parameters:**

- `config`: Full configuration object
- `target_language`: Target language for translation (overrides config)
- `model_id`: Model ID for capability lookup
- `detector`: Custom detector instance
- `translator`: Custom translator instance
- `cache`: Custom cache instance

### Methods

#### process

```python
def process(self, text: str) -> PipelineResult
```

Process text through the pipeline.

**Parameters:**
- `text`: Input text to process

**Returns:** `PipelineResult` with processed text and metadata

#### detect

```python
def detect(self, text: str) -> DetectionResult
```

Detect language of text.

**Parameters:**
- `text`: Text to analyze

**Returns:** `DetectionResult` with language and confidence

#### translate

```python
def translate(
    self,
    text: str,
    target_language: str,
    source_language: str | None = None,
) -> TranslationResult
```

Translate text to target language.

**Parameters:**
- `text`: Text to translate
- `target_language`: Target language code
- `source_language`: Source language code (optional)

**Returns:** `TranslationResult` with translated text

## PipelineResult

Result from pipeline processing.

```python
@dataclass
class PipelineResult:
    original_text: str
    processed_text: str
    segments: list[Segment]
    detections: list[DetectionResult]
    translations: list[TranslationResult]
    from_cache: bool
    metadata: dict[str, Any]

    @property
    def was_modified(self) -> bool: ...
```

## Factory Function

### create_pretok

```python
def create_pretok(
    model_id: str | None = None,
    target_language: str | None = None,
    **kwargs,
) -> Pretok
```

Factory function to create a Pretok instance.

**Parameters:**
- `model_id`: Model ID for capability lookup
- `target_language`: Target language (overrides model's primary language)
- `**kwargs`: Additional arguments passed to Pretok

**Returns:** Configured `Pretok` instance
